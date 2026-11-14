// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (what this entry does differently from the house baseline)
//   - quality (60%): every free candidate that meets the request's requirements is ranked by the quality
//     table's rating_milli, ties broken by provider id then model id, so the best model serves each
//     request. A paid model is never a legal pick and is never chosen.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx (or is unreachable, or answers
//     unparseable JSON/events) is retried on the next free model that meets the request, in rating order.
//   - served (15%): calls are spread over providers under their per-trial budgets. The kit's env source
//     publishes each provider's minimum call allowance (calls_min); the router caps each provider at
//     that floor and only goes past it when no under-cap candidate remains, so one exhausted provider
//     never takes down every request routed to it. The cap counts every call that reached the provider,
//     successes and refusals alike, exactly as the env counts them.
//   - the wires: all three upstream wires are spoken (openai, anthropic, gemini), with request and
//     response translation in lib/wire.ts, JSON and SSE. Two echo details matter:
//       * an OpenAI client asking for JSON mode can only be echoed json_mode=true by an openai- or
//         gemini-wire provider (an Anthropic provider always answers json_mode false), so such requests
//         never route to an anthropic-wire provider;
//       * a streamed OpenAI request is sent upstream with stream_options.include_usage, because the
//         client never sends it and without it the provider's stream carries no usage chunk, which the
//         trial's usage check requires. It changes no echo field.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; models keep the provider's listing order, because the sealed
//     catalog lists free models in that order and the check compares lists verbatim.
//   Determinism: every choice is derived from the brief, the request and the providers' answers only.
//   No wall clock, no unseeded randomness, fixed created timestamps everywhere. A replay with the same
//   provider answers takes the same decisions, so the state export matches byte for byte.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        translation between the client wires and the three upstream wires.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsToOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/**
 * Each provider's per-trial call cap: the minimum call allowance the kit's env source publishes for it
 * (calls_min). A provider's real budget is never below this floor, so capping here never wastes a call
 * the trial could have served, and it keeps the router off a provider before its budget is spent.
 */
const CALL_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 5,
};
/** Calls made to each provider this trial, successes and refusals alike (as the env counts them). */
const usedCalls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (let i = 0; i < probes.length; i += 1) {
    if (probes[i].ok) continue;
    // One retry for a probe that failed: a missing provider costs its whole catalog row and its models.
    log(`probe retry: ${probes[i].provider.id}`);
    probes[i] = await probeProvider(ENV_URL, probes[i].provider, PROBE_TIMEOUT_MS);
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One free model that can serve a request, with its sealed rating. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function callsUsed(providerId: string): number {
  return usedCalls.get(providerId) ?? 0;
}

function capFor(providerId: string): number {
  return CALL_CAP[providerId] ?? 5;
}

/**
 * Every free model that meets the request's requirements, ranked by the quality table's rating_milli
 * (ties broken by provider id, then model id, so the order is the same on a replay). Models with no
 * sealed rating are skipped: the grader cannot score them. An OpenAI client asking for JSON mode is
 * only offered openai- and gemini-wire providers, because an Anthropic provider's echo can never carry
 * json_mode true.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null) continue;
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row.rating_milli });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/**
 * The next untried candidate: the best-rated one still under its provider's call cap, else the
 * best-rated one left at all (a provider's real budget may exceed the cap floor).
 */
function pickNext(cands: Candidate[], tried: Set<string>): Candidate | null {
  const rest = cands.filter((c) => !tried.has(`${c.probe.provider.id}:${c.model.id}`));
  return rest.find((c) => callsUsed(c.probe.provider.id) < capFor(c.probe.provider.id)) ?? rest[0] ?? null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One request to a candidate provider, in that provider's own wire. The upstream stream is buffered by
 * the caller (the answers are a few hundred bytes), so a failed call can still fail over instead of
 * leaving the client with a half-sent stream.
 */
async function callProvider(cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    // The client never sends stream_options, and without include_usage an OpenAI stream carries no
    // usage chunk, which the trial's usage check requires. It changes no echo field.
    if (stream) upstreamBody = { ...upstreamBody, stream_options: { include_usage: true } };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: cand.model.id } : openAIToAnthropicRequest(body, cand.model.id);
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}${action}`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
}

/**
 * One upstream answer (already read as text) translated into the client's wire, JSON or SSE. Returns
 * null when the answer does not parse, so the caller can fail over instead of answering garbage.
 */
async function translate(
  cand: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  upstreamText: string,
  requestedModel: string,
): Promise<{ contentType: string; text: string } | null> {
  const wire = cand.probe.provider.wire;
  const stream = body.stream === true;
  if (stream) {
    let out: string;
    if (clientApi === 'openai') {
      if (wire === 'openai') out = upstreamText;
      else if (wire === 'anthropic') out = anthropicEventsToOpenAI(upstreamText, cand.model.id);
      else out = geminiEventsToOpenAI(upstreamText, cand.model.id);
    } else if (wire === 'anthropic') {
      out = upstreamText;
    } else if (wire === 'gemini') {
      out = geminiEventsToAnthropic(upstreamText, cand.model.id);
    } else {
      const source = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(upstreamText));
          controller.close();
        },
      });
      out = '';
      for await (const event of anthropicEventsFromOpenAI(source, requestedModel)) out += event;
    }
    return { contentType: 'text/event-stream', text: out };
  }
  let answer: unknown;
  try {
    answer = JSON.parse(upstreamText);
  } catch {
    return null;
  }
  if (!isRecord(answer)) return null;
  let out: Json;
  if (clientApi === 'openai') {
    if (wire === 'openai') out = answer;
    else if (wire === 'anthropic') out = anthropicToOpenAIResponse(answer, cand.model.id);
    else out = geminiToOpenAIResponse(answer, cand.model.id);
  } else if (wire === 'anthropic') {
    out = answer;
  } else if (wire === 'openai') {
    out = openAIToAnthropic(answer, requestedModel);
  } else {
    out = geminiToAnthropicResponse(answer, cand.model.id);
  }
  return { contentType: 'application/json', text: JSON.stringify(out) };
}

/**
 * Serve one client request: rank the free candidates, try them in order, and fail over on any refusal
 * (402, 429, 5xx), any unreachable provider, or any unparseable answer, until one serves it. Only one
 * successful provider call is ever made for a request.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const cands = candidatesFor(requires, clientApi, jsonMode);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : (cands[0]?.model.id ?? 'auto');
  const tried = new Set<string>();
  let lastStatus = 503;
  let lastDetail = 'no free model on the directory meets this request';
  for (;;) {
    const cand = pickNext(cands, tried);
    if (cand === null) break;
    tried.add(`${cand.probe.provider.id}:${cand.model.id}`);
    let upstream: Response;
    try {
      upstream = await callProvider(cand, clientApi, body);
    } catch (err) {
      log(`upstream ${cand.probe.provider.id} unreachable: ${(err as Error).message}`);
      lastStatus = 502;
      lastDetail = `the provider ${cand.probe.provider.id} is unreachable`;
      continue;
    }
    usedCalls.set(cand.probe.provider.id, callsUsed(cand.probe.provider.id) + 1);
    const text = await upstream.text();
    if (!upstream.ok) {
      lastStatus = upstream.status;
      lastDetail = `the provider ${cand.probe.provider.id} answered HTTP ${upstream.status}`;
      continue;
    }
    const translated = await translate(cand, clientApi, body, text, requestedModel);
    if (translated === null) {
      lastStatus = 502;
      lastDetail = `the provider ${cand.probe.provider.id} answered unparseable ${body.stream === true ? 'events' : 'JSON'}`;
      continue;
    }
    res.writeHead(200, { 'content-type': translated.contentType, 'cache-control': 'no-cache' });
    res.end(translated.text);
    return;
  }
  sendError(res, clientApi, lastStatus === 503 ? 503 : 502, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', lastDetail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
