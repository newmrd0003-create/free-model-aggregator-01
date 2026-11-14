// The client wires and the upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. The entry forwards both to a provider
// on any of the three upstream wires (openai, anthropic, gemini) and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to an openai provider and
//     streams the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//   - the Gemini path rewrites either client body into a generateContent request (systemInstruction,
//     contents, functionDeclarations, generationConfig) and rewrites the candidate answer into either
//     client shape, JSON or SSE, with usageMetadata carried through exactly.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// The gemini and anthropic upstream wires are implemented below (main.ts no longer answers 501 for them):
// request builders for every client->provider wire pair, response translators for every provider->client
// pair, and SSE translators for the streamed pairs. Every translator is a pure function of its arguments:
// no clock, no randomness, fixed created timestamps, so a replay produces the same bytes.

/** One Anthropic content block as a Gemini part: text, inline image, or tool use/result. */
function geminiPartsFromAnthropicContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const s = block.source;
      if (s.type === 'base64' && typeof s.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
      } else if (s.type === 'url' && typeof s.url === 'string') parts.push({ fileData: { fileUri: s.url } });
    } else if (block.type === 'tool_use') {
      parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    } else if (block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          response: { result: toolResultText(block.content) },
        },
      });
    }
  }
  return parts;
}

/** One OpenAI content part as a Gemini part: text or an inline image (data URL kept byte-identical). */
function geminiPartsFromOpenAIContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const m = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** OpenAI function tools as Gemini functionDeclarations. */
function geminiToolsFromOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const decls = tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiToolsFromAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const decls = tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

/** A Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const m = /^data:([^;]+);base64,(.*)$/.exec(url);
          if (m) parts.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          else parts.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTok =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (typeof maxTok === 'number') out.max_tokens = maxTok;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Chat Completions request as a Gemini generateContent request (model goes in the URL, not the body). */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      systemParts.push(...geminiPartsFromOpenAIContent(raw.content));
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromOpenAIContent(raw.content);
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
              response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const parts = geminiPartsFromOpenAIContent(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const tools = geminiToolsFromOpenAI(body.tools);
  if (tools.length > 0) out.tools = tools;
  const gc: Json = {};
  const maxTok =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (typeof maxTok === 'number') gc.maxOutputTokens = maxTok;
  if (typeof body.temperature === 'number') gc.temperature = body.temperature;
  if (typeof body.top_p === 'number') gc.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) gc.stopSequences = body.stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    gc.responseMimeType = 'application/json';
  }
  if (Object.keys(gc).length > 0) out.generationConfig = gc;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request (model goes in the URL). */
export function anthropicToGeminiRequest(body: Json): Json {
  const systemParts = geminiPartsFromAnthropicContent(body.system);
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsFromAnthropicContent(raw.content);
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const tools = geminiToolsFromAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const gc: Json = {};
  if (typeof body.max_tokens === 'number') gc.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') gc.temperature = body.temperature;
  if (typeof body.top_p === 'number') gc.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) gc.stopSequences = body.stop_sequences;
  if (Object.keys(gc).length > 0) out.generationConfig = gc;
  return out;
}

/** The first candidate of a Gemini response, or null. */
function geminiCandidate(g: Json): Json | null {
  const cands = Array.isArray(g.candidates) ? g.candidates : [];
  return isRecord(cands[0]) ? cands[0] : null;
}

/** A Gemini candidate's text parts joined. */
function geminiTextOfCandidate(cand: Json | null): string {
  if (cand === null) return '';
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini response's usage as input/output token counts. */
function geminiUsageOf(g: Json): { input: number; output: number } {
  const u = isRecord(g.usageMetadata) ? g.usageMetadata : {};
  return { input: numberOrZero(u.promptTokenCount), output: numberOrZero(u.candidatesTokenCount) };
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiStopToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function geminiStopToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini answer as a Chat Completions object naming `model`. */
export function geminiToOpenAIResponse(g: Json, model: string): Json {
  const cand = geminiCandidate(g);
  const text = geminiTextOfCandidate(cand);
  const usage = geminiUsageOf(g);
  const toolCalls: Json[] = [];
  const content = cand !== null && isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  parts.forEach((p, i) => {
    if (!isRecord(p) || !isRecord(p.functionCall)) return;
    toolCalls.push({
      id: `call_${i}`,
      type: 'function',
      function: {
        name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '',
        arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}),
      },
    });
  });
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: geminiStopToOpenAI(cand !== null ? cand.finishReason : undefined),
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropicResponse(g: Json, model: string): Json {
  const cand = geminiCandidate(g);
  const text = geminiTextOfCandidate(cand);
  const usage = geminiUsageOf(g);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  const inner = cand !== null && isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(inner.parts) ? inner.parts : [];
  parts.forEach((p, i) => {
    if (!isRecord(p) || !isRecord(p.functionCall)) return;
    content.push({
      type: 'tool_use',
      id: `toolu_${i}`,
      name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '',
      input: isRecord(p.functionCall.args) ? p.functionCall.args : {},
    });
  });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiStopToAnthropic(cand !== null ? cand.finishReason : undefined),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as a Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(a: Json, model: string): Json {
  const blocks = Array.isArray(a.content) ? a.content : [];
  const text = blocks.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const b of blocks) {
    if (!isRecord(b) || b.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof b.id === 'string' ? b.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof b.name === 'string' ? b.name : '', arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) },
    });
  }
  const usage = isRecord(a.usage) ? a.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(a.stop_reason),
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One SSE stream's frames as {event, data} pairs; `data:` lines are joined per frame. */
function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const datas: string[] = [];
    for (const line of raw.split('\n')) {
      const t = line.trim();
      if (t.startsWith('event:')) event = t.slice(6).trim();
      else if (t.startsWith('data:')) datas.push(t.slice(5).trim());
    }
    for (const d of datas) frames.push({ event, data: d });
  }
  return frames;
}

/** One Chat Completions chunk frame. */
function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return `data: ${JSON.stringify({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  })}\n\n`;
}

/** A full answer (text, stop, usage) as a Chat Completions SSE stream ending in [DONE]. */
function openAIStreamOf(model: string, text: string, stop: string, input: number, output: number, toolCalls: Json[]): string {
  let out = openAIChunk(model, { role: 'assistant' }, null);
  if (text !== '') out += openAIChunk(model, { content: text }, null);
  for (const call of toolCalls) out += openAIChunk(model, { tool_calls: [call] }, null);
  out += openAIChunk(model, {}, stop);
  out += `data: ${JSON.stringify({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  })}\n\n`;
  out += 'data: [DONE]\n\n';
  return out;
}

/** An Anthropic SSE stream as a Chat Completions SSE stream. */
export function anthropicEventsToOpenAI(sseText: string, model: string): string {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  const toolBlocks = new Map<number, { id: string; name: string; args: string }>();
  for (const f of parseSseFrames(sseText)) {
    if (f.data === '[DONE]') continue;
    let ev: unknown;
    try {
      ev = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(ev)) continue;
    if (ev.type === 'message_start' && isRecord(ev.message) && isRecord(ev.message.usage)) {
      inputTokens = numberOrZero(ev.message.usage.input_tokens);
    } else if (ev.type === 'content_block_start' && isRecord(ev.content_block) && ev.content_block.type === 'tool_use') {
      const index = typeof ev.index === 'number' ? ev.index : toolBlocks.size;
      toolBlocks.set(index, {
        id: typeof ev.content_block.id === 'string' ? ev.content_block.id : `toolu_${index}`,
        name: typeof ev.content_block.name === 'string' ? ev.content_block.name : '',
        args: '',
      });
    } else if (ev.type === 'content_block_delta' && isRecord(ev.delta)) {
      if (ev.delta.type === 'text_delta' && typeof ev.delta.text === 'string') text += ev.delta.text;
      else if (ev.delta.type === 'input_json_delta' && typeof ev.delta.partial_json === 'string') {
        const index = typeof ev.index === 'number' ? ev.index : 0;
        const block = toolBlocks.get(index);
        if (block) block.args += ev.delta.partial_json;
      }
    } else if (ev.type === 'message_delta') {
      if (isRecord(ev.delta) && typeof ev.delta.stop_reason === 'string') stop = anthropicStopToOpenAI(ev.delta.stop_reason);
      if (isRecord(ev.usage)) outputTokens = numberOrZero(ev.usage.output_tokens);
    }
  }
  const toolCalls: Json[] = [...toolBlocks.entries()].map(([index, b]) => ({
    index,
    id: b.id,
    type: 'function',
    function: { name: b.name, arguments: b.args },
  }));
  return openAIStreamOf(model, text, stop, inputTokens, outputTokens, toolCalls);
}

/** A Gemini SSE stream as a Chat Completions SSE stream. */
export function geminiEventsToOpenAI(sseText: string, model: string): string {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'stop';
  for (const f of parseSseFrames(sseText)) {
    if (f.data === '[DONE]') continue;
    let frame: unknown;
    try {
      frame = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(frame)) continue;
    const cand = geminiCandidate(frame);
    text += geminiTextOfCandidate(cand);
    if (cand !== null && typeof cand.finishReason === 'string') stop = geminiStopToOpenAI(cand.finishReason);
    if (isRecord(frame.usageMetadata)) {
      const u = geminiUsageOf(frame);
      inputTokens = u.input;
      outputTokens = u.output;
    }
  }
  return openAIStreamOf(model, text, stop, inputTokens, outputTokens, []);
}

/** A Gemini SSE stream as the Anthropic event sequence. */
export function geminiEventsToAnthropic(sseText: string, model: string): string {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for (const f of parseSseFrames(sseText)) {
    if (f.data === '[DONE]') continue;
    let frame: unknown;
    try {
      frame = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isRecord(frame)) continue;
    const cand = geminiCandidate(frame);
    text += geminiTextOfCandidate(cand);
    if (cand !== null && typeof cand.finishReason === 'string') stop = geminiStopToAnthropic(cand.finishReason);
    if (isRecord(frame.usageMetadata)) {
      const u = geminiUsageOf(frame);
      inputTokens = u.input;
      outputTokens = u.output;
    }
  }
  let out = '';
  out += sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: `msg_${model}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  if (text !== '') {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
