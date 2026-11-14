# Build task (for the Jules agent)

Build the entry for make.fun competition free-model-aggregator-01 ("The best free AI model aggregator", $500, entries close 2026-11-16T02:30:00Z).

Read `RULES.txt` fully first — it is the spec. The entry is one server in Node 22 (`main.ts` at repo root, stdlib only, TypeScript runs directly) serving the OpenAI Chat Completions API (`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`), JSON + SSE streaming, plus `GET /v1/models` in the wire the auth header names. It registers port + catalog with the trial env, serves the env's 20 requests with the best free model meeting each request's requirements, and exits 0 when the env says the workload is done. `node main.ts --list` prints the catalog byte-for-byte as registered.

Start from `main.ts` + `lib/providers.ts` + `lib/wire.ts` (the starter: a working baseline worth ~25%, deliberately weak). Its header lists the gaps to fix — that is your work list, in priority order:
1. quality (60%): rank free candidates by rating_milli at the request's requirements, prefer the best. Never pick a paid model.
2. failover (10%): a provider call answering 402, 429 or 5xx must retry on the next free model that meets the request.
3. served (15%): spread calls over providers under their per-trial budgets; one exhausted provider must not take down every request routed to it.
4. wires: build the Anthropic-wire and Gemini-wire upstream requests and their event shapes (starter only does OpenAI-wire upstream; the catalog probe already speaks all three).
5. catalog (5%): complete rows even where a listing hides prices, using the quality table's free flag and capabilities.

Determinism is graded: derive every choice from the brief, the requests and the providers' answers only. No wall clock, no unseeded randomness. The final trial runs twice (live, then replayed with the same provider answers) and must leave the same state export byte-for-byte. A process still running when the 180s trial wall ends scores 0.

Validate locally with `practice/trial-0|1|2` (input env + prompt, reference catalog + ratings) and the kit's practice harness under `tools/env` (env server) and `tools/scorer` (scorer). Do not modify anything under `tools/` — it is the kit's harness, for local validation only. Keep `main.ts` at the repo root, keep the MIT LICENSE, do not commit any API key or secret. When the build is done, run the practice trials end-to-end and report the scores.
