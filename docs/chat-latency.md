# Chat latency diagnostics

Enable `CHAT_LATENCY_ENABLED=true` on the sandbox backend **and** agent. Default is false.
`CHAT_LATENCY_SLOW_MS=1000` marks slow measurements; it is not a request timeout.
Backend `CHAT_LATENCY_DB_ENABLED=false` suppresses repository spans if log volume is excessive.
Disabling diagnostics does not change model, prompts, business rules, retries or timeouts.

Every record starts with `CHAT_LATENCY ` and contains JSON: component, trace_id, span_id,
parent_span_id, stage, started_at (UTC), duration_ms, outcome and slow. There are no raw
prompts, guest messages, names, telephone numbers, catalog arguments or credentials in these records.
Application logs that already existed are outside this guarantee.

## Coverage

| Stage | Meaning |
|---|---|
| http.inbound | Backend request handling, including security/intake and acknowledgement |
| inbound.upstream_age | Age of the router's receivedAt timestamp at backend intake; wall-clock approximation |
| inbound.queue_wait | Time from submit until the conversation turn's worker starts, including serialized earlier turns |
| inbound.turn | Context, all agent iterations, tools and response enqueue; recovered failures retain error outcome |
| backend.Class.method | Spring service boundary including its transaction and nested work |
| repository.Interface.method | Repository boundary; includes pool/DB work there, not a raw SQL profiler |
| http.agent / AgentRuntimeV2Client.execute | Transport to headers / complete call including deserialization |
| http.agent_request | Agent HTTP handling including validation and serialization |
| agent.semaphore_wait / agent.threadpool_wait | Agent concurrency admission / worker thread admission |
| agent.execution_budget / agent.execution | Awaited computation budget / synchronous worker computation |
| agent.classify_scope / resolve_language / understand_order / spa_extract | Individual agent operations |
| catalog.* / knowledge.* / http.backend* | Catalog matching, normalization, knowledge cache and backend reads |
| model.sdk_request | One SDK call, including any SDK retries/backoff; purpose, timeout, token counts and configured retry limit |
| model.parse_json / model.record_usage / model.total | Parsing, existing telemetry and the enclosing model helper |
| agent.tool | One backend domain tool including validation and ledger writes; result status is explicit |
| FluxnovaServiceProcessAdapter.start / process.* | Process start and BPM delegate execution |
| process.age_since_handoff | Cumulative wall age since process start/resume, **not** a pure executor queue timer |
| outbound.age_since_enqueue | Total outbox age including previous attempts and retry backoff |
| outbound.dispatch_lag | Time past next_attempt_at; excludes scheduled backoff and includes batch/head-of-line waiting |
| outbound.attempt / http.router | Delivery attempt, localization and gateway HTTP call |
| chat.backend_to_gateway_accepted | Backend intake to a specific response accepted by the delivery gateway |

The final metric ends at gateway acceptance, **not delivery/read on the guest's phone**.
The shared WhatsApp router, Meta network and device are not instrumented internally by this release.
Multiple responses create multiple acceptance observations; they are not separate inbound turns.
`configured_max_retries` is a limit, not proof that a retry occurred. To identify repeated model
calls inspect model.sdk_request count/purpose within one trace. Server/client intervals include
serialization and network overhead differently; do not call their difference pure network latency.

## Correlation and async work

Backend HTTP clients propagate X-Chat-Latency-Trace and X-Chat-Latency-Parent; agent callbacks
propagate them back. No new headers are required to authenticate, route or validate requests.
Input identifiers are length/character validated. Each request restores its thread/contextvar.
The inbound executor explicitly restores a captured context. Outbox correlation survives retries
and restarts via three nullable columns (backend migration V27). Existing rows remain valid.
BPM start/resume stores opaque diagnostic variables; guest/staff actions refresh them so a new
action is not incorrectly attributed to the original request. Off-mode does not add variables.
Long-lived timer events may retain their initiating trace: process age is not guest response time.

Durations use monotonic clocks inside a process. Persisted/cross-process ages use UTC wall
time and require reasonably synchronized clocks. `kind=interval` marks cumulative/queue intervals.
Spans are **inclusive**; nested work, process handoff ages and client/server measurements overlap.
Never sum every duration to estimate end-to-end latency. A worker can finish after the existing
request timeout; its later span records real work, not a second successful HTTP response.

## Reading a slow conversation

Export backend and agent logs for the same time range, then run:

```text
python tools/analyze_latency.py backend.log agent.log --top 30
python tools/analyze_latency.py backend.log agent.log --trace <trace_id>
```

The report deduplicates spans and shows sample count, distinct traces, p50/p95/max and non-ok
outcomes per stage/purpose/tool. Small samples do not establish a representative percentile.
Start with inbound.queue_wait, context creation, number of model requests, domain tools,
localization and outbound.dispatch_lag. Follow parent_span_id for nesting.

No extra OpenAI requests or real guest messages are required for local validation. Capture real
latency from normal authorized sandbox use. Changing timeouts or model before collecting a trace
would mix diagnosis and optimization, so this release does neither.
