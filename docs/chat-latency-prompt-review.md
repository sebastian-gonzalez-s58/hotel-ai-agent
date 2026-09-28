# Diagnostic instrumentation review

This is a local implementation review, not an independent human approval.

The deployed agent before this change was 609c6bfaa723e2b11688231b07c90d7a28c47550.
An offline baseline run reproduced the existing prompt snapshot mismatch and the existing
catalog test fixture error. The fixture validated a single AgentMessage as a whole AgentTurnResponse;
it now validates the actual strict AgentMessage contract and retains the same content/action assertions.
No production schema was loosened.

The old reviewed snapshot preceded already-deployed catalog recovery/translation fixes. Those
differences were inspected separately: item removal/cancellation, normalized following order lines,
messageDraftId, and local translation/candidate matching. They are present at the deployed SHA,
not new conversational behavior in this branch.

New production changes add opt-in timing decorators, context propagation, admission/worker timing,
and spans around the existing SDK request/JSON/telemetry operations. Model call arguments, schemas,
templates, rules and timeout/retry policy are unchanged. Diagnostic output never includes prompts
or guest text. Exceptions retain their original behavior; cancelled admission cannot release an
unowned semaphore permit. Async workers explicitly carry isolated request context.

Before refreshing dependency snapshots, the review script requires that all captured effective
prompts and model options exactly match the existing reviewed baseline with diagnostics disabled
and enabled. Critical anchors remain required. The refreshed snapshots retain all sources/tests;
no guard, behavioral rule, mandatory conversation, skip policy or CI assertion is removed.

Final exact-source paired gate: `target/ci/chat-latency-final/summary.json`, PASS.
Agent unit suite: 540 tests, 530 passed and 10 explicit live-model skips. Backend: 672 tests,
671 passed and the separately-run conversation integration suite skipped in the ordinary batch.
All 68 Spring/H2 conversation cases pass with no pending required checks. The full application
also passes with diagnostics enabled. A local HTTP/worker/SDK-boundary probe confirms identical
responses on/off and one simulated model call per request with preserved cross-thread context.
Rendered prompts are a limited offline
contract check and cannot certify live model behavior. This work does not use paid model calls.
