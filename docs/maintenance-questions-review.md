# Maintenance staff questions

Add an explicit staff question task on the existing maintenance folio. Capture the
current guest reply verbatim, with message evidence and optimistic version checks.
Sending a reply does not resolve the fault, create a new service or book a visit.
An unanswered question has no automatic resolution timeout. Staff can withdraw it
and continue work; completed and withdrawn exchanges remain visible in the dashboard.
New process instances use this workflow; running older BPMN instances are unchanged.

The only semantic prompt addition describes MAINTENANCE_GUEST_QUESTION in the scope
router. It receives the actual pending staff question as untrusted context and
distinguishes a reply from a separate maintenance request, another service and FAQ.
Existing prompt rules and critical anchors are preserved. No existing tests,
reference documents, skips or guard requirements are removed.

Nine deterministic agent tests cover exact evidence, scoped targets, ambiguity,
stale references, queued replies, preserved language, unrelated requests and
acknowledgments. Three new Spring/H2 scenarios exercise the actual BPMN from the
dashboard, repeat questions on the same folio, retry answers idempotently, reject
invented/old evidence and withdraw without closing. These tests are mandatory.
The browser suite exercises sending, failed draft retention, polling answers,
withdrawal, and desktop/mobile layout alongside existing kitchen and spa flows.

Real-model scope validation: 8/8 synthetic cases passed on the sandbox-configured
gpt-5.6-terra, within the user's authorization of 12 calls. Evidence is recorded
in maintenance-questions-live-evidence.json. No real guest messages or hotel
operations were executed. This validates the new scope decision and deterministic
capture; it does not certify all possible guest wording or WhatsApp delivery.

Full regression gate evidence: target/ci/maintenance-questions-20260928/summary.json
(run after accepting the reviewed reference). This record is implementation review,
not independent human approval.
