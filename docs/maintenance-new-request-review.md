# Keep new maintenance complaints independent of closed history

A guest selected Maintenance, was asked to describe the problem, and replied
"El ac no funciona". The deployed scope classifier treated that plain complaint
as recurrence because its context contained several closed reports of the same
fault. The planner then asked which old folio was meant instead of starting the
new request. The previous live evaluations did not cover this negative case.

The scope classifier no longer receives closed maintenance issue descriptions.
Old operation details remain available to the planner after the guest explicitly
reports persistence/recurrence or presses a historical resolution button. The
scope instructions explicitly distinguish "not working" from "still not working"
and require the relationship to the prior problem in the current message itself.
The guest may report the same equipment and fault as a new independent request.
An open resolution task keeps its existing decision handling.

This changes the interpretation boundary without adding model calls, a keyword
router, backend changes or a database migration. Existing recurrence confirmation,
ownership, original-input preservation and idempotency safeguards are unchanged.

The new history-independence test fails against the deployed classifier and passes
after this change. Four required Spring/H2 fixtures cover Spanish/English, direct
requests and menu selection, each with two identical closed maintenance issues.
They require one new process and unchanged old operations. Unit tests also retain
explicit recurrence with one or multiple candidate sources.

The prompt review retains previous instructions, anchors, skip allowances and
mandatory tests. Historical issue context is intentionally removed from the scope
prompt; it is not removed from the request or domain state. Real-model evidence
and full paired gate results are recorded separately. No real WhatsApp or BPM
execution is implied by local integration results.

The initial authorized 16-call live evaluation passed 15 cases and exposed a
misclassification of "¿Qué hago si el aire vuelve a fallar?". Additional prompt
instructions distinguish conditional guidance from a report of an actual fault.
The recurrence planner also rejects conflicting follow-up flags on hotel questions
and other unrelated scope kinds. Two additional Spring/H2 cases protect this
boundary. The full gate is rerun against this final change before release.

The separately authorized eight-call follow-up passed all cases with
gpt-5.6-terra, reasoning none: the failed question, conditional paraphrases, ordinary
new requests and actual recurrence. Both runs are preserved, including the failure;
the initial run is not represented as a full pass. Evidence:
`tests/conversation_regression/reports/maintenance-new-request-validation.json`.

The first final integration run exposed an assertion error in the two newly added
hypothetical fixtures: an unknown FAQ correctly creates the existing human-required
FAQ follow-up, but the assertion prohibited every process. The final fixtures
explicitly require that FAQ follow-up, zero MAINTENANCE operations and unchanged
closed maintenance operations. Runtime code did not change after the successful
eight-call live recheck. The failed run is retained alongside the final evidence.

A subsequent unchanged-code run passed all 60 integration conversations but hit
an OptimisticLockingException in the existing backend spa process test
`guestChangesUseASeparateTaskAndCanReturnToStaffMoreThanOnce`. The full gate is
repeated once against identical source fingerprints without adding retries to
individual tests, changing assertions or increasing skip allowances. Both gate
results remain available; test concurrency isolation is a separate concern.

The unchanged-source repeat passed the complete gate: 461 agent unit tests, 588
backend unit tests and all 60 Spring/H2 conversations (213 mandatory turns, zero
pending integration assertions). Source fingerprints match the release candidate.
