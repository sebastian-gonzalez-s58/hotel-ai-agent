# Maintenance recurrence after a closed resolution task

An old NOT_RESOLVED button could start a new maintenance process with only the
button label as the issue. The planner now resolves historical buttons using
persisted backend task ownership, recovers the original issue and asks for a new
confirmation. Existing open resolution tasks retain their ordinary same-folio
completion/follow-up behavior.

The confirmation is bound to the source operation and a fresh token. Cancelled,
replaced, malformed and unknown confirmations cannot start another process.
Resolution confirmed by the guest and closure without confirmation have different
messages. The backend rejects old resolution messages as START_SERVICE evidence,
checks guest/stay ownership and the emitted confirmation, and requires the exact
original input. It locks the source operation and uses a stable recurrence key
to prevent multiple successor operations. The source folio stays closed; the new
operation summary, process variables and durable command result retain its link.

Typed reports use a separate evidence-bound recurrence classification. This field
only permits asking for confirmation, not opening a process. Ambiguous references
ask which maintenance request is meant. Synthetic tests do not establish real
model accuracy. Existing records whose only issue is a button label ask for a
description instead of copying the invalid detail.

Prompt review is additive: the scope prompt receives maintenance state and three
classification fields; existing room-service instructions and all 14 anchors are
retained. All prior mandatory tests, conversation cases and skip allowances stay
unchanged. Implementation review is local, not independent human approval.

Validation: targeted Python tests and Spring/H2 conversations cover ES/EN old
buttons, decline/retry, original issue preservation, one linked successor,
duplicate delivery and closure without guest confirmation. Backend unit tests
cover missing/replaced/used confirmations and cross-guest/stay rejection. The
full paired gate passed: 456 agent unit tests, 588 backend unit tests and 54
Spring/H2 conversations. Existing allowed skips were unchanged. Process execution
and WhatsApp delivery are simulated.

Separately authorized live evaluation passed 12 synthetic Spanish/English cases
with the Render sandbox model, gpt-5.6-terra, and reasoning effort none. It covers
scope classification followed by deterministic recurrence planning; it does not
execute hotel services or validate the whole production conversation. An earlier
12-call evaluation accidentally used the stale local gpt-4.1-mini configuration.
That mismatch was disclosed; the user authorized the additional 12-call run.
The durable report records both runs and the exact paired source fingerprints:
`tests/conversation_regression/reports/maintenance-recurrence-validation.json`.

Scope: maintenance only. Spa reservation lifecycle and front-desk follow-up remain
separate pending changes. No database migration or running-service change is
required for this patch; deployment is a separate step.
