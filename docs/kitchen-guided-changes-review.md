# Kitchen-requested option changes

The sandbox incident REQ-20260928-5BC157E8 requested a Chorizo replacement.
Only a staff note reached the process, so the guest was asked to repeat the
whole order. The guest's response was then rejected because "con salsa roja"
left the category word "salsa" as an unknown modifier.

The dashboard now sends selected order-line indices and option names bound to
the current order and kitchen task. Spring validates them against that order and
catalog, captures the affected option groups, and routes new process instances
directly to a choice task. Existing BPMN instances keep their original flow.
The initial prompt offers available replacements and their supplement prices.
Unavailable selections can be identified even after disabling them in the catalog.

The agent seeds the existing order, replaces only the selected options and keeps
other lines, quantities, preparation/allergy notes and delivery location.
Multi-select groups use a single natural-language answer; single-select groups
can use the existing versioned catalog buttons. The catalog is checked again at
reply and confirmation time. A current confirmation is required for the updated
price. Cancellation requires current guest evidence and closes only that order.
Both agent and backend reject unrelated edits in an option-only change.

The phrase normalization removes category words only when an actual catalog option
has matched; "salsa roja y langosta" still produces an unknown-modifier clarification.
There are no model-prompt changes, new model calls or removed critical anchors.
All effective prompt captures remain identical. The four reviewed snapshot changes
are the new deterministic helper, its planner integration, this bounded normalization
and required CI test IDs. No baseline document, existing test or anchor is removed.

Validation includes the real incident with synthetic ES/EN data, duplicate product
lines, retained notes, multi-select options, optional omission, changed prices,
stale confirmations, invalid targets, unavailable options, cancellation, and the
actual Spring/H2 process from dashboard action to revised kitchen review.
The new regression IDs are mandatory in the gate.

The full backend suite also caught an unnecessary kitchen-context read in other
conversation-task types. The read is now scoped to targeted room-service tasks,
and the original strict delegate tests remain unchanged.

Final exact-source validation: target/ci/kitchen-guided-final2-20260928/summary.json.
Tests make no paid model calls and send no real hotel messages. Live guest wording
and WhatsApp delivery remain a separate check. This is Codex implementation review,
not independent human approval.
