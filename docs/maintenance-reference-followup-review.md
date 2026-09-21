# Preserve the maintenance reference selection across turns

After reporting persistence, a guest with several closed maintenance requests was
asked for a reference. The planner did not persist that question. The reference
reply fell through to general issue capture, and describing the issue triggered
the same reference question again. Previous regression tests stopped at the first
clarification and missed this continuation.

The planner now records an awaiting-reference phase and the candidate operation
IDs. A complete reference immediately following the clarification is resolved
case-insensitively against current authorized maintenance snapshots and the saved
candidate set. The next response shows the original issue and asks for fresh
confirmation. It never uses the reference itself as an issue description.

Unknown or ambiguous references get a specific correction prompt; cancel clears
the pending selection. A source that is still active cannot start another report.
Plain new complaints still follow the independent-request path. An unrelated
outbound prompt does not consume a reference as maintenance follow-up. Existing
conversations without saved selection state can recover when the immediately
preceding outbound message exactly matches the old reference question.

Backend ownership, fresh confirmation, original input and idempotency checks
remain unchanged. No prompts, model configuration, backend code or migrations
change. No additional OpenAI calls are needed or performed for this deterministic
state fix. Synthetic unit tests cover Spanish/English continuation, invalid and
corrected references, cancel, legacy recovery, changed/removed/wrong-service
snapshots and new-request boundaries. Two required Spring/H2 conversations cover
report, invalid reference, correct reference, confirmation, persisted source link
and replay without duplication. Previous required coverage and prompt anchors
are retained. Real WhatsApp verification remains a separate user-phone check.

The complete paired local gate passed: 468 agent tests, 588 backend tests and all
62 Spring/H2 conversations (225 required turns, no pending integration assertions).
Source fingerprints are checked again before publishing the release. Evidence is
recorded in tests/conversation_regression/reports/maintenance-reference-validation.json.
