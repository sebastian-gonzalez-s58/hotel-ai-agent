# Stage 4: multilingual input understanding

## Deployment boundary

- Target: the Telware agent on `architecture/conversation-v2`.
- No Spring Boot, router, frontend, database, or BPMN changes are required.
- No shared FluxNova definitions are deployed or migrated by this stage.
- The new understanding path requires the existing `trigger.eventPayload.languageContext.version = 1` handshake from stages 1-2. No new environment variable is required.
- Do not merge this stage into `cristalino-dev` until Telware acceptance tests are complete. That branch and its deployment are unchanged.

## Decisions

The existing hotel-scope classification call now also identifies a pure current-step decision: confirm, change, cancel, resolved, not resolved, ambiguous, or none. This is not a second classification call.

Runtime acceptance requires a context reply, confidence of at least 0.9, and evidence equal to the complete original guest message. A fragment inside a negated or conditional sentence is insufficient. The model does not choose a task identifier. Buttons and the request's task/operation references, focus, or a single unambiguous pending task determine the target.

A generic yes with several pending tasks and no target asks which request the guest means. A room-service draft confirmation must not complete a maintenance task. Starting another service while maintenance is open remains allowed; there is no conversation-wide START_SERVICE lock.

## Room service

The new extractor accepts multilingual text, code-switching, number words, and Unicode decimal digits. Its only output is a typed order extraction, not executable tools.

- Full replacement and partial edits are different modes.
- Each product, quantity, and modifier must cite the current original message.
- Product names and modifier clauses remain exact source quotes. They are also protected from the later presentation translator.
- Explicit numeric quantities are checked against the quoted digits; missing, fractional, negative, conflicting, and uncertain quantities are not silently converted to one.
- Partial edits preserve unedited items and fields. Appending a restriction preserves previous restrictions. Replacing all modifiers requires an explicit replacement interpretation.
- Removing an item is not cancelling the operation. An empty extraction cannot implicitly cancel it.
- A kitchen-requested replacement requires a complete new order, not an inferred patch to an old order.
- The confirmation includes modifiers. Long summaries use a full text message instead of truncating them into a WhatsApp button body; very large orders require clarification.
- Order details supplied in the first message are retained while asking for the delivery location. A review and explicit confirmation still precede START_SERVICE.

An extraction is used only above the confidence threshold and after provenance validation. The original inbound message is not rewritten; mutations continue to reference original message IDs. Translation is presentation-only and cannot manufacture evidence.

## SPA

The existing structured extraction now supports any input language instead of a Spanish/English-only prompt. Each field includes confidence and current-message evidence. Treatment names remain original quotes.

Dates are interpreted against the request timestamp in the hotel's time zone. The existing future-date, hotel-local-time, and daylight-saving checks remain in force. Ambiguous dates such as `03/04`, unclear AM/PM times, uncertain treatment choices, altered explicit ISO dates, and unsupported evidence clear the affected field for clarification rather than silently retaining an older value.

Current draft/task confirmation tokens, expected versions, and explicit confirmation remain mandatory. Staff alternatives continue to use the existing ACCEPT/CHANGE/CANCEL contract.

## Failure handling and compatibility

Order and SPA extraction each use a single model call with an at-most-eight-second timeout, reduced when the overall turn budget is nearly exhausted. The extraction calls disable SDK retries. An invalid response or dependency timeout asks for clarification without mutating a process. Order extraction is cached only inside the current turn, so validation cannot repeat the same call or reuse another guest's interpretation. Token usage is counted once.

Requests without the language handshake retain the legacy understanding path. Existing identifiers, tool contracts, operation folios, process notifications, and independently running operations are unchanged.

## Verification

Run `python -m tests.run_offline`. The runner blocks outbound network access.

Stage 4 adds 29 regression tests, including French, German, Portuguese, Japanese, Arabic, Chinese, Spanish/English mixtures, and full-width digits. They cover provenance rejection, numeric drift, preserved negations and restrictions, partial edits, kitchen replacement, bounded failures, direct initial orders, task targeting, parallel services, multilingual SPA capture and confirmation, and presentation masking. The complete suite runs 226 tests: 225 passed and one opt-in live evaluation skipped.

These are deterministic contract and flow tests using model stubs. They do not prove real-model accuracy for every language. Before promoting to Cristalino, use Telware to verify:

1. Order two items in French or Portuguese, change one quantity, and confirm that the other item and all restrictions remain.
2. Have kitchen request changes, choose to edit, and submit a complete replacement in another language.
3. Request SPA using a relative date and then an ambiguous date/time; confirm only after reviewing the clarified schedule.
4. Leave maintenance open, create a room-service or SPA request, and confirm each through its own message.
5. Reply negatively or conditionally to maintenance resolution; ensure it is not marked resolved.
6. Verify folios and notification language through the deployed backend/outbox, since this stage does not alter those services.

Rollback is an agent-only redeploy of the stage-3 commit `9b7b8f8`. No database rollback is needed.
