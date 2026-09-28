# Strict catalog validation — implementation review

Scope: sandbox agent and backend, based on current origin/sandbox. No deployment or catalog-data mutation is performed by this change.

Behavior:
- CATALOG_ITEMS exposes ACTIVE_ITEMS_ONLY in sandbox. The agent resolves guest wording to offered item IDs, canonical names, required options and current BASE price plus option adjustments.
- Ambiguous items and missing variants prompt an explicit, draft-scoped choice. Unknown items/services do not produce orders/reservations. Original wording, quantities, restrictions and notes remain captured.
- Room-service summaries show canonical names, chosen variants, line totals and total. SPA summaries show canonical service/variant, date/time and listed price, subject to staff confirmation.
- A catalog change between summary and confirmation requires a fresh summary. Kitchen-requested replacements also require a current catalog-bound summary and explicit confirmation.
- Spring independently validates starts and task updates against the configured live catalog. Cancellation stays available. Missing IDs, foreign/unavailable options, inconsistent names, quantities or room-service prices fail before process mutation.
- Catalog reads join single-valued configuration and batch collections, without availability caching.

Prompt review:
- Added PR-013 / V2_CATALOG_MATCH with structured output, offered-ID allowlist, exact input evidence and conservative confidence threshold.
- All pre-existing effective prompts are unchanged; no old textual anchor was removed. PA-015 protects the no-substitution instruction.
- Reviewed the 12 changed documents in target/catalog-prompt-review/report.json: new prompt/schema, new tests and registered source dependencies. Existing behavioral contracts and mandatory case requirements were retained; 23 agent tests and 13 backend tests were added to the minimum gate.
- This is an implementation review by Codex, not an independent human approval.

Evidence before the full gate:
- 165 focused existing/new agent tests passed; 23 strict catalog tests passed after the final fixes.
- Backend catalog policy and process-boundary tests passed, including H2 rejection of stale price/unavailable item with no extra operation.
- Batched catalog test reads 25 complete items in at most 10 statements and sees an availability edit on the next read.
- Synthetic live evaluation: gpt-5.6-terra, reasoning none, 22 of 24 authorized calls; 15/15 cases passed. English/Spanish orders, spelling, ambiguity, unknown items/services, restrictions, SPA dates/times and unknown positive extras. No hotel services were created.
- Imported sandbox catalog snapshot: 184 food items / 244 variant configurations and 14 active SPA services / 30 configurations normalized successfully. Draft SPA services remain excluded.

Limits and rollout:
- Full replay/H2 coverage uses the existing synthetic library; strict catalog behavior is additionally required by the new unit and persistence tests. Live synthetic results do not certify every guest phrase or hotel workflow.
- Availability means active catalog/category/item/option state. Scheduling capacity, staff approval and restaurant opening hours remain governed by the hotel's workflows; textual opening hours are not treated as structured inventory.
- Free preparation/restriction notes are preserved. Explicit unknown added extras cause clarification; semantic understanding is still subject to uncertainty.
- The backend flag chatbotinn.catalog.enforce-guest-selections defaults false outside sandbox and true in application-sandbox.yaml. Deploy the paired agent before enabling the backend flag. Existing drafts are revalidated before starting; historical operations can still be cancelled.
- Source reports: target/catalog-live.json, target/catalog-live-spa.json, target/catalog-import-validation.json. Final deterministic gate result is recorded separately in target/ci/catalog-validation-final-20260922.

Final review addendum: an explicit variant clarification removes only standalone obsolete variant notes (e.g. Chico/Grande), preserving allergy and preparation notes. The new mandatory regression verifies this correction. The first gate run predates this final correction and cannot serve as final-source evidence.
Malformed model item/metadata shapes now fail through the normal model-validation error path before any mutation; its regression is mandatory.
The initial integration run identified a missing bean import in the test-only Spring context; the adapter now imports the real catalog policy. Application component scanning already includes it. Final gate is rerun with this corrected adapter.

Final gate: PASS (target/ci/catalog-validation-final-20260922/summary.json). 502 agent tests passed (10 separately opt-in live tests skipped); 614 backend tests passed (the 1 opt-in conversation adapter runs in its dedicated integration stage). All 68 Spring/H2 conversation cases achieved the mandatory integration coverage. Fresh source fingerprints and the reviewed prompt baseline matched. No push or deploy has been performed.

Pre-deployment latency incident (2026-09-22): the supplied sandbox log showed PostgreSQL 55P03 lock timeouts updating SPA service_operations while AgentRuntimeV2ContextFactory held its context transaction open. Hikari later reported the connection returned (unleaked). The compatibility migration had joined the context transaction and locked each SPA operation before loading the complete catalog. It now runs in REQUIRES_NEW; the outer context is read-only. A concurrent H2 regression requires another transaction to acquire the operation lock while the context transaction remains open. Catalog code metadata uses a scalar query instead of rebuilding every linked catalog. Its regression checks one statement, zero loaded entities, active-only filtering and deduplication. Both tests are mandatory. Context and agent durations are now logged separately, without guest text. These changes add no prompt edits; the gate policy snapshot is reviewed again. The earlier gate above predates this incident fix; release evidence must come from target/ci/catalog-latency-final.
