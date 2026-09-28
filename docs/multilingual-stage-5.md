# Stage 5: multilingual FAQ retrieval and grounding

## Scope

Telware only, on `architecture/conversation-v2`. No Cristalino branch, router,
frontend, database data/schema or BPMN deployment changes. Requires the backend
stage-5 commit and the existing stage-1 language context handshake. No new
environment variables or dependencies.

## Turn flow

1. Keep the guest question in its original language and retain its evidence message ID.
2. Request `SEARCH_KNOWLEDGE` with `offeringCode=FAQ` and `semantic=true`.
3. Spring returns approved, active, available entries from catalogs linked to that offering.
   `retrievalMode=SEMANTIC_V1` and `candidateSetComplete=true` identify the new contract.
   The list is not filtered by Spanish words or Latin characters.
4. The first model call selects a source by meaning and proposes a concise answer in
   the session language, including one brief offer of further help.
5. Deterministic checks require a real source ID, exact source quotes and no new
   numeric/URL literals. A separate model call checks subject, full coverage,
   contradictions across candidates, factual support, negations, language and wording.
6. Start the existing FAQ process using `START_SERVICE`. Automatic resolution carries
   the original approved answer/question/item ID, not the rewritten guest answer.
   Spring rechecks that this source is still approved, linked and unchanged.
7. Only after a successful process-start receipt, send the checked guest-language
   answer linked to the operation, without a visible folio. Do not translate it again.
   Clear the temporary FAQ capture; the conversation and other operations remain open.

Unknown, ambiguous, conflicting, invalid or timed-out retrieval starts the same FAQ
process with `HUMAN_REQUIRED`. Its existing staff task and professional staff-answer
delivery continue unchanged. `FAQ_KNOWLEDGE_STALE` also triggers human resolution,
without repeating the model calls. A verified answer is never sent before the process
start succeeds.

## Evidence and isolation

The verified draft is stored in the conversation summary between tool iterations.
Its fingerprint binds the hotel, inbound message, original query, session language
and selected source. It is not a cross-guest answer cache. Original catalog text and
IDs remain unchanged in process input. Catalog content is data, not instructions.

The backend semantic search is enabled only for application schema
`chatbotinn_telware_demo`. Agents without `semantic=true` keep the legacy contract.
An older backend returns the legacy search result, so deploy Spring first, then the
agent, before acceptance testing. Do not merge these changes into Cristalino yet.

## Bounds and tradeoffs

This is exhaustive, bounded LLM semantic selection, not an embedding/vector index.
It supports the current small hotel FAQ catalogs without new database tables or
external indexing. At most 64 unique approved entries and 40,000 source characters
are sent; an oversized catalog returns an incomplete marker with no candidates and
is handed to staff rather than treating a truncated list as exhaustive.

At most two FAQ model calls, each at most 6 seconds and constrained by the remaining
request budget. No SDK retries for these calls. Model failures fall back to staff;
no model/HTTP call is added inside a Spring database transaction. Each successful
call's token usage is counted once. Evidence quotes are bounded to five short excerpts.

Selection requires confidence >= 0.90 and independent verification >= 0.95, with
all checks passing. These scores are model judgments, not calibrated probabilities
or a guarantee against every hallucination. Keep live language acceptance tests.
Numeric facts retain the approved literal representation (for example `22:00`),
even when surrounding prose changes language.

Structured schemas follow the existing Responses API helper. Schema conformance
does not replace semantic verification: see [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Verification

Run `python -m tests.run_offline`. Outbound sockets and live model evaluations are
disabled. Fixtures cover English, French, German, Portuguese, Japanese, Chinese and
Arabic; wrong subject, partial questions, conflicting sources, invented IDs/quotes,
changed numeric facts/URLs, model failures, stale source fallback, process accounting,
summary binding and no second translation. These fixtures test routing and validation
with stubbed model responses, not live linguistic accuracy.

After both Telware deployments, test on WhatsApp:

1. Ask the pool closing time in English against the approved Spanish pool FAQ.
   Expect only the closing time and a brief help offer, one completed FAQ operation.
2. Ask when the hotel closes. With only pool hours available, expect a staff FAQ task,
   not the pool's hours. Reply from the dashboard in the guest's language.
3. Repeat a supported and unsupported question in French and a non-Latin language.
4. Ask an ambiguous question and a two-part question containing an undocumented fact.
   Expect staff escalation, no fabricated partial answer.
5. Continue into maintenance, room service or spa in the same conversation; existing
   operations must remain independent, with their normal folios and updates.

No live OpenAI, WhatsApp, catalog edits or staff actions were executed by the offline suite.
