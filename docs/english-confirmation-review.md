# English room-service confirmation regression

Sandbox incident, September 26: the session and request remained English. The runtime
rendered an English order confirmation, but the presentation translator returned Spanish
with valid protected placeholders and the response was still tagged `en`. The earlier
catalog journey tests replaced `localize_response` with an identity function and therefore
missed this final-stage failure. A separate `Hellow` opening displayed a Spanish menu
because greeting refinement recognized an opening without establishing its language.

The new regression fixture uses synthetic catalog data and the real localization pipeline,
with a translator stub that deliberately returns Spanish while preserving placeholders.
It fails on the prior code. The fixed planner exempts only a complete, exact match of its
runtime-rendered English/Spanish confirmation, checked against the current captured state,
including body, title, labels, decision IDs, purpose and associations. A message merely
claiming `language=en` is not trusted. Other locales and altered/model-written responses
retain presentation localization. Legacy configured catalog URLs remain supported.

Standard delivery-location labels use reviewed English/Spanish copy. Hotel-specific place
names use approved display variants when available, otherwise their canonical name is
retained, just like product names, options and guest restrictions. Source values, catalog
IDs, quantities, prices, order confirmation/versioning and tool authorization are unchanged.

The standalone `hellow` alias now establishes English without an interpretation call;
it neither overrides an explicitly selected language nor matches a sentence containing
an actual service request. Greeting during a draft preserves its captured order.

The new tests are mandatory in the CI policy and linked to the prompt manifest. No existing
test, anchor, model instruction or baseline document is removed. The implementation and
template dependency snapshots are reviewed explicitly, not accepted by the deployment
runner. This record is implementation review by Codex, not independent human approval.

The six effective-prompt diffs contain only the previous assistant summary's localized
delivery-location label and generated confirmation tokens; model instruction prefixes and
the rest of the parsed context match. One older exact-text test now expects the reviewed
spelling `Habitación` instead of `Habitacion`, with all its behavior assertions retained.

Validation evidence: 74 focused tests; offline replay of the incident snapshot produces
the English summary and English buttons with no translation call. Final paired gate is
recorded at `target/ci/english-confirmation-final/summary.json`. No paid OpenAI requests or real
hotel operations are used by these tests. Live WhatsApp delivery remains a separate check.
