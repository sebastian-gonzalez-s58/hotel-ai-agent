# Multilingual input hardening (sandbox)

This change belongs to the agent's `sandbox` branch. It does not require a database
migration, router change, BPMN deployment, frontend change, or environment variable.

## Consistent interpretation

- The general planner uses the effective `guest.preferredLanguage` chosen by the
  runtime. It does not redetect the output language from a short reply, catalog name,
  or staff text. Changing language preserves pending drafts and operations.
- General planning and order extraction share one quantity policy. Explicit number
  words and unambiguous articles are valid; absent quantities never default to one.
  Multilingual capture does not fall back to the legacy string parser.
- Data for the offering currently being captured or its focused task continues that flow, including
  polite phrases such as "I would like two burgers". An explicitly separate request
  for that same offering remains possible via the classifier's `separateRequest`;
  requests for another offering remain independent.
- New-item restrictions are checked against the original message. A model's KEEP
  label on a newly added item cannot discard those restrictions or reject otherwise
  valid data. KEEP with new restrictions on an existing item remains invalid.
- An update may echo an unchanged stored product or quantity without new evidence:
  these are not new facts. Changed values still require current-message evidence,
  and numeric evidence that contradicts a quantity is rejected. A product quote may
  include its quantity, provided the product itself is an exact substring of that quote.

## Failure handling

Order extraction records a bounded reason code, not guest content or provider payloads:
`AMBIGUOUS_ORDER`, `MISSING_PRODUCT`, `MISSING_QUANTITY`, `COMPLETE_ORDER_REQUIRED`,
`TIMEOUT`, `DEPENDENCY_ERROR`, or `INVALID_RESPONSE`. The turn also returns a warning
with the `ORDER_INPUT_` prefix. Identical extractions in one turn reuse the result and
reason rather than calling the model repeatedly.

Technical failures show a reviewed retry message instead of asking for missing data.
Captured items and delivery location are retained. A failed edit disarms the old
confirmation and enters `INPUT_RETRY`, so an old button cannot confirm an edit that
was never understood. A retry must be parsed successfully and presented for confirmation.
Existing kitchen-change tasks remain open and keep their operation association.

This is not an automatic retry mechanism. The guest is asked to resend the failed
message. An ambiguous or incomplete order still requires clarification.

## Presentation

Room-service clarification, retry, replacement, cancellation and kitchen-change copy
uses reviewed Spanish/English templates. The kitchen menu link comes from the offering
catalog rather than a hardcoded hotel's URL. Localization changes visible text only;
task IDs, operation IDs and button action codes are unchanged.
Optional/null operation references are filtered before placeholder protection, so a
missing folio cannot crash presentation localization.

## Verification

Offline regression suite:

```powershell
python -m unittest discover -s tests
```

Optional real-model checks using synthetic messages only:

```powershell
$env:RUN_LIVE_MULTILINGUAL_EVALS = '1'
python -m unittest tests.test_multilingual_understanding_live
```

Set `OPENAI_API_KEY` securely before running the optional checks. They exercise the
real classifier/extractor and proposed tool calls, but never execute hotel tools,
send WhatsApp messages, or create/update operations. Real WhatsApp end-to-end testing
is still a separate acceptance step.
