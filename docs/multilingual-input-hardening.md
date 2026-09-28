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

### Menu label limits

An English greeting could be followed by Spanish options because the translation
"Frequently Asked Questions" is 26 characters, exceeding WhatsApp's 24-character
list limit. One invalid label previously discarded all translated fields and was
cached, causing subsequent menus to repeat the fallback.

Localization now supplies a per-field length constraint in both the prompt and the
strict response schema (20 for buttons, 24 for list options). Cache keys include
that constraint and the length of protected parameter expansion. Only validated
translations enter the cache; a rejected field does not discard valid sibling
translations. Provider failures still produce a marked `LOCALIZATION_FALLBACK`
without executing tools again. The translation call retains its four-second budget.

Offering labels use the full catalog name as their translation source, even if the
initial channel draft was truncated. Approved hotel translations take precedence;
the built-in FAQ label has a concise reviewed English version, "Hotel questions".
Notification localization passes the same field limits without changing its API.

### Semantic greetings

Exact greetings still use the fast menu path. Other opening messages, including
typos such as "Hellow" or "Hi there", previously fell into `SOCIAL` and relied on
the general planner to remember to attach an interaction. A separate, bounded
social-opening check now selects the deterministic welcome/menu response for
these messages. It runs only for unmixed SOCIAL messages, never service/task
requests, language changes or exact greetings. Thanks and farewells continue on
their original path. Showing the menu preserves drafts and outstanding operations.
The existing service classifier prompt, schema and action/evidence validation are
unchanged. The new check has a three-second timeout and falls back to ordinary
social handling on provider failure or low confidence.

## Verification

Offline regression suite:

```powershell
python -m unittest discover -s tests
```

Optional real-model checks using synthetic messages only:

```powershell
$env:RUN_LIVE_MULTILINGUAL_EVALS = '1'
python -m unittest tests.test_multilingual_understanding_live
python -m unittest tests.test_menu_localization.LiveMenuLocalizationTest
```

Set `OPENAI_API_KEY` securely before running the optional checks. They exercise the
real classifier/extractor and proposed tool calls, but never execute hotel tools,
send WhatsApp messages, or create/update operations. Real WhatsApp end-to-end testing
is still a separate acceptance step.
