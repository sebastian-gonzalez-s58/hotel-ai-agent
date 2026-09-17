# Multilingual stages 1 and 2

Enabled by the backend capability marker `trigger.eventPayload.languageContext.version=1`.
No new credentials or required environment variables. Without that marker, existing V2
business behavior is preserved.

## Contract

The backend sends `guest.preferredLanguage` as the effective session locale, plus
`languageContext.source` and `languageContext.explicit`. Only an inbound current message
can establish a `languageDecision` (locale/source/confidence/messageId). The runtime
overwrites any model-proposed metadata. Tool-result turns and buttons inherit language.

Explicit choices are sticky. Confident language detection is integrated into hotel scope
classification, with local greetings for common languages. Changing language alone returns
a confirmation without invoking tools or clearing the pending service draft. Language does
not change original guest evidence or the values used to start/complete processes.

Clear standalone intents establish a language at high confidence without a minimum letter
count, including short greetings and CJK requests. Capture values, task decisions, ambiguous
intents, buttons, OK, digits and emoji inherit the current locale unless explicitly changed.

## Presentation

`app/services/message_templates.json` is the versioned ES/EN registry. Use `template(key,
locale, **parameters)`; missing or unexpected parameters are rejected. Additional locales
and dynamic display content pass through a bounded, schema-constrained translation batch
at the end of a response, outside the tool loop. Known reviewed templates and backend
display overrides are resolved locally.

The backend may supply `inputSchema.x-chatbotinn-localization.variants` as a map of locale
to original display text to approved translation. Only current resource translations are
included. Exact locale then base locale is the lookup order. Existing canonical codes,
schemas, field values and action bindings remain unchanged.

The in-memory cache has a 1,000-entry bound and 24-hour TTL, namespaced by hotel, locale,
template version and masked source. URLs, guest identity, references, numbers and explicit
placeholders are protected. Invalid translation output, excessive text length, dependency
failure or timeout defers presentation with `LOCALIZATION_REQUIRED` and language `mul`.
The backend queues these drafts for mandatory outbox localization before gateway delivery.
Retries retain the draft ID and operation links; they do not repeat tools or service starts.
The backend also recognizes the older `LOCALIZATION_FALLBACK` warning for rolling upgrades.
Deploy the backend consumer before this agent version. Successful reviewed/approved messages
still bypass the extra presentation call. Spanish is not exempt: unreviewed catalog content
is checked even when the hotel and guest both have Spanish as their default language.

One translation request gets at most four seconds and no SDK retries, capped again by the
remaining request budget. This is independent of business model-call retries.

Run `python -m tests.run_offline` for the network-blocked regression suite. Live linguistic
acceptance is separate. The remaining stages migrate process/outbox producers, free-text
parsers, FAQ retrieval and staff UI; this change is not a claim of universal-language E2E readiness.
