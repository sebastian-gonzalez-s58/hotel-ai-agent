# Stage 3: outbound process notifications

`POST /internal/v2/localizations` is a V2-only, internal-token-authenticated presentation
endpoint. It also validates the request timestamp, X-Request-Id and Idempotency-Key
(equal to messageId). The body includes:

```json
{
  "messageId": "10000000-0000-0000-0000-000000000001",
  "namespace": "chatbotinn_telware_demo",
  "locale": "fr-CA",
  "texts": [{"text": "Cancelar pedido", "maxLength": 20}],
  "protectedValues": ["REQ-20260908-ABC12345"]
}
```

Response: `{ "locale": "fr-CA", "version": "1", "texts": ["Annuler la commande"] }`.
The order and number of text slots must be preserved. The request forbids extra fields,
including tools or operation commands. It accepts at most 25 text slots and 30,000 source
characters. Canonical task/action IDs never enter this contract.

Known Spanish and English notifications use `process_message_templates.json`. This covers
maintenance confirmation, room-service changes/delivery/cancellation, spa alternatives
and confirmations, and common recovery errors. Notes appended by staff are not treated
as part of a menu URL and remain eligible for translation.

Other locales and dynamic text reuse the stage-2 masked translation batch/cache, with one
four-second model call and no retries. The endpoint's five-second limit includes queue
time. The language model is instructed to translate supplied facts, never to answer the
message or add facts. Spring additionally verifies protected tokens and channel limits.

This endpoint is pure presentation: it does not call the turn planner, start a service,
complete a task, use conversation history or change session language. Its idempotency
header identifies the message for authentication/trace purposes; durable rendering and
delivery idempotency are owned by Spring's outbox, not this in-memory translation cache.

Deploy this Telware agent branch before its corresponding backend. No new environment
variables are needed. Do not deploy it to `cristalino-dev` during this test stage. Spring
hard-gates the new outbox behavior to `chatbotinn_telware_demo`, with an optional
`CHATBOTINN_OUTBOUND_LOCALIZATION_ENABLED=false` switch for newly enqueued messages.

On timeout or validation failure Spring preserves and delivers the original notification,
recording `FALLBACK`. Subsequent delivery retries reuse that saved content. Full universal
language quality is not guaranteed by local tests; test English and another guest language
in Telware, including maintenance's negative confirmation and kitchen/spa change requests.
