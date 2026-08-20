# Agent Runtime V2: Stage 0 Decisions

Status: accepted baseline for `architecture/conversation-v2`.

## Runtime responsibility

The agent runtime is a reasoning and language component. It receives a complete,
bounded turn context and returns a structured turn plan.

It owns:

- Language detection and response formulation.
- Semantic interpretation of one inbound message or process event.
- Selection of relevant active operations and Conversation Tasks.
- Selection and arguments of allowed domain tools.
- Structured extraction, confidence and message evidence.
- Model token usage for the turn.

It does not own:

- The canonical conversation transcript.
- Guest, stay, offering, operation or task persistence.
- Tool authorization or execution.
- FluxNova process transitions.
- WhatsApp delivery or provider credentials.
- Human Task completion.

## Turn protocol

Spring invokes `POST /internal/v2/turns` for one of these triggers:

```text
INBOUND_MESSAGE
CONVERSATION_TASK_CREATED
PROCESS_STATUS_CHANGED
STAFF_MESSAGE_READY
TIMER
TOOL_RESULTS
```

The request contains recent messages, a conversation summary, active operation
snapshots, pending Conversation Tasks, available offerings and an explicit tool
policy. Context supplied by Spring is authoritative.

The agent returns one disposition:

```text
RESPONSE_READY
TOOL_CALLS_REQUIRED
NO_ACTION
HANDOFF_REQUIRED
```

For `TOOL_CALLS_REQUIRED`, Spring validates and executes each call independently,
persists the results and invokes another turn with trigger `TOOL_RESULTS`. Guest
messages that claim an operation succeeded are not delivered before the relevant
tool result succeeds.

## Tool safety

Tool calls are proposals, not direct network calls from model-generated code.
Spring applies hotel, guest, stay, offering, operation and task scope. IDs emitted
by the model never expand the resources visible in the request.

The agent can request domain-level commands such as starting a service, completing
a Conversation Task or executing an advertised operation action. It cannot move a
BPMN token, set arbitrary variables, complete a Human Task or delete a process.

## Concurrent process interpretation

The request may contain multiple active operations and multiple open Conversation
Tasks. The agent may target more than one task in one turn when evidence is clear.

Each mutating tool call includes:

- A target operation or Conversation Task.
- Structured arguments.
- Confidence.
- Evidence message IDs.

When task association is ambiguous, the correct result is a clarification message,
not a low-confidence command.

`focusedConversationTaskId` is a conversational hint. It never hides other pending
tasks and never recreates a single `currentProcess` assumption.

## Memory

Spring supplies bounded recent history and a persisted summary. The agent may use
LangGraph checkpoints or a short-lived cache for execution support, but the turn
must remain reconstructible from the request.

No tool decision may depend exclusively on hidden model memory.

## Idempotency and retries

`agentTurnId` is generated and persisted by Spring before invocation. Retries reuse
the same `Idempotency-Key` and identical body.

The agent endpoint has no business side effects. Spring persists the first accepted
turn plan and executes each tool call using an idempotent command ID scoped to the
turn. Duplicate or late responses cannot execute the same command twice.

## Observability

Every turn carries `agentTurnId`, `traceId`, `conversationId` and a request ID.
Responses include model name, input tokens, cached input tokens, output tokens,
reasoning tokens, total tokens and latency when available.

Model chain-of-thought is never requested, persisted or returned. Evidence refers
only to supplied messages, catalog records, knowledge records, operations and tasks.

## Compatibility

- V2 uses `/internal/v2/turns` and `schemaVersion: "2.0"`.
- Existing `/hotel/tasks` and specialized `/hotel/*` routes remain during migration.
- V2 may reuse current prompts, OpenAI clients, task capabilities and catalog clients.
- Legacy endpoints are removed only after the V2 Maintenance and Room Service
  vertical slices pass end-to-end tests.
