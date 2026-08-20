# ChatbotInn Agent Runtime V2 Contract

`agent-runtime.openapi.json` is the target Stage 0 contract owned by the Python
agent runtime. It describes a reconstructible agent turn: Spring provides durable
conversation and process context, the agent returns structured guest messages and
domain tool calls, and Spring authorizes and executes those calls.

The contract is additive. Existing `/hotel/*` endpoints remain available during
the incremental migration.

The domain tool definitions are owned by the Spring repository in
`contracts/v2/domain-tool-catalog.json`.
