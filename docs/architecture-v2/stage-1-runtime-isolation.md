# ChatbotInn Agent V2: Stage 1 Runtime Isolation

Stage 1 introduces a runtime mode without changing task execution, prompts,
graphs, endpoints or model behavior.

## Configuration

```text
AGENT_RUNTIME_MODE=legacy
AGENT_CONTRACT_VERSION=2.0
```

Supported modes:

| Mode | Purpose |
| --- | --- |
| `legacy` | Run only the current production-compatible agent. This is the default. |
| `shadow` | Reserved for evaluating V2 turns without executing tool calls or sending their output. |
| `v2` | Reserved for the V2 turn planner and structured domain tool contract. |

Stage 1 only validates and reports the mode. Later stages will attach behavior to
`shadow` and `v2`; no code should read `AGENT_RUNTIME_MODE` directly outside the
settings object.

## Operational visibility

`GET /health` and `GET /ready` expose `runtimeMode` and `contractVersion`. This
allows Render health checks and deployment diagnostics to prove which runtime is
active without logging credentials or request content.

Example:

```json
{
  "status": "ok",
  "service": "chatbotinn-agent",
  "version": "0.1.0",
  "environment": "local",
  "runtimeMode": "legacy",
  "contractVersion": "2.0"
}
```

## Rollback

Set `AGENT_RUNTIME_MODE=legacy` and redeploy. The Spring application must also use
`CHATBOTINN_AGENT_RUNTIME=LEGACY`. Because Stage 1 does not persist agent-owned
state, rollback requires no data migration.
