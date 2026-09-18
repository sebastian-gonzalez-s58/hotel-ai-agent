# Multilingual Sandbox

Deploy `sandbox` as the sandbox Render Docker web service. The branch
preserves multilingual stages 1-6. Existing deployment branches are unchanged.

The backend repository's `deploy/sandbox/Prepare-Environment.ps1` generates the
private `agent.env`; `render.sandbox.yaml` describes all three services.

Use `AGENT_RUNTIME_MODE=v2`, `APP_ENV=sandbox`, one worker, and the sandbox Spring URL.
`AGENT_INTERNAL_TOKEN` and `CHATBOTINN_API_INTERNAL_TOKEN` must both match the sandbox
backend's `CHATBOTINN_AI_AGENT_INTERNAL_TOKEN`. Do not use Telware/Cristalino URLs.

Keep the existing OpenAI credential private; the generated env is ignored by Git.
Health endpoint: `/health`. WhatsApp stays on the existing router; this agent has no
direct webhook and does not require a Meta token. Verify live model/WhatsApp behavior
after all three services are deployed; branch creation alone does not test delivery.

## GPT-5.6 Compatibility

Only change the sandbox agent's environment for this experiment:

```dotenv
OPENAI_MODEL=gpt-5.6-terra
OPENAI_REASONING_EFFORT=none
```

Use `gpt-5.6-luna` to compare the lower-cost tier. Set `OPENAI_API_KEY` privately in
Render; never commit the credential. The three `OPENAI_*_COST_PER_MILLION` settings
may all be absent or blank; token accounting still works without cost estimates.

`OPENAI_REASONING_EFFORT` accepts `none`, `low`, `medium`, `high`, `xhigh`, and `max`.
Absent/blank values default to `none`, explicitly overriding the provider's medium
default for these two GPT-5.6 models. Invalid values fail at startup. The shared
client omits sampling parameters for these models. GPT-4.1 keeps `temperature=0`
and receives no reasoning parameter, including when rolling back the model env.

This setting applies to all model calls, including scope classification, extraction,
turn planning, and translation. Start with `none`; compare `low` only after measuring
quality and latency. Higher levels can exceed the existing short translation and
turn deadlines. This change does not relax deadlines or modify prompts, workflows,
database schemas, router configuration, or other deployments.

Before changing the live model, run `python -m tests.run_offline`, then synthetic
Responses API checks for JSON schemas, translated menus, and order changes using
the sandbox credential. Do not execute service tools or send guest notifications.
Rollback: set `OPENAI_MODEL=gpt-4.1-mini`; `OPENAI_REASONING_EFFORT` is ignored for it.

Sources: [Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra),
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

Validation on 2026-09-18: 290 offline tests passed (9 opt-in tests skipped).
Both models accepted strict JSON with `none` and reported zero reasoning tokens.
Terra also passed four synthetic live workflow checks: independent orders, kitchen
replacement, unresolved maintenance, and order capture/edit/confirmation. The menu
and delivery translation completed in 3.86s and 2.37s in single samples, not a
latency guarantee. No workflow tool was executed and no WhatsApp message was sent.
