# Multilingual Sandbox

Deploy `codex/sandbox-multilingual` as a NEW Render Docker web service. The branch
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
