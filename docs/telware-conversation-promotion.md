# Telware conversation promotion

Promote sandbox agent c7cb3f5 and backend 4dc38f6 to the existing Telware branches. Preserve Telware's contact reassignment protections, dashboard and environment connections. This release includes the SPA BPMN and additive V22 migration; the September 18 rollout note describes an earlier release.

Merge reconciliation retains Telware's deterministic room-service capture continuity when the scope classifier labels a follow-up as a new request. Explicit separate requests keep their independent path. Existing location/item collection and legacy non-multilingual location handling remain covered. Suspended drafts retain the new fresh-confirmation boundary. Old unversioned cancellation buttons remain stale; the previous cancellation test now uses an actually rendered current button, with a separate assertion that the old button cannot cancel the draft.

All sandbox behavior rules, prompt anchors, mandatory cases and skip policies remain intact. Telware-specific continuity tests are additionally mandatory. No model prompt text is changed by this merge reconciliation. The paired gate is run against these integrated sources before publishing. No synthetic OpenAI calls or guest messages are sent as part of deployment checks.

Deploy backend before agent, preserving Telware URLs, credentials, router registration and schemas. Match the agent model/reasoning to the sandbox configuration validated by the user. Verify database/engine isolation and V22 migration state. Record previous deployment commits; do not remove migrations or roll back the backend under the newer agent/BPMN. Preserve automatic deployment settings after the coordinated rollout.

Hosted CI still requires paired-revision/read-token configuration; local passing evidence does not claim remote blocking. Implementation review is not independent human approval.
