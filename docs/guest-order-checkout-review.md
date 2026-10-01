# Guest web order capture

This opt-in change adds `guestExperience: WEB_ORDER` and `OPEN_GUEST_ORDER_SESSION`. Only configured offerings use web capture; the backend independently rejects agent START_SERVICE for them. Disabled offerings retain the existing capture flow.

The prompt adds seven lines defining this channel and retains the previous instructions and textual anchors. The new deterministic helper presents the exact server-issued link in English or Spanish and never claims submission. Session identity is never inferred by the model. Pending kitchen tasks, delivered-order buttons and status requests retain their original routes. Launch failure produces a bounded response rather than a retry loop.

Review: Codex implementation review, not human approval. Eight focused new offline contract tests passed, plus the existing room/kitchen/spa/interruption regression batch. New test inputs are registered in the prompt manifest. The complete mandatory paired regression suite must pass before release. No paid model calls were made or implied.

First increment: initial guest cart and checkout. Kitchen-requested changes continue in the existing conversation workflow. Native web editing of those tasks and direct bucket uploads are subsequent increments, not claims of this release.
