# Guest booking calendar

An opt-in `WEB_BOOKING` offering opens `OPEN_GUEST_BOOKING_SESSION`. The backend
issues the scoped link. The agent preserves that URL, uses the current guest
language, and explains that staff approval is required. Selecting a service or
saying yes in chat cannot submit a reservation through `START_SERVICE`.

Existing SPA alternatives, cancellation, changes, and status requests retain
their current routes. Disabled offerings keep chat capture. The booking summary
preserves independent room-service drafts. Failures and invalid tool links produce
a bounded response without claiming a reservation or retrying automatically.

The prompt adds six lines; no earlier instructions or anchors were removed.
The schema adds one capability and one tool. The only replaced planner lines
extend the existing language check to both web session types. The prompt reference
was inspected against this scope, including generated response schemas.

Validation: 85 focused offline tests passed, including eight booking contract tests,
existing guest orders, SPA turns, reservation actions and room-service interruptions.
These checks do not validate real-model interpretation. No paid model calls were
made. The complete paired regression gate remains required before release.

Review: Codex implementation review; no independent human approval is claimed.
