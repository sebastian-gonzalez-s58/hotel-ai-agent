# SPA reservation lifecycle review

The previous process ended with `COMPLETED / RESERVATION_CONFIRMED` as soon as staff accepted a booking. This prevented subsequent changes and cancellation, and made old menus appear to describe a new request. A confirmed booking now remains active until staff marks the service completed.

## Intended behavior

- Initial staff acceptance produces `ACTIVE / RESERVATION_CONFIRMED`. Staff completion produces `COMPLETED / SERVICE_COMPLETED` and withdraws actions in the same transaction, without an async interval in which cancellation could still succeed.
- A guest can cancel a pending or confirmed reservation. Cancellation interrupts the process and closes outstanding guest and staff tasks.
- A confirmed reservation keeps its accepted service, date and time while a proposed change awaits staff approval. Pending details are separate. Approval replaces the accepted details; rejection or expiry of guest input restores the accepted reservation.
- Rejection of an initial request is reported as staff rejection, never as guest cancellation.
- Reservation management buttons identify a folio. Before executing a change/cancellation, a new confirmation shows the current details and binds the operation version and a nonce. The backend verifies the actual emitted prompt, current inbound evidence, ownership, action availability, version and idempotency.
- Historical draft/task buttons are resolved from persisted task ownership or a successful start ledger, never by selecting the newest reservation. Cancelled drafts and replaced drafts retain bounded history for specific explanations. Terminal bookings cannot be reactivated by old buttons.
- Written change/cancellation requests use the existing scope classifier and then the same confirmation boundary. Multiple bookings require selection. Draft edits and staff-alternative responses keep their existing handlers. A change request leads to the details step; it does not promise that a requested date is available.

## Compatibility and scope

The backend upgrades only SPA operations owned by the guest and stay of the inbound message. Existing staff-review/guest-response waits migrate to the new definition with their current task and timer. Old `COMPLETED / RESERVATION_CONFIRMED` records resume under the same folio, retaining their old process history and sending no additional confirmation. This does not infer that the service was delivered; the legacy process did not record that event. Cancelled or actually completed services are excluded.

This release wires SPA's `spaReservationProcessV2` template. Other booking offerings need explicit process capabilities and lifecycle mapping before claiming the same support. FAQ and front desk are unchanged. Room service and maintenance remain covered by the full regression suite.

## Review and verification

Existing behavior rules, prompt anchors, mandatory test IDs, allowed skips and conversation requirements are retained. The scope prompt gains existing-booking intent examples; it does not authorize domain actions. Two existing BPM assertions intentionally change from terminal-at-acceptance to active-at-acceptance, with an added assertion that actual service completion is terminal.

New mandatory coverage includes reservation management in Spanish and English, fresh confirmations, version/nonce invalidation, exact operation selection, draft history, backend ownership, duplicate command replay, confirmed-slot preservation, cancellation at different waits, staff rejection, timeout and legacy migration.

Final reports are generated under `target/ci/spa-lifecycle-gate-verified`. Earlier gate attempts were stopped after an outdated mock for status input was found and after review identified an async completion interval; they are not passing evidence. Deterministic replay and Spring/H2 integration use synthetic fixtures and no real hotel actions. Model-driven language interpretation requires a separately authorized live run; local scripted replies do not establish live-model success. WhatsApp and remote deployment are separate verification steps. The local review is an implementation review, not independent human approval or proof that remote branch protection is configured.
