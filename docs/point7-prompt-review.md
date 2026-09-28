# Point 7: implementation review of protected inputs

Scope: local sandbox implementation review by Codex. This is not independent human PR approval or deployment authorization.

Compared the proposed snapshot against the accepted SC-010 baseline. Six documents change: new coverage policy, CI gate package and runner, gate rejection tests, two added snapshot-input declarations in the prompt guard, and the offline unit runner's machine-readable report and three explicit live-evaluation flags.

No application source snapshot, effective prompt, prompt option, registered prompt family, behavior rule or conversation fixture changes. No snapshot document is removed. All 14 critical prompt anchors survive. Unit tests retain their existing execution logic and network block; the report adds IDs, skips and failures without masking the process exit status. The policy preserves current SC-010 coverage, including known pending steps and checks; it does not label them as fully passing conversations.

Reviewed rejection behavior: missing mandatory unit test, unexpected skip, failed or expected-failing unit test, duplicate IDs, mandatory conversation NOT_RUN, missing/reordered turns, failed assertion under a PASS label, newly pending assertions, stale source hash, inconsistent totals, missing backend XML and prompt review/baseline mismatch. All 17 gate-specific tests passed before baseline acceptance. The complete paired gate must still run on every candidate; this review never substitutes for fresh CI evidence.

The machine review record hashes the canonical baseline index. Changes to this baseline require an updated record and reviewed evidence. Independent review, required GitHub checks and remote Render activation remain separate, unapplied controls as described in `conversation-ci-gate.md`.
