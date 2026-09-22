# SPA initial capture messages

New SPA booking drafts now send the catalog and missing-detail questions as plain text, without a Cancel button. The final booking summary retains Confirm, Change and Cancel. Written cancellation and existing reservation/task management controls retain their current behavior.

The change is a conditional on deterministic interaction rendering. No model prompts, extraction rules, reservation state transitions or backend sources change. Existing SPA tests assert that the catalog and clarification messages have no interaction; existing final-confirmation and task tests retain their controls. No additional OpenAI calls are needed for this presentation change.

The snapshot review preserves all behavior rules, anchors, mandatory case IDs and skip policies. Validation runs through the paired local conversation gate at `target/ci/spa-capture-gate`. This implementation review is not independent human approval, live-model evidence or proof of remote branch protection. The change applies to newly sent messages.
