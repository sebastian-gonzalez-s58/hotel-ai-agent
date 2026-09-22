# Mandatory conversation checks (point 7)

Implemented in the local sandbox. No repository rules, remote workflows, Render services or deployments have been changed. Remote enforcement remains an activation step.

## Activation preflight update: single developer

The owner confirmed there is no second developer. For this project, the intended activation uses required PRs, mandatory checks, administrator enforcement and no force pushes, with **zero required external approvals** and no required code-owner approval. The independent-review steps below describe a future multi-developer configuration and must not lock out the sole maintainer. The baseline record is implementation review, not independent human approval.

The existing Git credential works even though the separate GitHub CLI keyring credential is invalid. GitHub confirms admin access to both repositories. The public agent's `sandbox` branch is unprotected and matches local HEAD before these changes. The private backend's branch-protection API returns HTTP 403: `Upgrade to GitHub Pro or make this repository public to enable this feature.` Do not change repository visibility or purchase an upgrade as part of publishing these changes. Full merge enforcement is blocked by that plan limitation.

The older Blueprint names and branch differ from the later local Render change record: the latter identifies `chatbotinn-sandbox-backend` and `chatbotinn-sandbox-agent`, changed to branch `sandbox`. Verify the actual services and current auto-deploy state in Render before merging. Publishing a feature branch/PR does not establish active protection, configure the missing paired checkout secrets/variables, or authorize a deployment.

## What the gate requires

`python -m tests.ci_gate.run --backend ../sandbox-backend --out target/ci/RUN_NAME`

Use a new output directory for every run. Add `--maven-offline` locally when dependencies are already cached. On Windows set `REGRESSION_MAVEN_REPOSITORY` to the local Maven cache and put Maven on PATH if necessary. CI uses Python 3.12 and Java 17. Maven initially resolves dependencies online; the Spring conversation runner then uses that same cache offline.

Every invocation runs the prompt contract, all offline agent unit tests, all offline conversations, the complete backend test suite with H2, and the Spring conversation suite. Failures do not prevent independent stages from producing evidence. Missing reports, removed mandatory tests, unexpected skips, failed assertions, missing turns and changed source fingerprints block the final result. Existing report directories cannot be reused.

`contracts/ci/gate.v1.json` freezes required test IDs and the exact current coverage of all 43 conversation cases. Every case is now mandatory in the Spring/H2 stage, with no pending integration step or assertion. Required cases must retain their passing status and executed turns. New cases or coverage upgrades require a reviewed policy update; the runner never learns an allowlist from the run it is judging. Python-only replay still declares eight cases and backend-dependent steps/assertions as gaps. A green deterministic gate does not turn those offline gaps into passing coverage or certify real model behavior.

Only the named opt-in live-model unit tests can be skipped. The backend's opt-in conversation entry point may be skipped in the general suite because it must pass in the separate Spring stage. No OpenAI credentials or live hotel credentials are passed into test processes. The unit runner blocks outbound sockets. Spring uses local H2, a loopback Python bridge, and simulated BPM and message delivery.

Reports include paired Git revisions and source hashes, logs, per-test results, pending coverage and a summary. The full source hash includes build configuration, application code, tests, fixtures, policies and workflows; it also detects changes made during the run. A local dirty checkout is identified by content hashes, not represented as its clean HEAD. Artifacts belong to a specific run and cannot be supplied as substitute input to the gate.

## Prompt changes and reviews

The existing prompt guard requires an explicit diff review before changing the baseline. The gate implementation, policy and tests are now part of that protected snapshot too. `contracts/prompts/review.v1.json` binds a recorded implementation review and its evidence to the exact baseline index. CI checks the record; it never generates or accepts a new baseline.

That JSON is not proof of independent approval. GitHub must enforce a reviewer through branch rules and CODEOWNERS. Changing a prompt, updating its baseline and changing the review record in the same PR still requires independent review. Reviewers must inspect removed instructions, relevant regression coverage and any weakened policy, rather than accepting a regenerated baseline alone.

## Prepared workflows

Both repositories contain `.github/workflows/conversation-gate.yml`, with the stable required check name `conversation-gate`. It runs for every push, pull request, merge queue group and manual invocation, without path filters. Every run checks out both repositories. The paired repository must be configured with an exact 40-character commit SHA; there is no fallback to a moving branch.

Configure these only when activating the remote setup:

| Repository | Required variable | Meaning |
| --- | --- | --- |
| hotel-ai-agent | CONVERSATION_BACKEND_SHA | Reviewed backend commit to test with each agent change |
| chatbotinn_cristalino_dev | CONVERSATION_AGENT_SHA | Reviewed agent commit containing the gate to test with each backend change |

Both require `CONVERSATION_READ_TOKEN`, a narrowly scoped token with contents-read access to the paired repository. Checkout does not persist credentials. A fork PR without that secret fails closed; a maintainer must review it and run it on a trusted repository branch. Do not switch this workflow to `pull_request_target` to expose credentials to fork code.

The workflow has no job-level skip conditions or `continue-on-error`. Its final step rejects a skipped gate or missing successful report. Logs and summaries are uploaded even after failure. No model evaluation, image publishing, deployment or merge is performed.

## Remote activation still required

1. Review and publish both sandbox changes, including the gate, fixtures, Java regression adapter, prompt baseline and review evidence. Current pre-change HEADs do not contain this implementation and must not be used as bootstrap pins. If the configured branch auto-deploys, disable deployment in Render before publishing changes; this document does not authorize publishing.
2. Add an independent maintainer with write access to CODEOWNERS. The initial owner is the known repository owner; they cannot approve their own PR. Configure the read token and the reviewed paired SHAs after those commits exist.
3. Run both workflows on the intended branch and inspect the paired SHAs, full logs and pending coverage. The first green checks establish the check name for branch protection. Coordinated changes must be tested as the exact candidate pair before release; updating a SHA variable alone is not validation. Re-run checks and record both commits whenever changing the pair. Do not deploy an independently advancing pair on the strength of checks against older pins.
4. On each branch that feeds dev, require `conversation-gate` from GitHub Actions, require an up-to-date branch, at least one independent approval and code-owner review, dismiss stale approvals, require review of the latest push, protect administrators, prohibit force pushes/deletion and remove bypass permissions. These settings must be applied remotely; a workflow file alone cannot block a merge.
5. Review and apply the local Render configuration. `render.sandbox.yaml` changes only agent/backend sandbox services to `autoDeployTrigger: checksPass`; frontend is unchanged. Verify the actual remote service and branch use that setting. A different dev service needs its own reviewed setting. Do not assume this sandbox Blueprint controls existing dev. For coordinated cross-repository releases, keep auto-deploy off until the validated pair is ready, then promote that pair together.
6. Prove the remote block with a temporary sandbox PR that removes a protected instruction or skips a required test: the check must be red, merge unavailable and no deployment started. Restore the change, rerun and confirm a green result for the restored pair. Only then consider enforcement active.

Render accepts some neutral/skipped checks as passing, which is why the workflow always produces a decisive gate result. Source: [Render deploy checks](https://render.com/docs/deploys), [Blueprint autoDeployTrigger](https://render.com/docs/blueprint-spec). Repository enforcement: [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Local validation, 2026-09-21

The complete gate passed in `target/ci/point7-first/summary.json`: 414 agent unit tests passed (10 explicitly allowed live-model skips), 566 backend tests passed (the separately executed conversation entry point skipped in the general suite), 28 offline partial passes with 15 NOT_RUN, and 22 Spring/H2 partial passes with 21 NOT_RUN. All 14 prompt anchors survive. The 17 new gate rejection tests pass; a separate in-memory removal of anchor PA-001 returned BLOCKED without editing application files. Paired source hashes still matched after execution. No OpenAI calls were made.

The [validation record](../tests/conversation_regression/reports/point7-validation.json) records the source pair and limitations. Workflow YAML and embedded Python syntax were validated locally. Hosted GitHub execution and remote enforcement have not been tested or activated.

## Expanded coverage validation, 2026-09-21

The complete gate passed in `target/ci/coverage21-gate/summary.json`: 424 agent tests passed (10 allowed live-model skips), 569 backend tests passed (one entry point skipped here and executed in the separate Spring stage), 35 offline partial passes with 8 NOT_RUN, and all 43 Spring/H2 cases passed. All 110 required integration turns executed, with zero pending integration steps or assertions. The previous 21 pending cases are now mandatory. All previously mandatory test IDs and turns remain required, including the additional SC-012-failed-edit closure.

The 43 conversations also passed twice consecutively before the restored closure, which then passed both a targeted run and the full gate. The [validation record](../tests/conversation_regression/reports/coverage21-validation.json) identifies the exact source pair, newly completed cases and remaining limits. Prompt review retained all 14 critical anchors and removed no snapshot documents. No OpenAI calls, remote changes or deployments were made.

## Remaining limits after expansion

This protects the deterministic behavior already encoded in tests. The 43 current Spring adapters are implemented; the [coverage review](coverage21-prompt-review.md) explains the fixture boundaries and specific fixes. This does not guarantee identical model replies, test real WhatsApp/BPM, or solve coordinated deployment automatically. Live model evaluations remain separately authorized and budgeted. Python dependencies still follow requirements.txt rather than a release lockfile; the workflow records tool versions and must be validated on the hosted runner before remote enforcement is declared ready.
