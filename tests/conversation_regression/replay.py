"""Sandbox-only replay. Scripted model outputs test orchestration, not prompts."""
import argparse
from collections import Counter, deque
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

from app.agents.v2_turn_planner import plan_v2_turn
from app.core.errors import AgentTimeoutError
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.conversation_regression.library import DATA, ROOT, load_cases, profiles
from tests.conversation_regression.session import ConversationSession, check_observation


class ReplayBoundaryError(AssertionError):
    """Missing/changed script or prohibited external effect, not a product failure."""


class ScriptedModel:
    def __init__(self, replies):
        self.replies = deque(replies)
        self.calls = []

    def __call__(self, prompt, *, purpose=None, **kwargs):
        self.calls.append(purpose)
        if not self.replies:
            raise ReplayBoundaryError(f"Unscripted model call: {purpose}")
        reply = self.replies.popleft()
        if reply.purpose != purpose:
            raise ReplayBoundaryError(f"Expected {reply.purpose}, received {purpose}")
        if reply.error == "TIMEOUT":
            raise AgentTimeoutError("Synthetic regression timeout")
        return OpenAiJsonResult(deepcopy(reply.payload), OpenAiTokenUsage(), "synthetic")

    def assert_consumed(self):
        if self.replies:
            raise ReplayBoundaryError("Unused model replies: " + ", ".join(r.purpose for r in self.replies))


MODEL_MODULES = (
    "app.services.openai_client", "app.agents.v2_scope_router",
    "app.agents.v2_turn_planner", "app.agents.spa_turns", "app.agents.social_opening",
    "app.services.input_understanding", "app.services.localized_content",
    "app.services.faq_grounding",
)


def prohibit_external_effect(*args, **kwargs):
    raise ReplayBoundaryError("External effects are prohibited in offline replay")


def replay_case(case, profile):
    result = {"id": case.id, "scenario_id": case.scenario_id,
              "status": "NOT_RUN", "turns": [],
              "required_levels": case.required_levels}
    if not case.offline_ready:
        result["reason"] = "Requires model scripts and/or a Spring integration adapter"
        return result
    session = ConversationSession(case, profile)
    for step in case.steps:
        if step.event.kind == "backend":
            result.update(status="OFFLINE_PARTIAL_PASS" if result['turns'] else "NOT_RUN", reason="Agent prefix only; backend actions require Spring/H2",
                          not_run_steps=[s.id for s in case.steps[case.steps.index(step):]])
            return result
        model = ScriptedModel(step.model_replies)
        checks = [c for c in step.checks if not c.path.startswith('/backend/')]
        pending = [c.model_dump() for c in step.checks if c.path.startswith('/backend/')]
        turn = {"id": step.id, "checks": len(checks), "reviews_pending": step.review}
        if pending:
            turn['checks_pending'] = pending
        result["turns"].append(turn)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(socket.socket, "connect", prohibit_external_effect))
                stack.enter_context(patch.object(socket.socket, "connect_ex", prohibit_external_effect))
                stack.enter_context(patch("app.services.openai_client.get_openai_client",
                                          side_effect=prohibit_external_effect))
                for name in MODEL_MODULES:
                    module = importlib.import_module(name)
                    if hasattr(module, "call_openai_json_result"):
                        stack.enter_context(patch.object(module, "call_openai_json_result", side_effect=model))
                # Presentation has separate tests. This runner cannot certify translation quality.
                stack.enter_context(patch("app.agents.v2_turn_planner.localize_response",
                                          side_effect=lambda request, response, started_at: response))
                request = session.receive(step.event)
                response = plan_v2_turn(request)
                model.assert_consumed()
                observation = session.accept(response)
            turn["observed"] = observation
            turn["errors"] = check_observation(observation, checks)
            if turn["errors"]:
                result["status"] = "FAILED"
                break
        except ReplayBoundaryError as error:
            result.update(status="HARNESS_ERROR", reason=str(error))
            break
        except Exception as error:
            result.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
            break
        finally:
            turn["model_calls"] = model.calls
    else:
        result["status"] = "OFFLINE_PARTIAL_PASS"
    return result


def source_fingerprint():
    def digest(paths):
        value = hashlib.sha256()
        for path in sorted(paths):
            value.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            value.update(b"\0")
            value.update(path.read_bytes())
        return value.hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {
        "git_head": revision,
        "agent_source_sha256": digest(list((ROOT / "app").rglob("*.py")) + list((ROOT / "app").rglob("*.json"))),
        "library_sha256": digest(list(DATA.rglob("*.json")) + list(DATA.parent.glob("*.py"))),
        "contract_sha256": digest([ROOT / "contracts/behavior/conversation-behavior.v1.json"]),
        "prompt_contract_sha256": digest(
            [p for p in (ROOT / "contracts/prompts").rglob("*") if p.is_file()]
            + list((ROOT / "tests/prompt_contract").glob("*.py"))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "replay"])
    parser.add_argument("--case", default="*", help="Case ID or glob")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    cases = load_cases()
    selected = [case for case in cases if fnmatch.fnmatchcase(case.id, args.case)]
    if not selected:
        parser.error("No matching cases")
    if args.command == "validate":
        print(json.dumps({"cases": len(cases), "scenarios": len({c.scenario_id for c in cases}),
                          "offline_ready": sum(c.offline_ready for c in cases)}, ensure_ascii=False))
        return 0
    available_profiles = profiles()
    results = [replay_case(case, available_profiles[case.profile]) for case in selected]
    counts = dict(Counter(result["status"] for result in results))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "local-sandbox", "contract_version": "1.0.0",
        "mode": "scripted-model", "network": "blocked",
        "source_fingerprint": source_fingerprint(),
        "limitations": [
            "Model outputs are synthetic scripts; prompts and real-model behavior are not evaluated.",
            "Response localization is bypassed; multilingual correctness is not certified.",
            "Domain tools are proposed only; Spring persistence and execution are not evaluated.",
            "OFFLINE_PARTIAL_PASS is not approval for promotion. Review checks remain pending.",
            "Each case stops at its first failure; remaining turns have not run.",
        ],
        "counts": counts, "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    for result in results:
        print(f'{result["status"]}: {result["id"]}' +
              (f' ({result["reason"]})' if "reason" in result else ""))
    if counts.get("FAILED") or counts.get("HARNESS_ERROR"):
        return 1
    return 2 if counts.get("NOT_RUN") else 0


if __name__ == "__main__":
    raise SystemExit(main())
