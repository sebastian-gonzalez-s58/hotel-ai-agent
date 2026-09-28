"""Three explicit evaluation levels; reports never equate missing coverage with success."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from app.agents.v2_turn_planner import plan_v2_turn
from app.core.errors import AgentDependencyError, AgentTimeoutError
from tests.conversation_regression.library import ROOT, load_cases, profiles
from tests.conversation_regression.model_runtime import EvaluationLimit, LiveModel
from tests.conversation_regression.replay import replay_case, source_fingerprint
from tests.conversation_regression.session import ConversationSession, check_observation
from tests.conversation_regression.validation_adapter import validation_case


def live_case(case, profile, model):
    result = {"id": case.id, "scenario_id": case.scenario_id, "level": "live_model",
              "status": "NOT_RUN", "turns": [], "first_failure": None}
    if not case.agent_adapter_ready:
        result["reason"] = "Environment events require a separate adapter"
        return result
    if any(reply.error for step in case.steps for reply in (step.model_replies or [])):
        result["reason"] = "Fault-injection scenario belongs to controlled offline execution"
        return result
    if any(step.event.action == "fail_service_start" for step in case.steps):
        result["reason"] = "Injected service failure belongs to controlled offline/Spring execution"
        return result
    session = ConversationSession(case, profile)
    for step in case.steps:
        entry = {"id": step.id, "reviews_pending": step.review}
        result["turns"].append(entry)
        before = len(model.calls)
        started = time.perf_counter()
        try:
            request = session.receive(step.event)
            entry["request"] = request.model_dump(mode="json")
            with model.activate():
                response = plan_v2_turn(request)
            if model.stop_reason:
                raise EvaluationLimit(model.stop_reason)
            observation = session.accept(response)
            entry["observed"] = observation
            entry["errors"] = check_observation(observation, step.checks)
            # A model/localization dependency failure is not a successful language evaluation.
            errors = [c for c in model.calls[before:] if c["status"] == "ERROR"]
            if errors:
                dependency = any(c["error_type"] in {"AgentDependencyError", "AgentTimeoutError"} for c in errors)
                result.update(status="DEPENDENCY_ERROR" if dependency else "FAILED",
                              reason="Model call failed; see recorded call types")
            elif entry["errors"]:
                result["status"] = "FAILED"
            else:
                continue
        except EvaluationLimit as error:
            result.update(status="BUDGET_EXHAUSTED", reason=str(error))
        except (AgentDependencyError, AgentTimeoutError) as error:
            result.update(status="DEPENDENCY_ERROR", reason=type(error).__name__)
        except Exception as error:
            result.update(status="FAILED", reason=type(error).__name__)
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            entry["model_calls"] = model.calls[before:]
        result["first_failure"] = step.id
        break
    else:
        result["status"] = "LIVE_PARTIAL_PASS"
    return result


def decorate(result, case, repetition, contract):
    result["repetition"] = repetition
    result["rule_ids"] = next(s["ruleIds"] for s in contract["scenarios"] if s["id"] == case.scenario_id)
    result["required_levels"] = case.required_levels
    result.setdefault("first_failure", result["turns"][-1]["id"]
                      if result["status"] in {"FAILED", "HARNESS_ERROR"} and result["turns"] else None)
    return result


def exit_status(results):
    if any(r["status"] in {"FAILED", "HARNESS_ERROR", "DEPENDENCY_ERROR", "BUDGET_EXHAUSTED"} for r in results):
        return 1
    return 2 if not results or any(r["status"] == "NOT_RUN" for r in results) else 0


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites: repetitions and failed runs must remain reviewable.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, choices=["offline", "live_model", "integration"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=ROOT.parent / "sandbox-backend")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.repeat > 10:
        parser.error("repeat must be between 1 and 10")
    if args.report.exists():
        parser.error("Report exists; use a new filename to preserve previous evidence")
    all_cases = load_cases()
    cases = [c for c in all_cases if any(fnmatch.fnmatchcase(c.id, p) for p in (args.case or ["*"]))]
    if not cases:
        parser.error("No matching cases")
    contract = json.loads((ROOT / "contracts/behavior/conversation-behavior.v1.json").read_text(encoding="utf-8"))
    report = {"run_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
              "level": args.level, "source_fingerprint": source_fingerprint(),
              "selected_cases": [c.id for c in cases], "repetitions": args.repeat,
              "limitations": ["Partial results never certify all required levels.",
                              "Offline/live_model tool-result steps simulate empty FAQ lookup and successful service starts; they do not execute backend tools.",
                              "Free-text review criteria require complementary review.",
                              "First failure stops the case; later steps remain unevaluated."],
              "results": []}
    model = None
    try:
        if args.level == "live_model":
            key = os.getenv("REGRESSION_OPENAI_API_KEY")
            if not key or not args.model:
                parser.error("Set REGRESSION_OPENAI_API_KEY and --model explicitly")
            model = LiveModel(key, args.model, args.max_calls, args.max_tokens)
            report["model"] = args.model
            report["model_configuration"] = model.configuration()
            report["limits"] = {"calls": args.max_calls, "tokens_stop_after_call": args.max_tokens,
                                "max_output_tokens_per_call": model.output_limit}
        if args.level == "integration":
            from tests.conversation_regression.integration import run_integration
            report.update(run_integration(args.backend, cases, args.repeat))
            by_id = {case.id: case for case in cases}
            for result in report["results"]:
                if result["id"] in by_id:
                    decorate(result, by_id[result["id"]], result.get("repetition", 1), contract)
        else:
            available = profiles()
            for repetition in range(1, args.repeat + 1):
                for case in cases:
                    if model is None:
                        result = (validation_case(case, available[case.profile])
                                  if len(case.steps) == 1 and case.steps[0].event.action == "submit_invalid_plan"
                                  else replay_case(case, available[case.profile]))
                    else:
                        result = live_case(case, available[case.profile], model)
                    result["level"] = args.level
                    report["results"].append(decorate(result, case, repetition, contract))
        if model:
            report["model_call_count"], report["total_tokens"] = len(model.calls), model.tokens
    finally:
        if model:
            model.close()
    report["counts"] = dict(Counter(r["status"] for r in report["results"]))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(args.report, report)
    print(json.dumps(report["counts"]))
    for result in report["results"]:
        if result["status"] not in {"OFFLINE_PARTIAL_PASS", "LIVE_PARTIAL_PASS", "INTEGRATION_PARTIAL_PASS"}:
            print(f'{result["id"]}: {result["status"]} at {result.get("first_failure")} {result.get("reason", "")}')
    return exit_status(report["results"])


if __name__ == "__main__":
    raise SystemExit(main())
