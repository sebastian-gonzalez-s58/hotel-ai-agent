import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.errors import AgentDependencyError
from tests.conversation_regression.evaluate import decorate, exit_status, live_case, write_report
from tests.conversation_regression.integration import Bridge
from tests.conversation_regression.library import ROOT, load_cases, profiles
from tests.conversation_regression.model_runtime import LiveModel, EvaluationLimit
from tests.conversation_regression.validation_adapter import validation_case
from tests.conversation_regression.library import Check
from tests.conversation_regression.session import check_observation
from tests.test_conversation_regression_library import response


class FakeLiveModel:
    def __init__(self):
        self.calls, self.stop_reason = [], None
    @contextmanager
    def activate(self):
        yield self


class ConversationEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c.id: c for c in load_cases()}
        cls.profile = profiles()["hotel"]

    def test_existing_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, {"failed": True})
            with self.assertRaises(FileExistsError):
                write_report(path, {"passed": True})
            self.assertEqual({"failed": True}, json.loads(path.read_text()))

    def test_nested_quantity_boolean_cannot_pass_as_one(self):
        checks = [Check(path="/items", op="equals", value=[{"quantity": 1}])]
        self.assertTrue(check_observation({"items": [{"quantity": True}]}, checks))

    def test_exit_codes_keep_missing_and_failures_visible(self):
        self.assertEqual(1, exit_status([{"status": "FAILED"}, {"status": "LIVE_PARTIAL_PASS"}]))
        self.assertEqual(1, exit_status([{"status": "DEPENDENCY_ERROR"}]))
        self.assertEqual(2, exit_status([{"status": "NOT_RUN"}]))
        self.assertEqual(2, exit_status([]))
        self.assertEqual(0, exit_status([{"status": "OFFLINE_PARTIAL_PASS"}]))

    def test_live_records_first_failed_turn_without_using_scripts(self):
        case = self.cases["SC-005-edit-items"]
        with patch("tests.conversation_regression.evaluate.plan_v2_turn",
                   side_effect=lambda req: response(req, "{}")) as planner:
            result = live_case(case, self.profile, FakeLiveModel())
        self.assertEqual("FAILED", result["status"])
        self.assertEqual(case.steps[0].id, result["first_failure"])
        self.assertEqual(1, planner.call_count)

    def test_dependency_failure_is_not_classified_as_product_failure(self):
        with patch("tests.conversation_regression.evaluate.plan_v2_turn",
                   side_effect=AgentDependencyError("Synthetic")):
            result = live_case(self.cases["SC-005-edit-items"], self.profile, FakeLiveModel())
        self.assertEqual("DEPENDENCY_ERROR", result["status"])

    def test_fault_injection_is_not_silently_replaced_by_live_model(self):
        with patch("tests.conversation_regression.evaluate.plan_v2_turn") as planner:
            result = live_case(self.cases["SC-012-failed-edit"], self.profile, FakeLiveModel())
        planner.assert_not_called()
        self.assertEqual("NOT_RUN", result["status"])

    def test_live_reports_environment_adapter_as_missing(self):
        with patch("tests.conversation_regression.evaluate.plan_v2_turn") as planner:
            result = live_case(self.cases["SC-013-route-renewal"], self.profile, FakeLiveModel())
        planner.assert_not_called()
        self.assertEqual("NOT_RUN", result["status"])

    def test_budget_is_checked_before_sending_request(self):
        model = LiveModel("synthetic-not-a-secret", "gpt-4.1-mini", max_calls=1)
        try:
            configuration = model.configuration()
            self.assertEqual("gpt-4.1-mini", configuration["requested_model"])
            self.assertEqual(4096, configuration["generation"]["max_output_tokens"])
            self.assertNotIn("synthetic-not-a-secret", json.dumps(configuration))
            model.calls.append({"purpose": "previous"})
            with patch.object(model, "original") as send:
                with self.assertRaises(EvaluationLimit):
                    model("prompt", purpose="NEW")
                send.assert_not_called()
        finally:
            model.close()

    def test_invalid_live_limits_and_missing_credentials_rejected(self):
        for key, calls in [("", 1), ("synthetic", 0)]:
            with self.assertRaises(ValueError):
                LiveModel(key, "gpt-4.1-mini", max_calls=calls)

    def test_validator_rejects_six_concrete_plans_for_the_right_reason(self):
        for case in self.cases.values():
            if case.scenario_id == "SC-016":
                with self.subTest(case=case.id):
                    result = validation_case(case, self.profile)
                    self.assertEqual("OFFLINE_PARTIAL_PASS", result["status"], result)
                    self.assertTrue(result["turns"][0]["checks_pending"])

    def test_rejected_plan_oracle_detects_valid_plan_mutation(self):
        case = self.cases["SC-016-forbidden-tool"].model_copy(deep=True)
        case.steps[0].event.data["allowed_tools"] = ["START_SERVICE"]
        self.assertEqual("FAILED", validation_case(case, self.profile)["status"])

    def test_integration_checks_read_persisted_state_not_ideal_response(self):
        case = self.cases["SC-005-edit-items"]
        bridge = Bridge([case])
        bridge.handle("/step", {"case_id": case.id, "step_id": case.steps[0].id})
        bridge.trace = [{"response": {"toolCalls": [], "messages": [],
                                      "updatedConversationSummary": '{"quantity":2}'}}]
        result = bridge.handle("/check", {"step_id": case.steps[0].id, "persisted_summary": "{}",
                                          "guest": {"preferredLanguage": "es-MX"}})
        self.assertTrue(result["errors"])
        self.assertEqual({}, result["observed"]["state"])

    def test_unknown_bridge_steps_and_non_synthetic_requests_rejected(self):
        case = self.cases["SC-005-edit-items"]
        bridge = Bridge([case])
        with self.assertRaises(ValueError):
            bridge.handle("/step", {"case_id": case.id, "step_id": "invented"})
        payload = deepcopy(self.profile)
        payload["hotel"]["hotelCode"] = "REAL"
        with self.assertRaises(ValueError):
            bridge.handle("/internal/v2/turns", payload)


if __name__ == "__main__":
    unittest.main()
