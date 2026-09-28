"""Regression checks for accidental prompt loss and review bypasses, entirely offline."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from tests.prompt_contract import guard
from tests.prompt_contract.render import effective_prompts


class PromptContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = guard.read_json(guard.MANIFEST)
        cls.current = guard.collect(cls.manifest)
        cls.baseline = guard.read_snapshot(guard.BASELINE)

    def test_current_prompts_match_reviewed_baseline(self):
        errors = guard.validate_manifest(self.manifest)
        self.assertEqual([], errors)
        report = guard.assess(self.manifest, self.baseline, self.current)
        self.assertEqual("PASS", report["status"], json.dumps({"errors": report["errors"],
            "changed": [c["document"] for c in report["changes"]]}, ensure_ascii=False))

    def test_effective_prompts_are_repeatable(self):
        again = {"effective/" + key: value for key, value in effective_prompts().items()}
        self.assertEqual({k: v for k, v in self.current.items() if k.startswith("effective/")}, again)

    def test_rule_removal_is_caught_even_if_source_keeps_it_as_a_comment(self):
        changed = dict(self.current)
        anchor = self.manifest["anchors"][0]
        for name in list(changed):
            if guard.fnmatch.fnmatchcase(name, anchor["document_glob"]):
                changed[name] = changed[name].replace(anchor["text"], "")
        self.assertEqual(self.current["sources/app/prompts/v2_turn.py.txt"], changed["sources/app/prompts/v2_turn.py.txt"])
        errors = guard.validate_anchors(self.manifest, changed)
        self.assertTrue(any(anchor["id"] in e for e in errors))

    def test_blind_baseline_refresh_does_not_silence_missing_critical_rule(self):
        changed = dict(self.current)
        anchor = next(a for a in self.manifest["anchors"] if a["prompt_id"] == "PR-002")
        for key in changed:
            if guard.fnmatch.fnmatchcase(key, anchor["document_glob"]):
                changed[key] = changed[key].replace(anchor["text"], "")
        report = guard.assess(self.manifest, changed, changed)
        self.assertEqual([], report["changes"])
        self.assertEqual("BLOCKED", report["status"])

    def test_context_loss_is_detected_without_relying_on_anchor_phrases(self):
        changed = dict(self.current)
        key = "effective/builders/V2_AGENT_TURN/en-US.prompt.txt"
        changed[key] = changed[key].split("Runtime context:")[0]
        self.assertEqual([], guard.validate_anchors(self.manifest, changed))
        report = guard.assess(self.manifest, self.current, changed)
        self.assertEqual("BLOCKED", report["status"])
        self.assertIn("PR-001", [p["id"] for p in report["affected_prompts"]])

    def test_schema_changes_are_reviewed_even_with_identical_prompt_text(self):
        changed = dict(self.current)
        key = next(k for k in changed if k.endswith(".options.json") and "V2_HOTEL_SCOPE" in k)
        options = json.loads(changed[key])
        options["response_schema"]["required"] = []
        changed[key] = guard.canonical(options)
        report = guard.assess(self.manifest, self.current, changed)
        self.assertEqual("BLOCKED", report["status"])
        self.assertIn("PR-002", [p["id"] for p in report["affected_prompts"]])

    def test_template_change_affects_localization_and_message_rules(self):
        changed = dict(self.current)
        key = "sources/app/services/message_templates.json.txt"
        registry = json.loads(changed[key])
        registry["templates"]["order.change"]["en"] = "Repeat the whole order."
        changed[key] = guard.canonical(registry)
        self.assertNotEqual(self.current[key], changed[key])
        report = guard.assess(self.manifest, self.current, changed)
        self.assertEqual("BLOCKED", report["status"])
        self.assertIn("PR-011", [p["id"] for p in report["affected_prompts"]])

    def test_shared_quantity_policy_affects_both_planner_and_extractor(self):
        changed = dict(self.current)
        changed["sources/app/prompts/input_policy.py.txt"] = "ORDER_QUANTITY_POLICY = ''\n"
        report = guard.assess(self.manifest, self.current, changed)
        self.assertTrue({"PR-001", "PR-003", "PR-012"}.issubset({p["id"] for p in report["affected_prompts"]}))

    def test_unknown_rule_and_case_links_fail_closed(self):
        manifest = deepcopy(self.manifest)
        manifest["prompts"][0]["rule_ids"].append("BC-999")
        manifest["prompts"][0]["case_ids"].append("SC-999-missing")
        errors = guard.validate_manifest(manifest)
        self.assertTrue(any("Unknown rule BC-999" in e for e in errors))
        self.assertTrue(any("SC-999-missing" in e for e in errors))

    def test_new_alias_model_client_and_new_builder_are_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "app/prompts").mkdir(parents=True)
            (root / "app/new_agent.py").write_text("from app.services.openai_client import call_openai_json_result as model\n", encoding="utf-8")
            (root / "app/prompts/new.py").write_text("def prompt(): return 'new'\n", encoding="utf-8")
            self.assertEqual({"app/new_agent.py", "app/prompts/new.py"}, guard.model_sources(root))
        with patch.object(guard, "model_sources", return_value={"app/new_agent.py"}):
            self.assertIn("Unregistered prompt/model source: app/new_agent.py", guard.validate_manifest(self.manifest))

    def test_missing_capture_cannot_satisfy_anchor(self):
        docs = {k: v for k, v in self.current.items() if "V2_SPA_EXTRACTION" not in k}
        self.assertTrue(any("BC-003" in error for error in guard.validate_anchors(self.manifest, docs)))

    def test_snapshot_tampering_and_unindexed_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            guard.write_snapshot(directory, {"prompt.txt": "original"})
            (directory / "prompt.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                guard.read_snapshot(directory)
            (directory / "prompt.txt").write_text("original", encoding="utf-8")
            (directory / "unindexed.txt").write_text("unregistered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unindexed"):
                guard.read_snapshot(directory)

    def test_snapshot_paths_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "outside root"):
                guard.safe_path(Path(temp), "../outside.txt")

    def test_diff_records_additions_removals_and_hashes(self):
        changes = guard.compare({"prompt.txt": "keep\nrequired rule\n"}, {"prompt.txt": "keep\nnew wording\n"})
        self.assertEqual(1, changes[0]["removed_lines"])
        self.assertEqual(1, changes[0]["added_lines"])
        self.assertIn("-required rule", changes[0]["diff"])
        self.assertNotEqual(changes[0]["before_sha256"], changes[0]["after_sha256"])

    def test_proposal_never_accepts_baseline_or_claims_test_success(self):
        changed = dict(self.current)
        changed["sources/app/prompts/input_policy.py.txt"] += "\n# proposed edit\n"
        with tempfile.TemporaryDirectory() as temp, patch.object(guard, "collect", return_value=changed):
            out = Path(temp) / "review"
            self.assertEqual(1, guard.main(["propose", "--out", str(out), "--reason", "Synthetic review test"]))
            report = guard.read_json(out / "report.json")
            self.assertEqual("PENDING", report["review_status"])
            self.assertFalse(report["deployment_authorized"])
            self.assertTrue(all(c["result_for_this_change"] == "NOT_RUN" for p in report["affected_prompts"] for c in p["cases"]))
            self.assertEqual(self.baseline, guard.read_snapshot(guard.BASELINE))
            self.assertEqual(changed, guard.read_snapshot(out / "candidate"))
            self.assertEqual(1, guard.main(["propose", "--out", str(out), "--reason", "Must not overwrite"]))

    def test_prompt_capture_blocks_accidental_network_access(self):
        def accidental_network(text):
            with socket.socket() as connection:
                connection.connect(("127.0.0.1", 1))
        with patch("app.agents.social_opening.classify_social_opening", side_effect=accidental_network):
            with self.assertRaisesRegex(AssertionError, "External effects are prohibited"):
                effective_prompts()


if __name__ == "__main__":
    unittest.main()
