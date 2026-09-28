"""The gate must reject missing, skipped, stale and deceptively green evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests.ci_gate.run import read_backend_reports, validate_regression, validate_review, validate_tests


class ConversationGateTest(unittest.TestCase):
    def setUp(self):
        self.test_policy = {'required_ids': ['critical'], 'allowed_skips': ['live']}
        self.tests = {'status': 'PASS', 'tests': 2, 'executed_ids': ['critical', 'live'],
                      'skipped_ids': ['live'], 'errors': [], 'failures': []}
        self.policy = {'C1': {'status': 'OFFLINE_PARTIAL_PASS', 'turn_ids': ['first'],
                              'not_run_steps': [], 'checks_pending': {}},
                       'C2': {'status': 'NOT_RUN', 'turn_ids': [], 'not_run_steps': [], 'checks_pending': {}}}
        self.report = {'level': 'offline', 'repetitions': 1, 'source_fingerprint': {'source': 'current'},
                       'selected_cases': ['C1', 'C2'], 'counts': {'OFFLINE_PARTIAL_PASS': 1, 'NOT_RUN': 1},
                       'results': [{'id': 'C1', 'status': 'OFFLINE_PARTIAL_PASS',
                                    'turns': [{'id': 'first', 'errors': []}]},
                                   {'id': 'C2', 'status': 'NOT_RUN', 'turns': []}]}

    def validate(self):
        return validate_regression(self.report, self.policy, 'offline', {'source': 'current'})

    def test_only_explicit_pending_coverage_is_accepted_and_reported(self):
        result = self.validate()
        self.assertEqual([r['id'] for r in result['pending']], ['C2'])
        self.assertEqual(validate_tests(self.tests, self.test_policy)['passed'], 1)

    def test_missing_mandatory_unit_blocks(self):
        self.tests.update(tests=1, executed_ids=['live'])
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            validate_tests(self.tests, self.test_policy)

    def test_unexpected_skip_blocks(self):
        self.tests['skipped_ids'].append('critical')
        with self.assertRaisesRegex(ValueError, 'Unexpected skipped'):
            validate_tests(self.tests, self.test_policy)

    def test_failure_blocks_even_with_pass_status(self):
        self.tests['errors'] = ['critical']
        with self.assertRaises(ValueError):
            validate_tests(self.tests, self.test_policy)

    def test_expected_failure_cannot_replace_passing_test(self):
        self.tests['expected_failures'] = ['critical']
        with self.assertRaises(ValueError):
            validate_tests(self.tests, self.test_policy)

    def test_duplicate_test_ids_cannot_inflate_coverage(self):
        self.tests['executed_ids'] = ['critical', 'critical']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_tests(self.tests, self.test_policy)

    def test_required_conversation_not_run_blocks(self):
        self.report['results'][0]['status'] = 'NOT_RUN'
        self.report['counts'] = {'NOT_RUN': 2}
        with self.assertRaisesRegex(ValueError, 'required status'):
            self.validate()

    def test_removed_conversation_blocks(self):
        self.report['results'].pop(0)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.validate()

    def test_duplicate_conversation_blocks(self):
        self.report['results'].append(deepcopy(self.report['results'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_skipped_step_blocks_even_with_pass_status(self):
        self.report['results'][0]['turns'] = []
        with self.assertRaisesRegex(ValueError, 'turns'):
            self.validate()

    def test_failed_assertion_blocks_even_with_pass_status(self):
        self.report['results'][0]['turns'][0]['errors'] = ['lost quantity']
        with self.assertRaisesRegex(ValueError, 'checks'):
            self.validate()

    def test_new_pending_assertion_blocks(self):
        self.report['results'][0]['turns'][0]['checks_pending'] = [{'path': '/backend/startCount'}]
        with self.assertRaisesRegex(ValueError, 'unevaluated'):
            self.validate()

    def test_stale_source_blocks(self):
        self.report['source_fingerprint'] = {'source': 'yesterday'}
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.validate()

    def test_green_counts_cannot_hide_failure(self):
        self.report['counts'] = {'OFFLINE_PARTIAL_PASS': 2}
        with self.assertRaisesRegex(ValueError, 'counts'):
            self.validate()

    def test_empty_backend_report_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'No fresh'):
                read_backend_reports(folder)

    def test_backend_xml_errors_are_not_green(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'TEST-one.xml').write_text('<testsuite tests="1"><testcase classname="suite" name="critical"><error/></testcase></testsuite>')
            with self.assertRaises(ValueError):
                validate_tests(read_backend_reports(folder), {'required_ids': ['suite.critical'], 'allowed_skips': []})

    def test_review_is_bound_to_exact_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts = root / 'contracts/prompts'
            (prompts / 'baseline').mkdir(parents=True)
            index = {'documents': {'a': 'digest'}}
            (prompts / 'baseline/index.json').write_text(json.dumps(index))
            (root / 'review.md').write_text('Inspected changes and linked behavioral evidence.')
            review = {'baseline_index_sha256': hashlib.sha256(json.dumps(index, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      'reason': 'test', 'reviewer': 'local', 'evidence': 'review.md', 'human_approval': False}
            (prompts / 'review.v1.json').write_text(json.dumps(review))
            validate_review(root)
            (prompts / 'baseline/index.json').write_text(json.dumps({'documents': {}}))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                validate_review(root)


if __name__ == '__main__':
    unittest.main()
