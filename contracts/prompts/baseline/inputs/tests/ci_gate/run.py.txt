"""Run fresh agent/backend checks, rejecting missing coverage and stale evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / 'contracts/ci/gate.v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(values, label):
    require(len(values) == len(set(values)), label + ': duplicate IDs')
    return set(values)


def validate_tests(report, policy):
    require(report['status'] == 'PASS', 'Test process did not pass')
    require(not any(report.get(k) for k in ('errors', 'failures', 'expected_failures', 'unexpected_successes')),
            'Failed, expected-failing or unexpectedly successful tests')
    executed = unique(report['executed_ids'], 'Executed tests')
    skipped = unique(report['skipped_ids'], 'Skipped tests')
    required = unique(policy['required_ids'], 'Required tests')
    allowed = unique(policy['allowed_skips'], 'Allowed skips')
    require(required and not (required & allowed), 'Empty or conflicting test policy')
    require(report['tests'] == len(executed), 'Test count does not match executed IDs')
    require(skipped <= executed and skipped <= allowed, 'Unexpected skipped tests')
    require(required <= executed - skipped, 'Missing required tests: ' + ', '.join(sorted(required - (executed - skipped))))
    return {'passed': len(executed - skipped), 'skipped': sorted(skipped)}


def read_backend_reports(directory):
    report = {'status': 'PASS', 'tests': 0, 'executed_ids': [], 'skipped_ids': [], 'failures': [], 'errors': []}
    files = list(Path(directory).glob('TEST-*.xml'))
    require(bool(files), 'No fresh backend test reports')
    for path in files:
        suite = ET.parse(path).getroot()
        cases = suite.findall('testcase')
        require(int(suite.attrib['tests']) == len(cases), 'Incomplete backend XML: ' + path.name)
        for case in cases:
            test_id = case.attrib['classname'] + '.' + case.attrib['name']
            report['executed_ids'].append(test_id)
            for tag, key in [('skipped', 'skipped_ids'), ('failure', 'failures'), ('error', 'errors')]:
                if case.find(tag) is not None:
                    report[key].append(test_id)
    report['tests'] = len(report['executed_ids'])
    return report


def validate_regression(report, policy, level, expected_fingerprint):
    success = 'OFFLINE_PARTIAL_PASS' if level == 'offline' else 'INTEGRATION_PARTIAL_PASS'
    require(any(c['status'] == success for c in policy.values()), 'No mandatory conversations')
    require(all(c['status'] in {success, 'NOT_RUN'} for c in policy.values()), 'Invalid coverage policy')
    require(report['level'] == level and report['repetitions'] == 1, 'Wrong regression level/repetitions')
    require(report['source_fingerprint'] == expected_fingerprint, 'Stale agent fingerprint')
    require(unique(report['selected_cases'], 'Selected cases') == set(policy), 'Changed selected coverage')
    rows = report['results']
    require(unique([r['id'] for r in rows], 'Result cases') == set(policy), 'Missing or unexpected result cases')
    require(report['counts'] == dict(Counter(r['status'] for r in rows)), 'Inconsistent result counts')
    pending = []
    for row in rows:
        case = policy[row['id']]
        require(row['status'] == case['status'], row['id'] + ': changed required status')
        require(not row.get('first_failure'), row['id'] + ': first failure present')
        require(row.get('not_run_steps', []) == case['not_run_steps'], row['id'] + ': steps omitted')
        require([t['id'] for t in row['turns']] == case['turn_ids'], row['id'] + ': missing/reordered turns')
        for turn in row['turns']:
            require('errors' in turn and not turn['errors'], row['id'] + ': failed/missing checks')
            require(turn.get('checks_pending', []) == case['checks_pending'].get(turn['id'], []),
                    row['id'] + ': additional checks unevaluated')
        if row['status'] == 'NOT_RUN' or case['not_run_steps'] or case['checks_pending']:
            pending.append({'id': row['id'], 'status': row['status'], 'not_run_steps': case['not_run_steps'],
                            'checks_pending': case['checks_pending']})
    return {'counts': report['counts'], 'pending': pending}


def fingerprint(root, agent=False):
    paths = set()
    directories = ('app', 'contracts', 'tests', '.github') if agent else ('src', '.github')
    for directory in directories:
        for path in (root / directory).rglob('*'):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and '__pycache__' not in path.parts and not relative.startswith('tests/conversation_regression/reports/'):
                paths.add(path)
    for name in ('pom.xml', 'requirements.txt', 'Dockerfile', 'render.sandbox.yaml', '.gitignore'):
        if (root / name).is_file():
            paths.add(root / name)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        # Normalize checkout line endings, but preserve all other content.
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {'git_head': head, 'inputs_sha256': digest.hexdigest()}


def validate_review(root):
    baseline = read(root / 'contracts/prompts/baseline/index.json')
    digest = hashlib.sha256(json.dumps(baseline, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    review = read(root / 'contracts/prompts/review.v1.json')
    require(review['baseline_index_sha256'] == digest, 'Prompt review does not match baseline')
    require(review['reason'].strip() and review['reviewer'].strip(), 'Missing recorded review')
    evidence = (root / review['evidence']).resolve()
    require(evidence.is_relative_to(root.resolve()) and evidence.is_file(), 'Missing review evidence')
    require(review.get('human_approval') is False, 'Local record must not impersonate PR approval')
    return {'baseline_index_sha256': digest, 'review': review,
            'limitation': 'Recorded implementation review; independent approval is enforced by repository rules.'}


def child_environment():
    # No inherited model, hotel or deployment credentials. Maven/Java need these local paths.
    names = ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'JAVA_HOME', 'TEMP', 'TMP',
             'USERPROFILE', 'HOME', 'LOCALAPPDATA', 'APPDATA', 'REGRESSION_MAVEN_REPOSITORY')
    env = {k: v for k, v in os.environ.items() if k.upper() in names}
    env.update(PYTHONIOENCODING='utf-8', RUN_LIVE_SCOPE_EVALS='0', RUN_LIVE_CATALOG_EVALS='0',
               RUN_LIVE_MULTILINGUAL_EVALS='0')
    return env


def command(argv, cwd, log, env, timeout=900, allowed=(0,)):
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode in allowed, f'Process exited {result.returncode}; see {log.name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--maven-offline', action='store_true')
    args = parser.parse_args(argv)
    backend, out = args.backend.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'BLOCKED', 'started_at': datetime.now(timezone.utc).isoformat(), 'stages': {},
               'live_model': 'NOT_RUN: explicit separate authorization required',
               'deployment_authorized': False}
    env = child_environment()
    try:
        require((backend / 'pom.xml').is_file(), 'Backend checkout missing')
        policy = read(POLICY)
        require(policy['version'] == 1, 'Unsupported gate policy')
        before = {'agent': fingerprint(ROOT, True), 'backend': fingerprint(backend)}
        summary['sources'] = before
        from tests.conversation_regression.replay import source_fingerprint
        expected = source_fingerprint()

        def stage(name, fn):
            print('Running ' + name, flush=True)
            try:
                summary['stages'][name] = {'status': 'PASS', 'result': fn()}
            except Exception as error:
                summary['stages'][name] = {'status': 'FAILED', 'error': str(error)}
            print(name + ': ' + summary['stages'][name]['status'], flush=True)

        def prompts():
            command([sys.executable, '-m', 'tests.prompt_contract.guard', 'check'], ROOT, out / 'prompts.log', env)
            return validate_review(ROOT)

        def unit():
            report = out / 'agent-unit.json'
            command([sys.executable, '-m', 'tests.run_offline', '--report', str(report)], ROOT, out / 'agent-unit.log', env)
            return validate_tests(read(report), policy['agent_unit'])

        def backend_tests():
            reports = out / 'backend-xml'
            reports.mkdir()
            mvn = shutil.which('mvn') or shutil.which('mvn.cmd')
            require(bool(mvn), 'Maven not found')
            cmd = [mvn, '-B', '-q', '-Dspring.profiles.active=local-h2', '-DredirectTestOutputToFile=true',
                   '-Dconversation.test.reports=' + str(reports), 'test']
            if args.maven_offline:
                cmd.insert(1, '-o')
            if env.get('REGRESSION_MAVEN_REPOSITORY'):
                cmd.insert(1, '-Dmaven.repo.local=' + env['REGRESSION_MAVEN_REPOSITORY'])
            command(cmd, backend, out / 'backend-unit.log', env)
            result = read_backend_reports(reports)
            write(out / 'backend-unit.json', result)
            return validate_tests(result, policy['backend_unit'])

        def regression(level):
            report = out / (level + '.json')
            command([sys.executable, '-m', 'tests.conversation_regression.evaluate', '--level', level,
                     '--backend', str(backend), '--report', str(report)], ROOT, out / (level + '.log'), env,
                    timeout=600, allowed=(0, 2))  # Exit 2 is acceptable only with exact reviewed pending coverage.
            return validate_regression(read(report), policy[level], level, expected)

        stage('prompt-contract', prompts)
        stage('agent-unit', unit)
        stage('offline-conversations', lambda: regression('offline'))
        stage('backend-unit', backend_tests)
        stage('spring-conversations', lambda: regression('integration'))
        require(before == {'agent': fingerprint(ROOT, True), 'backend': fingerprint(backend)}, 'Sources changed during gate')
        required = {'prompt-contract', 'agent-unit', 'offline-conversations', 'backend-unit', 'spring-conversations'}
        require(set(summary['stages']) == required and all(s['status'] == 'PASS' for s in summary['stages'].values()),
                'Mandatory checks failed; see stages and logs')
        summary['status'] = 'PASS'
    except Exception as error:
        summary['error'] = str(error)
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(out / 'summary.json', summary)
    lines = ['# Conversation gate: ' + summary['status'], '',
             '| Mandatory check | Result |', '| --- | --- |']
    lines += [f'| {k} | {v["status"]} |' for k, v in summary['stages'].items()]
    lines += ['', 'Live model: NOT_RUN. This gate does not certify model wording, real WhatsApp or remote BPM.',
              'See summary.json for exact pending cases/checks and paired source fingerprints.',
              'A PASS applies only to the configured deterministic coverage; it does not authorize deployment.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(summary['status'] + ': ' + str(out / 'summary.json'), flush=True)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
