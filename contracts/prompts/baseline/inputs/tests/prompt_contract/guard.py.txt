"""Fail on unreviewed prompt changes; propose a diff without accepting it.

Usage: python -m tests.prompt_contract.guard check
       python -m tests.prompt_contract.guard propose --out NEW_DIRECTORY --reason TEXT
"""
import argparse
import ast
import difflib
import fnmatch
import hashlib
import json
from pathlib import Path

from tests.conversation_regression.library import ROOT, load_cases

CONTRACT = ROOT / "contracts/prompts"
MANIFEST = CONTRACT / "manifest.v1.json"
BASELINE = CONTRACT / "baseline"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path outside root: {relative}")
    return path


def model_sources(root=ROOT):
    """Discover clients/imports, including aliases. Newly introduced callers must register.

    This is a static inventory, not a security boundary against arbitrary dynamic Python.
    """
    result = set()
    for path in (root / "app").rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative == "app/services/openai_client.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if path.parent == root / "app/prompts" and path.name != "__init__.py":
            result.add(relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"app.services.openai_client", "openai"}:
                result.add(relative)
            elif isinstance(node, ast.Import) and any(a.name in {"openai", "app.services.openai_client"} for a in node.names):
                result.add(relative)
    return result


def dependencies(paths, root=ROOT):
    """Conservatively snapshot whole local dependency files, not just prompt literals."""
    pending, found = list(paths), set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = safe_path(root, relative)
        if not path.is_file():
            raise ValueError(f"Missing prompt dependency: {relative}")
        found.add(relative)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
                modules.append(node.module)
                modules.extend(node.module + "." + alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names if alias.name.startswith("app."))
        for module in modules:
            candidate = module.replace(".", "/") + ".py"
            if (root / candidate).is_file():
                pending.append(candidate)
            # Package initializers can change the prompt-building dependencies too.
            parts = module.split(".")
            for count in range(1, len(parts)):
                candidate = "/".join(parts[:count]) + "/__init__.py"
                if (root / candidate).is_file():
                    pending.append(candidate)
    return sorted(found)


def validate_manifest(manifest, root=ROOT, cases=None):
    errors = []
    if manifest.get("version") != 1 or not manifest.get("prompts"):
        return ["Invalid or empty prompt manifest"]
    contract = read_json(root / "contracts/behavior/conversation-behavior.v1.json")
    rules = {r["id"]: r for r in contract["rules"]}
    cases = load_cases() if cases is None else cases
    cases_by_id = {c.id: c for c in cases}
    ids, registered = set(), set()
    for item in manifest["prompts"]:
        identity = item["id"]
        if identity in ids:
            errors.append(f"Duplicate prompt ID: {identity}")
        ids.add(identity)
        registered.update(item["sources"])
        if not item["rule_ids"] or not item["unit_files"] or not item["case_ids"]:
            errors.append(f"Missing rule/test links: {identity}")
        for rule in item["rule_ids"]:
            if rule not in rules:
                errors.append(f"Unknown rule {rule}: {identity}")
        scenarios = {scenario for rule in item["rule_ids"] if rule in rules for scenario in rules[rule]["scenarioIds"]}
        for case_id in item["case_ids"]:
            if case_id not in cases_by_id or cases_by_id[case_id].scenario_id not in scenarios:
                errors.append(f"Unknown or unrelated conversation case {case_id}: {identity}")
        for path in item["sources"] + item["unit_files"]:
            if not safe_path(root, path).is_file():
                errors.append(f"Missing mapped file {path}: {identity}")
        for path in item["unit_files"]:
            if (root / path).is_file():
                tree = ast.parse((root / path).read_text(encoding="utf-8"))
                if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in ast.walk(tree)):
                    errors.append(f"Mapped file contains no tests: {path}")
    for path in sorted(model_sources(root) - registered):
        errors.append(f"Unregistered prompt/model source: {path}")
    anchor_ids = set()
    for anchor in manifest.get("anchors", []):
        if anchor["id"] in anchor_ids or anchor["prompt_id"] not in ids or anchor["rule_id"] not in rules or not anchor["text"]:
            errors.append(f"Invalid critical anchor: {anchor['id']}")
        owner = next((p for p in manifest["prompts"] if p["id"] == anchor["prompt_id"]), None)
        if owner and anchor["rule_id"] not in owner["rule_ids"]:
            errors.append(f"Anchor rule is not mapped to its prompt: {anchor['id']}")
        anchor_ids.add(anchor["id"])
    if not anchor_ids:
        errors.append("No critical prompt anchors")
    return errors


def validate_anchors(manifest, documents):
    errors = []
    for anchor in manifest["anchors"]:
        matches = [text for key, text in documents.items() if fnmatch.fnmatchcase(key, anchor["document_glob"])]
        if not matches or any(anchor["text"] not in text for text in matches):
            errors.append(f"Critical instruction missing from rendered prompt: {anchor['id']} ({anchor['rule_id']})")
    return errors


def collect(manifest, root=ROOT):
    from tests.prompt_contract.render import effective_prompts
    docs = {"manifest.json": canonical(manifest)}
    for path in dependencies([s for p in manifest["prompts"] for s in p["sources"]], root):
        docs["sources/" + path + ".txt"] = (root / path).read_text(encoding="utf-8")
    # Pin the synthetic inputs, test links and capture code as reviewable dependencies.
    paths = ["contracts/behavior/conversation-behavior.v1.json", "tests/test_prompt_contract.py", "tests/run_offline.py"]
    paths += ["contracts/ci/gate.v1.json", "tests/test_ci_gate.py", "tests/test_conversation_coverage.py"]
    paths += [p.relative_to(root).as_posix() for p in (root / "tests/ci_gate").glob("*.py")]
    paths += [path for item in manifest["prompts"] for path in item["unit_files"]]
    paths += [p.relative_to(root).as_posix() for p in (root / "tests/prompt_contract").glob("*.py")]
    paths += [p.relative_to(root).as_posix() for p in (root / "tests/conversation_regression/fixtures").rglob("*.json")]
    paths += [p.relative_to(root).as_posix() for p in (root / "tests/conversation_regression").glob("*.py")]
    for path in sorted(set(paths)):
        docs["inputs/" + path + ".txt"] = (root / path).read_text(encoding="utf-8")
    docs.update({"effective/" + key: value for key, value in effective_prompts().items()})
    return dict(sorted(docs.items()))


def read_snapshot(directory):
    index = read_json(directory / "index.json")
    documents = {}
    for name, digest in index["documents"].items():
        text = safe_path(directory, name).read_text(encoding="utf-8")
        if sha(text) != digest:
            raise ValueError(f"Snapshot digest mismatch: {name}")
        documents[name] = text
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} - {"index.json"}
    if actual != set(documents):
        raise ValueError("Snapshot has unindexed or missing files")
    return documents


def write_snapshot(directory, documents):
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in documents.items():
        path = safe_path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", newline="\n")
    (directory / "index.json").write_text(canonical({"version": 1, "documents": {name: sha(value) for name, value in documents.items()}}), encoding="utf-8")


def compare(before, after):
    changes = []
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        diff = list(difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(), fromfile="baseline/" + key, tofile="candidate/" + key, lineterm=""))
        changes.append({"document": key, "before_sha256": sha(old) if old is not None else None,
                        "after_sha256": sha(new) if new is not None else None,
                        "removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in diff),
                        "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in diff), "diff": "\n".join(diff)})
    return changes


def assess(manifest, before, after, errors=(), cases=None):
    changes = compare(before, after)
    impacted = []
    changed = {c["document"] for c in changes}
    cases = load_cases() if cases is None else cases
    case_map = {c.id: c for c in cases}
    for item in manifest["prompts"]:
        source_docs = {"sources/" + path + ".txt" for path in dependencies(item["sources"])}
        matched = any(fnmatch.fnmatchcase(name, pattern) for name in changed for pattern in item["effective_globs"])
        if changed & source_docs or matched or any(name == "manifest.json" or name.startswith("inputs/") for name in changed):
            impacted.append({"id": item["id"], "name": item["name"], "rule_ids": item["rule_ids"],
                "unit_files": item["unit_files"], "cases": [{"id": cid, "scripted_replay_ready": case_map[cid].offline_ready,
                    "required_levels": case_map[cid].required_levels, "result_for_this_change": "NOT_RUN"} for cid in item["case_ids"] if cid in case_map]})
    failures = list(errors) + validate_anchors(manifest, after)
    return {"status": "BLOCKED" if failures or changes else "PASS", "errors": failures, "changes": changes,
            "affected_prompts": impacted, "network": "blocked", "model_calls": 0,
            "meaning": "PASS means unchanged registered prompts/dependencies and surviving textual anchors; it does not certify model behavior or authorize deployment."}


def check():
    manifest = read_json(MANIFEST)
    errors = validate_manifest(manifest)
    if errors:
        return {"status": "BLOCKED", "errors": errors, "changes": []}
    return assess(manifest, read_snapshot(BASELINE), collect(manifest))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "propose"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.command == "propose" and (not args.out or not args.reason or not args.reason.strip()):
        parser.error("propose requires a new --out directory and a nonblank --reason")
    try:
        manifest = read_json(MANIFEST)
        errors = validate_manifest(manifest)
        if errors:
            print(canonical({"status": "BLOCKED", "errors": errors}))
            return 1
        before, after = read_snapshot(BASELINE), collect(manifest)
        report = assess(manifest, before, after)
        if args.command == "propose":
            args.out.mkdir(parents=True, exist_ok=False)
            report.update(reason=args.reason, review_status="PENDING", deployment_authorized=False)
            write_snapshot(args.out / "candidate", after)
            (args.out / "report.json").write_text(canonical(report), encoding="utf-8")
            lines = ["# Prompt change review", "", args.reason, "", "Status: " + report["status"],
                "Review pending. A generated candidate never replaces the baseline.",
                "Linked behavioral tests are NOT_RUN for this change until independently executed.", ""]
            lines += ["- " + error for error in report["errors"]]
            for item in report["affected_prompts"]:
                lines.extend(["", "## " + item["id"] + " — " + item["name"], ", ".join(item["rule_ids"]),
                              "Unit files: " + ", ".join(item["unit_files"]),
                              "Conversation cases: " + ", ".join(c["id"] for c in item["cases"])])
            for change in report["changes"]:
                lines.extend(["", "## " + change["document"], "```diff", change["diff"], "```"])
            (args.out / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(canonical({"status": report["status"], "changed_documents": len(report["changes"]), "errors": report["errors"],
                         "affected_prompts": [p["id"] for p in report["affected_prompts"]]}))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, ValueError, AssertionError) as error:
        print(canonical({"status": "BLOCKED", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
