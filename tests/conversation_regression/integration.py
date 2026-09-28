"""Loopback bridge to the real planner, driven by Spring/H2 tests."""
from contextlib import contextmanager
import json
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Thread

from app.agents.v2_turn_planner import plan_v2_turn
from app.schemas.v2_turns import AgentTurnRequest
from tests.conversation_regression.library import ROOT, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.session import check_observation, structured_summary
from tests.conversation_regression.library import ModelReply


class Bridge:
    def __init__(self, cases):
        self.cases = {c.id: c for c in cases}
        self.case, self.step = None, None
        self.trace, self.results = [], []
        self.aliases = {}

    def handle(self, path, data):
        if path == "/step":
            self.case = self.cases[data["case_id"]]
            self.aliases = data.get('aliases', {})
            self.step = next((s for s in self.case.steps if s.id == data["step_id"]), None)
            if self.step is None and data["step_id"] != "__complete":
                raise ValueError("Unknown fixture step")
            if self.step is None or self.step.event.kind == "guest" or self.step.id == self.case.steps[0].id:
                self.trace = []
            return {}
        if path == "/internal/v2/turns":
            request = AgentTurnRequest.model_validate(data)
            if request.hotel.hotelCode != "SYNTHETIC_REGRESSION":
                raise ValueError("Only synthetic hotel requests accepted")
            scripted = self.step.model_replies if self.step and request.trigger.type.value == "INBOUND_MESSAGE" else []
            if self.step and self.step.event.kind == 'proposed_plan':
                proposal = self.step.event.data
                encoded = json.dumps(proposal['tool_call']).replace(proposal['request_message']['messageId'], str(request.trigger.messageId))
                for source, target in self.aliases.items():
                    encoded = encoded.replace(source, target)
                response = {'schemaVersion': '2.0', 'agentTurnId': str(request.agentTurnId),
                            'disposition': 'TOOL_CALLS_REQUIRED', 'messages': [], 'toolCalls': [json.loads(encoded)],
                            'updatedConversationSummary': request.conversation.summary, 'warnings': [],
                            'usage': {'model': 'synthetic-invalid-plan', 'inputTokens': 0, 'outputTokens': 0,
                                      'cachedInputTokens': 0, 'reasoningTokens': 0, 'totalTokens': 0}}
                if request.previousToolResults:
                    response.update(disposition='NO_ACTION', toolCalls=[])
                self.trace.append({'request': data, 'response': response, 'boundary': 'invalid plan injected after Python validation, before Java validation'})
                return response
            if request.trigger.type.value != 'INBOUND_MESSAGE' and request.previousToolResults:
                last_tool = request.previousToolResults[-1].toolName
                if last_tool == 'SEARCH_KNOWLEDGE':
                    source_step = next((s for s in self.case.steps if s.event.action == 'provide_approved_faq'), None)
                    if source_step:
                        scripted = source_step.model_replies
            scripted = [ModelReply.model_validate(json.loads(self.replace_ids(json.dumps(r.model_dump(mode='json')), request)))
                        for r in (scripted or [])]
            trace = {"request": request.model_dump(mode="json")}
            self.trace.append(trace)
            try:
                with scripted_runtime(scripted):
                    response = plan_v2_turn(request)
                trace["response"] = response.model_dump(mode="json")
                return trace["response"]
            except Exception as error:
                trace["error_type"] = type(error).__name__
                trace["reason"] = str(error)
                raise
        if path == "/check":
            # State and language come from freshly read Spring persistence, not Python expectations.
            index = 0
            if self.step and self.step.event.kind == "tool_result":
                index = len(self.trace) - 1 if self.step.event.action in {'complete_faq_handoff', 'complete_service_start'} else 1
            first = self.trace[index].get("response") if len(self.trace) > index else None
            if self.step and self.step.event.kind == "backend":
                first = next((t.get("response") for t in reversed(self.trace) if t.get("response")), None)
            if first is None and self.step and self.step.event.kind not in {'backend', 'proposed_plan'}:
                return {"id": data["step_id"], "errors": [{"reason": "No agent response"}], "trace": self.trace}
            observation = {
                "state": structured_summary(data["persisted_summary"]),
                "guest": data["guest"],
                "response": first, "tools": first['toolCalls'] if first else [],
                "tool_names": [c['toolName'] for c in first['toolCalls']] if first else [],
                "text": '\n'.join(m['text'] for m in first['messages']) if first else '',
                "message_purposes": [m['purpose'] for m in first['messages']] if first else [],
                "environment": {"source": "spring-h2", "createdOperations": data.get("created_operations", {}),
                                "operationInputs": data.get("operation_inputs", {})},
                "backend": data.get("backend", {}),
            }
            return {"id": data["step_id"], "observed": observation,
                    "errors": check_observation(observation, [c.model_copy(update={'value': self.mapped_value(c.value)}) for c in self.step.checks]) if self.step else [],
                    "trace": self.trace, "reviews_pending": self.step.review if self.step else []}
        if path == "/result":
            self.results.append(data)
            return {}
        raise ValueError("Unknown bridge route")

    def mapped_value(self, value):
        encoded = json.dumps(value)
        for source, target in self.aliases.items():
            encoded = encoded.replace(source, target)
        return json.loads(encoded)

    def replace_ids(self, encoded, request):
        for source, target in self.aliases.items():
            encoded = encoded.replace(source, target)
        return encoded.replace('$TRIGGER_MESSAGE_ID', str(request.trigger.messageId)).replace('$AGENT_TURN_ID', str(request.agentTurnId))


@contextmanager
def bridge_server(bridge):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    body = bytearray()
                    while True:
                        size = int(self.rfile.readline(128).split(b";")[0], 16)
                        if size == 0:
                            self.rfile.readline(128)
                            break
                        if size < 0 or len(body) + size > 2_000_000:
                            raise ValueError("Invalid chunk size")
                        body.extend(self.rfile.read(size))
                        if self.rfile.read(2) != b"\r\n":
                            raise ValueError("Invalid chunk terminator")
                else:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 1 or size > 2_000_000:
                        raise ValueError("Invalid request size")
                    body = self.rfile.read(size)
                if not body:
                    raise ValueError("Invalid request size")
                result = bridge.handle(self.path, json.loads(body))
                status = 200
            except Exception as error:
                result = {"error_type": type(error).__name__, "reason": str(error)}
                status = 422
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def run_integration(backend, cases, repetitions):
    backend = backend.resolve()
    if not (backend / "pom.xml").is_file():
        raise ValueError("Backend checkout not found")
    supported = [c for c in cases if c.integration_ready and all(s.model_replies is not None for s in c.steps)]
    results = [{"id": c.id, "scenario_id": c.scenario_id, "level": "integration",
                "status": "NOT_RUN", "first_failure": None, "turns": [],
                "reason": "No integration event adapter for this case"} for c in cases if c not in supported]
    metadata = {"integration_limits": [
        "Spring context, conversation persistence, turns, tool registry and service operations use real code and H2.",
        "Offering schemas/catalog snapshots and guest/stay lookup are fixture providers.",
        "FAQ search and approved-answer validation are real services over synthetic approved or empty catalog snapshots.",
        "Tool-result fixture steps inspect recorded synchronous agent turns and the final persisted state of that guest turn, not intermediate database commits.",
        "BPM process port and outbound delivery are simulated; no router or WhatsApp connection.",
        "Python planner is real over loopback HTTP; model replies scripted and response localization bypassed.",
        "Outbox retry and channel limits use the real worker/localizer with a scripted translation client and delivery gateway.",
        "Queued inbounds are persisted together then processed sequentially; this does not establish concurrent worker safety.",
        "Does not test PostgreSQL migrations, production HTTP authentication or every environment event.",
    ]}
    fingerprint = hashlib.sha256()
    for source in sorted((backend / "src").rglob("*.java")):
        fingerprint.update(source.relative_to(backend).as_posix().encode())
        fingerprint.update(b"\0")
        fingerprint.update(source.read_bytes())
    metadata["backend_java_sha256"] = fingerprint.hexdigest()
    metadata["backend_git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=backend, text=True).strip()
    if not supported:
        return {**metadata, "results": results}
    maven = shutil.which("mvn")
    if not maven:
        return {**metadata, "results": results + [
            {"id": c.id, "scenario_id": c.scenario_id, "status": "NOT_RUN", "turns": [],
             "reason": "Maven is unavailable"} for c in supported]}
    bridge = Bridge(supported)
    output = backend / "target/conversation-regression"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=output) as directory, bridge_server(bridge) as url:
        input_path = Path(directory) / "input.json"
        input_path.write_text(json.dumps({"cases": [c.model_dump(mode="json") for c in supported],
                                         "profiles": profiles(), "repeat": repetitions}), encoding="utf-8")
        # Inherit runtime paths only; never forward database, channel or hotel credentials.
        env = {k: v for k, v in os.environ.items()
               if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "JAVA_HOME",
                                "TEMP", "TMP", "USERPROFILE", "HOME", "LOCALAPPDATA", "APPDATA"}}
        command = [maven, "-o", "-q", "-Dtest=ConversationRegressionIntegrationTest",
                   "-Dspring.profiles.active=local-h2", "-Dregression.enabled=true",
                   f"-Dregression.bridge={url}", f"-Dregression.input={input_path}",
                   "-DredirectTestOutputToFile=true", "test"]
        repo = os.getenv("REGRESSION_MAVEN_REPOSITORY")
        if repo:
            command.insert(1, f"-Dmaven.repo.local={repo}")
        log_path = output / (Path(directory).name + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, cwd=backend, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=300, check=False)
                metadata["maven_exit_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                metadata["maven_exit_code"] = "TIMEOUT"
        metadata["integration_log"] = str(log_path)
    for case in supported:
        for repetition in range(1, repetitions + 1):
            found = next((r for r in bridge.results if r["id"] == case.id and r["repetition"] == repetition), None)
            results.append(found or {"id": case.id, "scenario_id": case.scenario_id, "level": "integration",
                                     "repetition": repetition, "status": "HARNESS_ERROR", "turns": [],
                                     "reason": "Spring did not return a result; inspect integration log"})
    if metadata["maven_exit_code"] != 0 and not any(r["status"] in {"FAILED", "HARNESS_ERROR"} for r in results):
        results.append({"id": "integration-process", "status": "HARNESS_ERROR", "turns": [],
                        "reason": "Maven failed after returning case results"})
    return {**metadata, "results": results}
