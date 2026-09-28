"""Capture synthetic effective prompts without contacting a model or hotel service.

Conversation probes cover only scripted paths. Source snapshots cover other branches;
neither mechanism establishes real-model accuracy.
"""
from collections import OrderedDict
from contextlib import ExitStack
import importlib
import inspect
import json
import socket
import time
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from tests.conversation_regression import replay
from tests.conversation_regression.library import load_cases, profiles


class Captured(BaseException):
    """Stop at the model boundary, outside product fallback exception handlers."""


def effective_prompts():
    documents = {}
    counts = {}

    def record(label, prompt, kwargs):
        index = counts.get(label, 0) + 1
        counts[label] = index
        key = f"{label}/{index:03}"
        documents[key + ".prompt.txt"] = prompt
        # Timing changes on every run. Keep the model-facing schema and API options.
        options = {k: v for k, v in kwargs.items() if k != "timeout_seconds"}
        documents[key + ".options.json"] = json.dumps(options, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def stop(label):
        def capture(prompt, **kwargs):
            record(label, prompt, kwargs)
            raise Captured()
        return capture

    def probe(label, module, callback):
        with patch.object(module, "call_openai_json_result", side_effect=stop(label)):
            try:
                callback()
            except Captured:
                return
        raise AssertionError(f"Prompt probe did not reach the model boundary: {label}")

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket, "connect", replay.prohibit_external_effect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", replay.prohibit_external_effect))
        stack.enter_context(patch("app.services.openai_client.get_openai_client", side_effect=replay.prohibit_external_effect))
        # Stable generated IDs retain their relationships; input evidence IDs are not rewritten.
        serial = iter(range(100000))
        for name in replay.MODEL_MODULES:
            module = importlib.import_module(name)
            if hasattr(module, "uuid4"):
                stack.enter_context(patch.object(module, "uuid4", side_effect=lambda: uuid5(NAMESPACE_URL, f"prompt-probe:{next(serial)}")))
        original_model = replay.ScriptedModel
        current = [""]

        class RecordingModel(original_model):
            def __call__(self, prompt, *, purpose=None, **kwargs):
                record(f"conversations/{current[0]}/{purpose}", prompt, {"purpose": purpose, **kwargs})
                return super().__call__(prompt, purpose=purpose, **kwargs)

        with patch.object(replay, "ScriptedModel", RecordingModel):
            available = profiles()
            for case in load_cases():
                if not case.offline_ready or case.steps[0].event.kind == "backend":
                    continue
                current[0] = case.id
                result = replay.replay_case(case, available[case.profile])
                if result["status"] != "OFFLINE_PARTIAL_PASS":
                    raise AssertionError(f"Prompt capture replay failed: {case.id}: {result}")

        from app.agents import social_opening
        from app.services import localized_content, faq_grounding
        from app.prompts import hotel, capabilities
        from app.prompts.v2_turn import build_v2_turn_prompt
        from app.schemas.tasks import AgentTaskRequest, AgentTaskType
        from app.schemas.v2_turns import AgentTurnRequest
        request = AgentTurnRequest.model_validate(profiles()["hotel"])
        for locale in ("es-MX", "en-US"):
            request.guest.preferredLanguage = locale
            documents[f"builders/V2_AGENT_TURN/{locale}.prompt.txt"] = build_v2_turn_prompt(request)

        probe("boundary/V2_SOCIAL_OPENING", social_opening, lambda: social_opening.classify_social_opening("Hola"))
        # Isolate the translation cache so collection is repeatable even after unit tests.
        with patch.object(localized_content, "_cache", OrderedDict()):
            probe("boundary/V2_LOCALIZATION", localized_content, lambda: localized_content.translate_batch(
                ["Confirm 2 Pechuga  Herbal", "Change"], "es-MX", protected_values=["Pechuga  Herbal"], max_lengths=[100, 20]))
        search = {"query": "When does the pool close?", "candidateSetComplete": True, "matches": [
            {"catalogItemId": "SYNTHETIC-POOL", "question": "Pool hours", "answer": "The pool closes at 22:00."}]}
        probe("boundary/V2_FAQ_SEMANTIC_RETRIEVAL", faq_grounding,
              lambda: faq_grounding.resolve_semantic_faq(request, search, time.perf_counter()))
        from app.services import catalog_orders
        probe("boundary/V2_CATALOG_MATCH", catalog_orders, lambda: catalog_orders._semantic("classic burger", [
            {"id": "SYNTHETIC-BURGER", "code": "BURGER", "label": "Hamburguesa Clásica", "categoryLabel": "Hamburguesas"}]))
        from app.services.openai_client import OpenAiJsonResult
        from app.services.telemetry_client import OpenAiTokenUsage

        # Exercise the assembled planner prompt AND its validation-repair suffix.
        from app.agents import v2_turn_planner
        from app.agents.v2_scope_router import ScopeDecision
        from tests.conversation_regression.session import ConversationSession
        from tests.conversation_regression.library import Event
        task_case = next(c for c in load_cases() if c.id == "SC-015-ambiguous-tasks")
        planner_request = ConversationSession(task_case, profiles()["hotel"]).receive(Event(kind="guest", text="Sí, confirmo"))
        planner_request.activeOperations = planner_request.activeOperations[:1]
        operation = planner_request.activeOperations[0]
        task = operation.pendingConversationTasks[0]
        task.taskType = "SYNTHETIC_CONFIRMATION"
        task.requiredOutputSchema = {"type": "object", "required": ["confirmed"], "properties": {"confirmed": {"type": "boolean"}}}
        planner_request.conversation.focusedConversationTaskId = task.conversationTaskId
        planner_calls = []

        def invalid_then_capture(prompt, **kwargs):
            record("boundary/V2_AGENT_TURN_REPAIR", prompt, kwargs)
            planner_calls.append(prompt)
            if len(planner_calls) == 2:
                raise Captured()
            return OpenAiJsonResult({"disposition": "RESPONSE_READY", "messages": [{"purpose": "CONFIRMATION",
                "text": "Gracias por confirmar.", "language": "es-MX", "operationIds": [str(operation.operationId)],
                "conversationTaskIds": [str(task.conversationTaskId)]}]}, OpenAiTokenUsage(), "synthetic")

        with patch.object(v2_turn_planner, "call_openai_json_result", side_effect=invalid_then_capture), \
                patch.object(v2_turn_planner, "classify_hotel_scope", return_value=(ScopeDecision(
                    kind="CONTEXT_REPLY", offeringCode=None, relevantText="Sí, confirmo", hasRequestDetails=True,
                    containsUnrelatedTopic=False, confidence=1), OpenAiTokenUsage())), \
                patch.object(v2_turn_planner, "localize_response", side_effect=lambda req, resp, started: resp):
            try:
                v2_turn_planner.plan_v2_turn(planner_request)
            except Captured:
                pass
            else:
                raise AssertionError("Planner repair probe did not reach second model call")

        def faq_reply(prompt, **kwargs):
            if kwargs["purpose"] == "V2_FAQ_GROUNDING_CHECK":
                return stop("boundary/V2_FAQ_GROUNDING_CHECK")(prompt, **kwargs)
            return OpenAiJsonResult({"status": "ANSWERABLE", "sourceId": "SYNTHETIC-POOL", "sameSubject": True,
                "fullyAnswers": True, "conflictingSources": False, "confidence": 1.0,
                "supportingQuotes": ["The pool closes at 22:00."], "answer": "The pool closes at 22:00. Can I help further?"},
                OpenAiTokenUsage(), "synthetic")
        with patch.object(faq_grounding, "call_openai_json_result", side_effect=faq_reply):
            try:
                faq_grounding.resolve_semantic_faq(request, search, time.perf_counter())
            except Captured:
                pass
            else:
                raise AssertionError("FAQ verification probe did not reach model")

        for task in AgentTaskType:
            payload = AgentTaskRequest(taskType=task, taskId="synthetic-prompt-probe", latestMessage="Two teas, please",
                context={"language": "en-US"}, offeringCode="ROOM_SERVICE",
                allowedOfferings=[{"offeringCode": "ROOM_SERVICE", "name": "Room service"}],
                operation={"items": [{"name": "tea", "quantity": 2}]})
            documents[f"builders/capabilities/{task.value}.prompt.txt"] = capabilities.capability_prompt(payload, {"knowledgeResources": []})

        # Every legacy builder is included. Unknown arguments fail collection instead of
        # silently rendering an empty context or skipping a newly introduced builder.
        values = {"history_text": "GUEST: Two teas, please", "guest_message": "Two teas, please",
            "known_context": {"roomNumber": "TEST-101", "guestLanguage": "en-US"},
            "extraction": {"intent": "ROOM_SERVICE", "items": [{"name": "tea", "quantity": 2}]},
            "pending_order": {"items": [{"name": "tea", "quantity": 2}]}, "menu_knowledge": {},
            "purpose": "Ask for the missing field", "pending_reservation": {"serviceName": "Massage"},
            "staff_message": "Synthetic staff response", "staff_response": "Synthetic staff response",
            "status": "PENDING", "staff_status": "SOLVED", "staff_decision": "AVAILABLE",
            "from_phone_number": "+00000000000", "issue_description": "Synthetic maintenance issue"}
        for name, function in inspect.getmembers(hotel, inspect.isfunction):
            if not name.endswith("_prompt"):
                continue
            parameters = inspect.signature(function).parameters
            args = {key: values[key] for key, param in parameters.items() if key in values}
            missing = [key for key, param in parameters.items() if key not in args and param.default is inspect.Parameter.empty]
            if missing:
                raise AssertionError(f"Unconfigured legacy probe {name}: {missing}")
            documents[f"builders/legacy/{name}.prompt.txt"] = function(**args)
    return dict(sorted(documents.items()))
