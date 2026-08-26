import unittest
from unittest.mock import patch
from uuid import UUID

from app.core.errors import AgentModelError
from app.agents.v2_turn_planner import (
    AGENT_TURN_RESPONSE_SCHEMA,
    _normalize_response_envelope,
    _normalize_guest_experience,
    _validate_plan,
    plan_v2_turn,
)
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_turn_endpoint import MESSAGE_ID, TURN_ID, payload


class V2TurnPlannerTest(unittest.TestCase):
    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_requests_the_v2_response_schema_from_openai(self, openai_call):
        request = AgentTurnRequest.model_validate(payload())
        openai_call.return_value = OpenAiJsonResult(
            payload={
                "disposition": "RESPONSE_READY",
                "messages": [{
                    "messageDraftId": "81000000-0000-0000-0000-000000000001",
                    "purpose": "ANSWER",
                    "text": "Hola, ¿en qué servicio puedo ayudarte?",
                    "language": "es-MX",
                    "operationIds": [],
                    "conversationTaskIds": [],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="resp-test",
        )

        response = plan_v2_turn(request)

        self.assertEqual(response.disposition, "RESPONSE_READY")
        self.assertEqual(response.usage.totalTokens, 15)
        self.assertEqual(
            openai_call.call_args.kwargs["response_schema"],
            AGENT_TURN_RESPONSE_SCHEMA,
        )
        self.assertEqual(
            openai_call.call_args.kwargs["response_schema_name"],
            "agent_turn_response_v2",
        )

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_retries_once_when_the_model_returns_an_invalid_plan(self, openai_call):
        request = AgentTurnRequest.model_validate(payload())
        invalid = OpenAiJsonResult(
            payload={"disposition": "RESPONSE_READY", "messages": []},
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            response_id="resp-invalid",
        )
        valid = OpenAiJsonResult(
            payload={
                "disposition": "RESPONSE_READY",
                "messages": [{
                    "purpose": "ANSWER",
                    "text": "Hola, ¿en qué servicio puedo ayudarte?",
                    "language": "es-MX",
                    "operationIds": [],
                    "conversationTaskIds": [],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="resp-valid",
        )
        openai_call.side_effect = [invalid, valid]

        response = plan_v2_turn(request)

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertEqual(2, openai_call.call_count)

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_retries_focused_task_with_an_explicit_completion_template(self, openai_call):
        operation_id = "90000000-0000-0000-0000-000000000001"
        task_id = "91000000-0000-0000-0000-000000000001"
        request_payload = payload()
        active_operation = operation(operation_id)
        active_operation["pendingConversationTasks"] = [conversation_task(task_id, operation_id)]
        request_payload["activeOperations"] = [active_operation]
        request_payload["conversation"]["focusedConversationTaskId"] = task_id
        request_payload["conversation"]["recentMessages"][0]["text"] = "Sí, ya quedó resuelto"
        request_payload["toolPolicy"] = {
            "allowedTools": ["COMPLETE_CONVERSATION_TASK"],
            "maxToolCalls": 1,
            "allowMultipleConversationTaskCompletions": False,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        invalid = OpenAiJsonResult(
            payload={
                "disposition": "RESPONSE_READY",
                "messages": [{
                    "purpose": "CONFIRMATION",
                    "text": "Gracias por confirmar que el problema se resolvio.",
                    "language": "es-MX",
                    "operationIds": [operation_id],
                    "conversationTaskIds": [task_id],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="resp-invalid-task",
        )
        valid = OpenAiJsonResult(
            payload={
                "disposition": "TOOL_CALLS_REQUIRED",
                "messages": [],
                "toolCalls": [{
                    "toolName": "COMPLETE_CONVERSATION_TASK",
                    "targetOperationId": operation_id,
                    "targetConversationTaskId": task_id,
                    "arguments": {
                        "conversationTaskId": task_id,
                        "expectedVersion": 4,
                        "result": {"confirmed": True},
                    },
                    "confidence": 1,
                    "evidenceMessageIds": [MESSAGE_ID],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=12, output_tokens=6, total_tokens=18),
            response_id="resp-valid-task",
        )
        openai_call.side_effect = [invalid, invalid, valid]

        response = plan_v2_turn(request)

        self.assertEqual("TOOL_CALLS_REQUIRED", response.disposition)
        self.assertEqual(3, openai_call.call_count)
        repair_prompt = openai_call.call_args_list[1].args[0]
        self.assertIn("exactly one COMPLETE_CONVERSATION_TASK", repair_prompt)
        self.assertIn(task_id, repair_prompt)
        self.assertIn(f"evidenceMessageIds=[{MESSAGE_ID}]", repair_prompt)
        self.assertIn('"confirmed":{"type":"boolean"}', repair_prompt)

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_converts_repeated_start_service_after_success_into_acknowledgement(self, openai_call):
        operation_id = "90000000-0000-0000-0000-000000000020"
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request_payload["toolPolicy"] = {
            "allowedTools": ["START_SERVICE"],
            "maxToolCalls": 2,
        }
        request_payload["previousToolResults"] = [{
            "toolCallId": "80000000-0000-0000-0000-000000000020",
            "toolName": "START_SERVICE",
            "status": "SUCCEEDED",
            "result": {
                "operationId": operation_id,
                "offeringCode": "ROOM_SERVICE",
                "referenceCode": "REQ-20260825-ABC12345",
                "lifecycle": "ACTIVE",
                "detailedStatus": "PROCESS_STARTED",
                "version": 1,
            },
        }]
        request = AgentTurnRequest.model_validate(request_payload)
        openai_call.return_value = OpenAiJsonResult(
            payload={
                "disposition": "TOOL_CALLS_REQUIRED",
                "messages": [],
                "toolCalls": [{
                    "toolName": "START_SERVICE",
                    "arguments": {
                        "offeringCode": "ROOM_SERVICE",
                        "input": {"items": [{"name": "Hamburguesa", "quantity": 1}]},
                    },
                    "confidence": 0.9,
                    "evidenceMessageIds": [MESSAGE_ID],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="resp-repeated-start",
        )

        response = plan_v2_turn(request)

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertEqual([], response.toolCalls)
        self.assertEqual(1, len(response.messages))
        self.assertIn("REQ-20260825-ABC12345", response.messages[0].text)
        self.assertEqual([UUID(operation_id)], response.messages[0].operationIds)
        self.assertEqual(1, openai_call.call_count)

    def test_normalizes_server_owned_response_envelope(self):
        request = AgentTurnRequest.model_validate(payload())
        model_message_id = "81000000-0000-0000-0000-000000000001"
        model_tool_call_id = "82000000-0000-0000-0000-000000000001"

        normalized = _normalize_response_envelope(
            request,
            {
                "disposition": "TOOL_CALLS_REQUIRED",
                "messages": [{"messageDraftId": model_message_id}],
                "toolCalls": [{"toolCallId": model_tool_call_id}],
            },
        )

        self.assertEqual(normalized["schemaVersion"], "2.0")
        self.assertEqual(normalized["agentTurnId"], TURN_ID)
        self.assertNotEqual(normalized["messages"][0]["messageDraftId"], model_message_id)
        self.assertNotEqual(normalized["toolCalls"][0]["toolCallId"], model_tool_call_id)
        UUID(normalized["messages"][0]["messageDraftId"])
        UUID(normalized["toolCalls"][0]["toolCallId"])
        self.assertEqual(normalized["warnings"], [])
        self.assertIsNone(normalized["detectedLanguage"])

    def test_rejects_tool_not_in_policy(self):
        request = AgentTurnRequest.model_validate(payload())
        response = AgentTurnResponse(
            schemaVersion="2.0",
            agentTurnId=UUID(TURN_ID),
            disposition="TOOL_CALLS_REQUIRED",
            messages=[],
            toolCalls=[{
                "toolCallId": "80000000-0000-0000-0000-000000000001",
                "toolName": "START_SERVICE",
                "arguments": {"offeringCode": "spa", "input": {}},
                "confidence": 1,
                "evidenceMessageIds": [],
            }],
            usage={"model": "test", "inputTokens": 0, "cachedInputTokens": 0,
                   "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0},
            warnings=[],
        )
        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_accepts_start_service_for_an_available_offering_with_confirmation(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000002",
            "toolName": "START_SERVICE",
            "arguments": {
                "offeringCode": "ROOM_SERVICE",
                "input": {"items": [{"name": "Hamburguesa", "quantity": 1}]},
                "guestConfirmationEvidenceMessageId": MESSAGE_ID,
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

    def test_rejects_start_service_for_an_unknown_offering(self):
        request_payload = payload()
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000003",
            "toolName": "START_SERVICE",
            "arguments": {"offeringCode": "INVENTED", "input": {}},
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_rejects_start_service_when_required_offering_input_is_blank(self):
        request_payload = payload()
        maintenance = offering()
        maintenance.update({
            "offeringCode": "MAINTENANCE",
            "name": "Mantenimiento",
            "inputSchema": {
                "type": "object",
                "required": ["issue"],
                "properties": {"issue": {"type": "string"}},
            },
            "requiresExplicitGuestConfirmation": False,
        })
        request_payload["availableOfferings"] = [maintenance]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000010",
            "toolName": "START_SERVICE",
            "arguments": {"offeringCode": "MAINTENANCE", "input": {"issue": ""}},
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        with self.assertRaisesRegex(AgentModelError, "non-empty value.*issue"):
            _validate_plan(request, response)

    def test_accepts_only_an_advertised_service_action_at_the_current_version(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        request_payload = payload()
        request_payload["activeOperations"] = [operation(operation_id)]
        request_payload["toolPolicy"] = {
            "allowedTools": ["EXECUTE_SERVICE_ACTION"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000004",
            "toolName": "EXECUTE_SERVICE_ACTION",
            "targetOperationId": operation_id,
            "arguments": {
                "operationId": operation_id,
                "actionCode": "CONFIRM",
                "expectedVersion": 3,
                "input": {"confirmed": True},
                "evidenceMessageIds": [MESSAGE_ID],
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

        response.toolCalls[0].arguments["actionCode"] = "INVENTED"
        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_accepts_completion_of_open_conversation_task_at_current_version(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        task_id = "91000000-0000-0000-0000-000000000001"
        request_payload = payload()
        active_operation = operation(operation_id)
        active_operation["pendingConversationTasks"] = [conversation_task(task_id, operation_id)]
        request_payload["activeOperations"] = [active_operation]
        request_payload["conversation"]["focusedConversationTaskId"] = task_id
        request_payload["toolPolicy"] = {
            "allowedTools": ["COMPLETE_CONVERSATION_TASK"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000005",
            "toolName": "COMPLETE_CONVERSATION_TASK",
            "targetOperationId": operation_id,
            "targetConversationTaskId": task_id,
            "arguments": {
                "conversationTaskId": task_id,
                "expectedVersion": 4,
                "result": {"confirmed": True},
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

    def test_rejects_stale_or_mismatched_conversation_task_completion(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        task_id = "91000000-0000-0000-0000-000000000001"
        request_payload = payload()
        active_operation = operation(operation_id)
        active_operation["pendingConversationTasks"] = [conversation_task(task_id, operation_id)]
        request_payload["activeOperations"] = [active_operation]
        request_payload["toolPolicy"] = {
            "allowedTools": ["COMPLETE_CONVERSATION_TASK"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000006",
            "toolName": "COMPLETE_CONVERSATION_TASK",
            "targetConversationTaskId": task_id,
            "arguments": {
                "conversationTaskId": "91000000-0000-0000-0000-000000000099",
                "expectedVersion": 3,
                "result": {"confirmed": True},
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_accepts_partial_conversation_task_progress(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        task_id = "91000000-0000-0000-0000-000000000001"
        request_payload = payload()
        active_operation = operation(operation_id)
        active_operation["pendingConversationTasks"] = [conversation_task(task_id, operation_id)]
        request_payload["activeOperations"] = [active_operation]
        request_payload["toolPolicy"] = {
            "allowedTools": ["SAVE_CONVERSATION_TASK_PROGRESS"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000007",
            "toolName": "SAVE_CONVERSATION_TASK_PROGRESS",
            "targetConversationTaskId": task_id,
            "arguments": {
                "conversationTaskId": task_id,
                "expectedVersion": 4,
                "partialResult": {"requestedDate": "2026-08-21"},
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

    def test_personalizes_greeting_and_builds_menu_from_available_offerings(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000001",
                "purpose": "ANSWER",
                "text": "¿En qué puedo ayudarte hoy?",
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
                "interaction": {
                    "type": "BUTTONS",
                    "title": "Menu",
                    "body": "Elige una opcion.",
                    "buttonText": "Opciones",
                    "options": [{"id": "stale", "label": "Opcion anterior"}],
                },
            }],
        })

        message = normalized["messages"][0]
        expected_text = (
            "Hola, Sebastian. \u00bfC\u00f3mo podemos ayudarte hoy? "
            "Por favor, elige una opci\u00f3n del men\u00fa."
        )
        self.assertEqual(expected_text, message["text"])
        self.assertEqual(expected_text, message["interaction"]["body"])
        self.assertEqual("Servicios del hotel", message["interaction"]["title"])
        self.assertEqual("Ver servicios", message["interaction"]["buttonText"])
        self.assertEqual("BUTTONS", message["interaction"]["type"])
        self.assertEqual(
            [{"id": "offering:ROOM_SERVICE", "label": "Servicio a la habitacion"[:24]}],
            message["interaction"]["options"],
        )

    def test_maintenance_selection_collects_issue_as_free_text(self):
        request_payload = payload()
        maintenance = offering()
        maintenance.update({
            "offeringCode": "MAINTENANCE",
            "name": "Mantenimiento",
            "inputSchema": {
                "type": "object",
                "required": ["issue"],
                "properties": {"issue": {"type": "string"}},
            },
            "requiresExplicitGuestConfirmation": False,
        })
        request_payload["availableOfferings"] = [maintenance]
        request_payload["conversation"]["recentMessages"][0]["text"] = "Mantenimiento"
        request_payload["conversation"]["recentMessages"][0][
            "interactionReplyId"
        ] = "offering:MAINTENANCE"
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000040",
                "purpose": "CLARIFICATION",
                "text": "Por favor, describe el problema de mantenimiento.",
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
                "interaction": {
                    "type": "LIST",
                    "title": "Tipo de problema",
                    "body": "Selecciona una opción",
                    "buttonText": "Ver opciones",
                    "options": [
                        {"id": "ac", "label": "Aire acondicionado"},
                        {"id": "door", "label": "Puerta"},
                        {"id": "other", "label": "Otro problema"},
                    ],
                },
            }],
        })

        self.assertEqual(
            "Por favor, describe el problema de mantenimiento.",
            normalized["messages"][0]["text"],
        )
        self.assertIsNone(normalized["messages"][0]["interaction"])

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_greeting_cannot_start_service_from_historical_maintenance_context(self, openai_call):
        operation_id = "90000000-0000-0000-0000-000000000030"
        request_payload = payload()
        maintenance = offering()
        maintenance.update({
            "offeringCode": "MAINTENANCE",
            "name": "Mantenimiento",
            "inputSchema": {
                "type": "object",
                "required": ["issue"],
                "properties": {"issue": {"type": "string"}},
            },
            "requiresExplicitGuestConfirmation": False,
        })
        request_payload["availableOfferings"] = [maintenance]
        request_payload["recentOperations"] = [operation(operation_id)]
        request_payload["toolPolicy"] = {
            "allowedTools": ["START_SERVICE", "GET_OPERATION_STATUS"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        openai_call.return_value = OpenAiJsonResult(
            payload={
                "disposition": "TOOL_CALLS_REQUIRED",
                "messages": [],
                "toolCalls": [{
                    "toolName": "START_SERVICE",
                    "arguments": {
                        "offeringCode": "MAINTENANCE",
                        "input": {"issue": "Problema de mantenimiento anterior"},
                    },
                    "confidence": 0.9,
                    "evidenceMessageIds": [MESSAGE_ID],
                }],
            },
            usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            response_id="resp-greeting-history",
        )

        response = plan_v2_turn(request)

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertEqual([], response.toolCalls)
        self.assertEqual(1, len(response.messages))
        self.assertIn("Hola, Sebastian", response.messages[0].text)
        self.assertEqual(
            "offering:MAINTENANCE",
            response.messages[0].interaction.options[0].id,
        )

    def test_adds_folio_acknowledgement_after_start_service(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request_payload["trigger"] = {"type": "TOOL_RESULTS"}
        request_payload["previousToolResults"] = [{
            "toolCallId": "80000000-0000-0000-0000-000000000009",
            "toolName": "START_SERVICE",
            "status": "SUCCEEDED",
            "result": {
                "operationId": operation_id,
                "referenceCode": "REQ-20260824-ABC12345",
                "offeringCode": "ROOM_SERVICE",
                "lifecycle": "ACTIVE",
                "detailedStatus": "PROCESS_STARTED",
                "version": 1,
            },
        }]
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000002",
                "purpose": "ANSWER",
                "text": (
                    'Tu solicitud de Servicio a la habitacion para "una barbacoa ancestral" '
                    "ha sido iniciada con el folio REQ-20260824-ABC12345."
                ),
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
                "interaction": None,
            }],
        })

        self.assertEqual(1, len(normalized["messages"]))
        self.assertIn("REQ-20260824-ABC12345", normalized["messages"][0]["text"])
        self.assertNotIn("barbacoa ancestral", normalized["messages"][0]["text"])
        self.assertEqual(
            (
                "La solicitud de servicio a la habitacion ha sido iniciada con el folio "
                "REQ-20260824-ABC12345. Recibirás actualizaciones por este medio; "
                "por favor, mantente atento."
            ),
            normalized["messages"][0]["text"],
        )
        self.assertEqual([operation_id], normalized["messages"][0]["operationIds"])

    def test_rejects_task_acknowledgement_before_successful_completion(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        task_id = "91000000-0000-0000-0000-000000000001"
        request_payload = payload()
        active_operation = operation(operation_id)
        active_operation["pendingConversationTasks"] = [conversation_task(task_id, operation_id)]
        request_payload["activeOperations"] = [active_operation]
        request_payload["conversation"]["focusedConversationTaskId"] = task_id
        request = AgentTurnRequest.model_validate(request_payload)
        response = AgentTurnResponse.model_validate({
            "schemaVersion": "2.0",
            "agentTurnId": request_payload["agentTurnId"],
            "disposition": "RESPONSE_READY",
            "detectedLanguage": "es-MX",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000010",
                "purpose": "CONFIRMATION",
                "text": "Gracias por confirmar que el problema se resolvió.",
                "language": "es-MX",
                "operationIds": [operation_id],
                "conversationTaskIds": [task_id],
                "interaction": None,
            }],
            "toolCalls": [],
            "updatedConversationSummary": None,
            "usage": {
                "model": "test",
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningTokens": 0,
                "totalTokens": 0,
            },
            "warnings": [],
        })

        with self.assertRaisesRegex(
            AgentModelError,
            "Complete the referenced conversation task before acknowledging it",
        ):
            _validate_plan(request, response)

        request_payload["previousToolResults"] = [{
            "toolCallId": "80000000-0000-0000-0000-000000000011",
            "toolName": "COMPLETE_CONVERSATION_TASK",
            "status": "SUCCEEDED",
            "result": {
                "conversationTaskId": task_id,
                "operationId": operation_id,
                "status": "COMPLETED",
            },
        }]
        completed_request = AgentTurnRequest.model_validate(request_payload)

        _validate_plan(completed_request, response)

    def test_accepts_status_lookup_by_folio_without_exposing_an_operation_id(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request_payload["toolPolicy"] = {
            "allowedTools": ["GET_OPERATION_STATUS"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000010",
            "toolName": "GET_OPERATION_STATUS",
            "arguments": {"referenceCode": "REQ-20260824-ABC12345"},
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)


def tool_response(tool_call):
    return AgentTurnResponse(
        schemaVersion="2.0",
        agentTurnId=UUID(TURN_ID),
        disposition="TOOL_CALLS_REQUIRED",
        messages=[],
        toolCalls=[tool_call],
        usage={"model": "test", "inputTokens": 0, "cachedInputTokens": 0,
               "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0},
        warnings=[],
    )


def offering():
    return {
        "offeringCode": "ROOM_SERVICE",
        "name": "Servicio a la habitacion",
        "description": "Alimentos y bebidas",
        "executionMode": "PROCESS",
        "inputSchema": {"type": "object"},
        "requiresExplicitGuestConfirmation": True,
    }


def operation(operation_id):
    return {
        "operationId": operation_id,
        "offeringCode": "ROOM_SERVICE",
        "lifecycle": "WAITING_FOR_GUEST",
        "detailedStatus": "WAITING_FOR_CONFIRMATION",
        "summary": "Pedido de room service",
        "availableActions": [{
            "actionCode": "CONFIRM",
            "description": "Confirmar el pedido",
            "inputSchema": {"type": "object"},
            "requiresExplicitGuestConfirmation": True,
        }],
        "pendingConversationTasks": [],
        "version": 3,
    }


def conversation_task(task_id, operation_id):
    return {
        "conversationTaskId": task_id,
        "operationId": operation_id,
        "processInstanceId": "process-stage6",
        "activityId": "WaitForStage6ConversationTaskResult",
        "taskType": "STAGE6_CONFIRMATION",
        "status": "OPEN",
        "requiredOutputSchema": {
            "type": "object",
            "required": ["confirmed"],
            "properties": {"confirmed": {"type": "boolean"}},
        },
        "partialResult": {},
        "context": {"reason": "Stage 6 verification"},
        "priority": "NORMAL",
        "version": 4,
        "createdAt": "2026-08-20T18:00:00Z",
    }


if __name__ == "__main__":
    unittest.main()
