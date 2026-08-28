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
        self.assertEqual(normalized["messages"][0]["operationIds"], [])
        self.assertEqual(normalized["messages"][0]["conversationTaskIds"], [])
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

    def test_offering_capture_uses_configured_delivery_options(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["conversation"]["recentMessages"][0].update({
            "text": "Servicio a la habitacion",
            "interactionReplyId": "offering:ROOM_SERVICE",
        })
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "TOOL_CALLS_REQUIRED",
            "messages": [],
            "toolCalls": [{"toolName": "START_SERVICE"}],
        })

        self.assertEqual("RESPONSE_READY", normalized["disposition"])
        self.assertEqual([], normalized["toolCalls"])
        message = normalized["messages"][0]
        self.assertEqual("¿Dónde deseas recibir tu pedido?", message["text"])
        self.assertEqual("LIST", message["interaction"]["type"])
        self.assertEqual(
            [
                "field:ROOM_SERVICE:deliveryLocation:ROOM",
                "field:ROOM_SERVICE:deliveryLocation:DOCK_1",
                "field:ROOM_SERVICE:deliveryLocation:DOCK_2",
                "field:ROOM_SERVICE:deliveryLocation:POOL_1",
                "field:ROOM_SERVICE:deliveryLocation:POOL_2",
            ],
            [option["id"] for option in message["interaction"]["options"]],
        )
        self.assertIn(
            '"pendingOffering":"ROOM_SERVICE"',
            normalized["updatedConversationSummary"],
        )

    def test_delivery_selection_advances_to_catalog_items_with_visible_url(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["conversation"]["recentMessages"][0].update({
            "text": "Muelle 1",
            "interactionReplyId": "field:ROOM_SERVICE:deliveryLocation:DOCK_1",
        })
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000041",
                "purpose": "CLARIFICATION",
                "text": "Selecciona productos",
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
                "interaction": {
                    "type": "BUTTONS",
                    "title": "Menu",
                    "body": "Selecciona productos",
                    "buttonText": "Abrir menu",
                    "options": [{"id": "open-menu", "label": "Abrir menu"}],
                },
            }],
        })

        message = normalized["messages"][0]
        self.assertIn("https://hotel.example/menu", message["text"])
        self.assertIsNone(message["interaction"])
        self.assertIn(
            '"deliveryLocation":"DOCK_1"',
            normalized["updatedConversationSummary"],
        )

    def test_catalog_items_answer_advances_to_order_confirmation(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["conversation"]["recentMessages"] = [
            {
                "messageId": "20000000-0000-0000-0000-000000000010",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Servicio a la habitación",
                "interactionReplyId": "offering:ROOM_SERVICE",
                "createdAt": "2026-08-26T12:21:53Z",
            },
            {
                "messageId": "20000000-0000-0000-0000-000000000011",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Habitación",
                "interactionReplyId": "field:ROOM_SERVICE:deliveryLocation:ROOM",
                "createdAt": "2026-08-26T12:22:05Z",
            },
            {
                "messageId": "20000000-0000-0000-0000-000000000012",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": (
                    "Por favor, indícame los productos que deseas pedir, incluyendo cantidades "
                    "y modificaciones."
                ),
                "createdAt": "2026-08-26T12:22:09Z",
            },
            {
                "messageId": MESSAGE_ID,
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "2 chilaquiles rellenos.",
                "createdAt": "2026-08-26T12:22:43Z",
            },
        ]
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000044",
                "purpose": "CLARIFICATION",
                "text": "Por favor, indícame los alimentos y bebidas que deseas pedir.",
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
            }],
            "toolCalls": [],
        })

        self.assertEqual("RESPONSE_READY", normalized["disposition"])
        self.assertEqual([], normalized["toolCalls"])
        message = normalized["messages"][0]
        self.assertIn("- 2 x chilaquiles rellenos", message["text"])
        self.assertIn("Lugar de entrega: Habitacion", message["text"])
        self.assertNotIn("indícame los alimentos", message["text"])
        self.assertEqual("BUTTONS", message["interaction"]["type"])
        self.assertEqual(
            [
                "confirmation:ROOM_SERVICE:CONFIRM",
                "confirmation:ROOM_SERVICE:CHANGE",
                "confirmation:ROOM_SERVICE:CANCEL",
            ],
            [option["id"] for option in message["interaction"]["options"]],
        )
        self.assertIn('"deliveryLocation":"ROOM"', normalized["updatedConversationSummary"])
        self.assertIn('"name":"chilaquiles rellenos"', normalized["updatedConversationSummary"])
        self.assertIn('"quantity":2', normalized["updatedConversationSummary"])

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_room_service_draft_normalizes_items_without_reasking_quantities(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request_payload["conversation"]["summary"] = (
            '{"pendingOffering":"ROOM_SERVICE","phase":"CAPTURING_ITEMS",'
            '"capturedFields":{"deliveryLocation":"DOCK_1"},'
            '"awaitingExplicitConfirmation":false}'
        )
        request_payload["conversation"]["recentMessages"] = [{
            "messageId": MESSAGE_ID,
            "direction": "INBOUND",
            "actor": "GUEST",
            "text": "Traeme mejor unas enchiladas con un refresco",
            "createdAt": "2026-08-27T18:10:00Z",
        }]

        response = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertIn("- 1 x enchiladas", response.messages[0].text)
        self.assertIn("- 1 x refresco", response.messages[0].text)
        self.assertIn("Lugar de entrega: Muelle 1", response.messages[0].text)
        self.assertNotIn("indícame las cantidades", response.messages[0].text.casefold())
        self.assertEqual("BUTTONS", response.messages[0].interaction.type)
        self.assertIn('"awaitingExplicitConfirmation":true', response.updatedConversationSummary)
        openai_call.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_room_service_change_preserves_destination_and_clears_old_items(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request_payload["conversation"]["summary"] = (
            '{"pendingOffering":"ROOM_SERVICE","phase":"AWAITING_CONFIRMATION",'
            '"capturedFields":{"deliveryLocation":"POOL_2","items":['
            '{"name":"pozole","quantity":1,"modifications":[]}]},'
            '"awaitingExplicitConfirmation":true}'
        )
        request_payload["conversation"]["recentMessages"] = [{
            "messageId": MESSAGE_ID,
            "direction": "INBOUND",
            "actor": "GUEST",
            "text": "Cambiar",
            "interactionReplyId": "confirmation:ROOM_SERVICE:CHANGE",
            "createdAt": "2026-08-27T18:11:00Z",
        }]

        response = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertIn("pedido completo", response.messages[0].text)
        self.assertIn('"deliveryLocation":"POOL_2"', response.updatedConversationSummary)
        self.assertNotIn('"items"', response.updatedConversationSummary)
        openai_call.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_room_service_confirmation_starts_once_with_authoritative_draft(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request_payload["conversation"]["summary"] = (
            '{"pendingOffering":"ROOM_SERVICE","phase":"AWAITING_CONFIRMATION",'
            '"capturedFields":{"deliveryLocation":"ROOM","items":['
            '{"name":"enchiladas","quantity":1,"modifications":[]},'
            '{"name":"refresco","quantity":1,"modifications":[]}]},'
            '"awaitingExplicitConfirmation":true}'
        )
        request_payload["conversation"]["recentMessages"] = [{
            "messageId": MESSAGE_ID,
            "direction": "INBOUND",
            "actor": "GUEST",
            "text": "Confirmar",
            "interactionReplyId": "confirmation:ROOM_SERVICE:CONFIRM",
            "createdAt": "2026-08-27T18:12:00Z",
        }]

        response = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("TOOL_CALLS_REQUIRED", response.disposition)
        self.assertEqual(1, len(response.toolCalls))
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual("ROOM", response.toolCalls[0].arguments["input"]["deliveryLocation"])
        self.assertEqual(2, len(response.toolCalls[0].arguments["input"]["items"]))
        self.assertEqual(UUID(MESSAGE_ID), response.toolCalls[0].evidenceMessageIds[0])
        openai_call.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_room_service_free_text_cancel_clears_pending_draft(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["conversation"]["summary"] = (
            '{"pendingOffering":"ROOM_SERVICE","phase":"AWAITING_CONFIRMATION",'
            '"capturedFields":{"deliveryLocation":"ROOM","items":['
            '{"name":"pozole","quantity":1,"modifications":[]}]},'
            '"awaitingExplicitConfirmation":true}'
        )
        request_payload["conversation"]["recentMessages"] = [{
            "messageId": MESSAGE_ID,
            "direction": "INBOUND",
            "actor": "GUEST",
            "text": "Cancela mi pedido",
            "createdAt": "2026-08-27T18:13:00Z",
        }]

        response = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertIn("fue cancelado", response.messages[0].text)
        self.assertEqual("{}", response.updatedConversationSummary)
        self.assertEqual([], response.toolCalls)
        openai_call.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_spa_capture_collects_service_date_and_time_before_starting(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_spa_offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request_payload["conversation"]["recentMessages"] = [
            {
                "messageId": "20000000-0000-0000-0000-000000000030",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Reservación de spa",
                "interactionReplyId": "offering:SPA",
                "createdAt": "2026-08-26T14:00:00Z",
            },
        ]
        first = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))
        self.assertIn("https://spa.example/catalog", first.messages[0].text)
        self.assertIn("cuál deseas reservar", first.messages[0].text)

        request_payload["conversation"]["recentMessages"].extend([
            {
                "messageId": "20000000-0000-0000-0000-000000000031",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": first.messages[0].text,
                "createdAt": "2026-08-26T14:00:01Z",
            },
            {
                "messageId": "20000000-0000-0000-0000-000000000032",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Masaje relajante",
                "createdAt": "2026-08-26T14:01:00Z",
            },
        ])
        second = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))
        self.assertEqual("¿Para qué fecha deseas hacer la reservación?", second.messages[0].text)

        request_payload["conversation"]["summary"] = second.updatedConversationSummary or ""
        request_payload["conversation"]["recentMessages"].extend([
            {
                "messageId": "20000000-0000-0000-0000-000000000033",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": second.messages[0].text,
                "createdAt": "2026-08-26T14:01:01Z",
            },
            {
                "messageId": "20000000-0000-0000-0000-000000000034",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "30 de agosto",
                "createdAt": "2026-08-26T14:02:00Z",
            },
        ])
        third = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))
        self.assertEqual("¿A qué hora deseas la reservación?", third.messages[0].text)

        request_payload["conversation"]["summary"] = third.updatedConversationSummary or ""
        request_payload["conversation"]["recentMessages"].extend([
            {
                "messageId": "20000000-0000-0000-0000-000000000035",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": third.messages[0].text,
                "createdAt": "2026-08-26T14:02:01Z",
            },
            {
                "messageId": MESSAGE_ID,
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "5 de la tarde",
                "createdAt": "2026-08-26T14:03:00Z",
            },
        ])
        fourth = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("TOOL_CALLS_REQUIRED", fourth.disposition)
        self.assertEqual([], fourth.messages)
        self.assertEqual(1, len(fourth.toolCalls))
        call = fourth.toolCalls[0]
        self.assertEqual("START_SERVICE", call.toolName)
        self.assertEqual("SPA", call.arguments["offeringCode"])
        self.assertEqual(
            {
                "serviceName": "Masaje relajante",
                "reservationDate": "30 de agosto",
                "reservationTime": "5 de la tarde",
            },
            call.arguments["input"],
        )
        openai_call.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    def test_faq_selection_collects_question_then_searches_approved_knowledge(self, openai_call):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_faq_offering()]
        request_payload["toolPolicy"] = {
            "allowedTools": ["SEARCH_KNOWLEDGE", "START_SERVICE"],
            "maxToolCalls": 2,
        }
        request_payload["conversation"]["recentMessages"] = [{
            "messageId": "20000000-0000-0000-0000-000000000050",
            "direction": "INBOUND",
            "actor": "GUEST",
            "text": "Preguntas frecuentes",
            "interactionReplyId": "offering:FAQ",
            "createdAt": "2026-08-26T15:00:00Z",
        }]

        first = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("RESPONSE_READY", first.disposition)
        self.assertEqual("¿Qué información necesitas sobre el hotel?", first.messages[0].text)
        self.assertEqual([], first.toolCalls)

        request_payload["conversation"]["recentMessages"].extend([
            {
                "messageId": "20000000-0000-0000-0000-000000000051",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": first.messages[0].text,
                "createdAt": "2026-08-26T15:00:01Z",
            },
            {
                "messageId": MESSAGE_ID,
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "¿A qué hora es el check-out?",
                "createdAt": "2026-08-26T15:01:00Z",
            },
        ])

        second = plan_v2_turn(AgentTurnRequest.model_validate(request_payload))

        self.assertEqual("TOOL_CALLS_REQUIRED", second.disposition)
        self.assertEqual([], second.messages)
        self.assertEqual(1, len(second.toolCalls))
        call = second.toolCalls[0]
        self.assertEqual("SEARCH_KNOWLEDGE", call.toolName)
        self.assertEqual("FAQ", call.arguments["offeringCode"])
        self.assertEqual("¿A qué hora es el check-out?", call.arguments["query"])
        self.assertEqual([UUID(MESSAGE_ID)], call.evidenceMessageIds)
        self.assertIn('"question":"¿A qué hora es el check-out?"', second.updatedConversationSummary)
        openai_call.assert_not_called()

    def test_old_catalog_capture_does_not_hijack_a_later_greeting(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["conversation"]["recentMessages"] = [
            {
                "messageId": "20000000-0000-0000-0000-000000000020",
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Habitación",
                "interactionReplyId": "field:ROOM_SERVICE:deliveryLocation:ROOM",
                "createdAt": "2026-08-26T12:22:05Z",
            },
            {
                "messageId": "20000000-0000-0000-0000-000000000021",
                "direction": "OUTBOUND",
                "actor": "ASSISTANT",
                "text": "La solicitud fue completada.",
                "createdAt": "2026-08-26T12:30:00Z",
            },
            {
                "messageId": MESSAGE_ID,
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Hola",
                "createdAt": "2026-08-26T13:00:00Z",
            },
        ]
        request = AgentTurnRequest.model_validate(request_payload)

        normalized = _normalize_guest_experience(request, {
            "disposition": "RESPONSE_READY",
            "messages": [{
                "messageDraftId": "81000000-0000-0000-0000-000000000045",
                "purpose": "ANSWER",
                "text": "Hola",
                "language": "es-MX",
                "operationIds": [],
                "conversationTaskIds": [],
            }],
            "toolCalls": [],
        })

        self.assertIn("Hola, Sebastian", normalized["messages"][0]["text"])
        self.assertNotIn("Confirmación de pedido", normalized["messages"][0]["text"])

    def test_rejects_start_service_with_unknown_configured_option(self):
        request_payload = payload()
        offering_payload = guided_room_service_offering()
        request_payload["availableOfferings"] = [offering_payload]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000042",
            "toolName": "START_SERVICE",
            "arguments": {
                "offeringCode": "ROOM_SERVICE",
                "input": {
                    "deliveryLocation": "BEACH",
                    "items": [{"name": "Hamburguesa", "quantity": 1}],
                },
                "guestConfirmationEvidenceMessageId": MESSAGE_ID,
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        with self.assertRaisesRegex(AgentModelError, "configured option code"):
            _validate_plan(request, response)

    def test_accepts_one_confirmed_room_service_start_with_configured_input(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [guided_room_service_offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request_payload["conversation"]["recentMessages"][0].update({
            "text": "Confirmar",
            "interactionReplyId": "room-service:confirm",
        })
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000043",
            "toolName": "START_SERVICE",
            "arguments": {
                "offeringCode": "ROOM_SERVICE",
                "input": {
                    "deliveryLocation": "DOCK_2",
                    "items": [
                        {"name": "Barbacoa ancestral", "quantity": 1},
                        {"name": "Cerveza Indio", "quantity": 2},
                    ],
                },
                "guestConfirmationEvidenceMessageId": MESSAGE_ID,
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)
        self.assertEqual(1, len(response.toolCalls))
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual(
            "DOCK_2",
            response.toolCalls[0].arguments["input"]["deliveryLocation"],
        )
        self.assertEqual(2, len(response.toolCalls[0].arguments["input"]["items"]))

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


def guided_room_service_offering():
    delivery_options = [
        {"id": "1", "code": "ROOM", "label": "Habitacion"},
        {"id": "2", "code": "DOCK_1", "label": "Muelle 1"},
        {"id": "3", "code": "DOCK_2", "label": "Muelle 2"},
        {"id": "4", "code": "POOL_1", "label": "Alberca 1"},
        {"id": "5", "code": "POOL_2", "label": "Alberca 2"},
    ]
    return {
        "offeringCode": "ROOM_SERVICE",
        "name": "Servicio a la habitacion",
        "description": "Alimentos y bebidas",
        "executionMode": "PROCESS",
        "inputSchema": {
            "type": "object",
            "required": ["deliveryLocation", "items"],
            "properties": {
                "deliveryLocation": {
                    "type": "string",
                    "title": "Lugar de entrega",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "SINGLE_SELECT",
                        "displayOrder": 10,
                        "introMessage": "¿Dónde deseas recibir tu pedido?",
                        "catalog": {
                            "code": "ROOM_SERVICE_DELIVERY_LOCATIONS",
                            "name": "Lugares de entrega",
                            "type": "OPTION_LIST",
                            "options": delivery_options,
                        },
                    },
                },
                "items": {
                    "type": "array",
                    "items": {"type": "object"},
                    "title": "Productos",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "CATALOG_ITEMS",
                        "displayOrder": 20,
                        "introMessage": (
                            "Consulta el menú digital e indica productos, cantidades y modificaciones."
                        ),
                        "catalog": {
                            "code": "ROOM_SERVICE_MENU",
                            "name": "Menu",
                            "type": "MENU",
                            "externalUrl": "https://hotel.example/menu",
                            "options": [],
                        },
                    },
                },
            },
            "additionalProperties": False,
        },
        "requiresExplicitGuestConfirmation": True,
    }


def guided_spa_offering():
    return {
        "offeringCode": "SPA",
        "name": "Reservación de spa",
        "description": "Servicios y tratamientos de SPA",
        "executionMode": "PROCESS",
        "inputSchema": {
            "type": "object",
            "required": ["serviceName", "reservationDate", "reservationTime"],
            "properties": {
                "serviceName": {
                    "type": "string",
                    "title": "Servicio de SPA",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "CATALOG_ITEMS",
                        "displayOrder": 10,
                        "introMessage": (
                            "En este enlace puedes consultar los servicios que ofrecemos en el SPA. "
                            "Indícanos cuál deseas reservar."
                        ),
                        "catalog": {
                            "code": "SPA_SERVICES",
                            "name": "Servicios de SPA",
                            "type": "BOOKABLE_SERVICE",
                            "externalUrl": "https://spa.example/catalog",
                            "options": [],
                        },
                    },
                },
                "reservationDate": {
                    "type": "string",
                    "format": "date",
                    "title": "Fecha deseada",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "DATE",
                        "displayOrder": 20,
                        "introMessage": "¿Para qué fecha deseas hacer la reservación?",
                    },
                },
                "reservationTime": {
                    "type": "string",
                    "format": "time",
                    "title": "Hora deseada",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "TIME",
                        "displayOrder": 30,
                        "introMessage": "¿A qué hora deseas la reservación?",
                    },
                },
            },
            "additionalProperties": False,
        },
        "requiresExplicitGuestConfirmation": False,
    }


def guided_faq_offering():
    return {
        "offeringCode": "FAQ",
        "name": "Preguntas frecuentes",
        "description": "Información y políticas del hotel",
        "executionMode": "KNOWLEDGE",
        "inputSchema": {
            "type": "object",
            "required": ["question"],
            "properties": {
                "question": {
                    "type": "string",
                    "title": "Pregunta",
                    "x-source": "GUEST",
                    "x-chatbotinn-capture": {
                        "inputMode": "FREE_TEXT",
                        "displayOrder": 10,
                        "introMessage": "¿Qué información necesitas sobre el hotel?",
                    },
                },
            },
            "additionalProperties": False,
        },
        "requiresExplicitGuestConfirmation": False,
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
