import unittest
from unittest.mock import patch

from app.agents.schema_validation import satisfies_schema
from app.agents.v2_turn_planner import _satisfies_required_output_schema, _validate_conversation_task_call
from app.core.errors import AgentModelError
from app.schemas.v2_turns import DomainToolCall, DomainToolName
from tests.test_spa_turns import VALUES, spa_operation, spa_schema
from uuid import uuid4


class SchemaValidationTest(unittest.TestCase):
    def test_allof_if_then_does_not_accept_incomplete_update(self):
        schema = spa_schema()
        schema.pop("oneOf")
        schema["allOf"] = [{"if": {"properties": {"decision": {"const": "UPDATE"}}},
                            "then": {"required": list(VALUES)}}]
        for validator in (satisfies_schema, _satisfies_required_output_schema):
            self.assertFalse(validator({"decision": "UPDATE"}, schema))
            self.assertFalse(validator({"decision": "UPDATE", "serviceName": "Masaje"}, schema))
            self.assertTrue(validator({"decision": "UPDATE", **VALUES}, schema))
            self.assertTrue(validator({"decision": "CANCEL"}, schema))

    def test_oneof_update_and_cancel_contract(self):
        self.assertTrue(satisfies_schema({"decision": "CANCEL"}, spa_schema()))
        self.assertTrue(satisfies_schema({"decision": "UPDATE", **VALUES}, spa_schema()))
        for invalid in ({}, {"decision": "ACCEPT"}, {"decision": "CHANGE"}, {"decision": "UPDATE"},
                        {"decision": "UPDATE", **VALUES, "serviceName": ""},
                        {"decision": "UPDATE", **VALUES, "serviceName": "x" * 201},
                        {"decision": "UPDATE", **VALUES, "reservationDate": "2026-02-30"},
                        {"decision": "UPDATE", **VALUES, "reservationDate": "manana"},
                        {"decision": "UPDATE", **VALUES, "reservationTime": "5 pm"},
                        {"decision": "UPDATE", **VALUES, "reservationTime": "17:00:00"}):
            with self.subTest(invalid=invalid):
                self.assertFalse(satisfies_schema(invalid, spa_schema()))

    def test_nested_constraints_refs_boolean_schemas_and_additional_properties(self):
        schema = {"type": "object", "required": ["items"], "additionalProperties": False,
                  "$defs": {"item": {"type": "integer", "minimum": 1}},
                  "properties": {"items": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/item"}}}}
        self.assertTrue(satisfies_schema({"items": [1, 2]}, schema))
        for invalid in ({"items": []}, {"items": [True]}, {"items": [0]}, {"items": [1], "extra": 1}):
            self.assertFalse(satisfies_schema(invalid, schema))
        self.assertFalse(satisfies_schema({}, False))
        self.assertTrue(satisfies_schema({}, True))

    def test_invalid_schema_and_remote_refs_fail_closed_without_network(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network not allowed")):
            self.assertFalse(satisfies_schema({}, {"$ref": "https://example.com/schema"}))
            self.assertFalse(satisfies_schema({}, {"type": "not-a-type"}))
            self.assertFalse(satisfies_schema({}, {"$schema": "https://example.com/draft"}))

    def test_complete_rejects_incomplete_update_but_allows_partial_progress(self):
        task = spa_operation("SPA_RESERVATION_CHANGE_DETAILS").pendingConversationTasks[0]
        call = DomainToolCall(toolCallId=uuid4(), toolName="COMPLETE_CONVERSATION_TASK",
                              targetConversationTaskId=task.conversationTaskId, targetOperationId=task.operationId,
                              arguments={"conversationTaskId": str(task.conversationTaskId),
                                         "expectedVersion": task.version, "result": {"decision": "UPDATE"}},
                              confidence=1, evidenceMessageIds=[uuid4()])
        with self.assertRaisesRegex(AgentModelError, "requiredOutputSchema"):
            _validate_conversation_task_call(call, {task.conversationTaskId: task})
        call.toolName = DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS
        call.arguments["partialResult"] = call.arguments.pop("result")
        _validate_conversation_task_call(call, {task.conversationTaskId: task})
        call.arguments["partialResult"].update(VALUES)
        with self.assertRaisesRegex(AgentModelError, "use COMPLETE_CONVERSATION_TASK"):
            _validate_conversation_task_call(call, {task.conversationTaskId: task})


if __name__ == "__main__":
    unittest.main()
