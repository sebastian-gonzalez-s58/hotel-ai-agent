"""Carry actual V2 responses forward; expected checks never write conversation state.

This is an agent-context adapter, not a replacement for Spring integration tests.
It can simulate explicit empty-FAQ searches and successful starts linked to actual
proposed calls. It cannot execute domain tools, change sessions or deliver messages.
"""
import json
from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse, ConversationMessage, TurnTrigger, TurnTriggerType, ToolResult, OperationSnapshot


def structured_summary(text):
    for line in reversed(text.strip().splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


class ConversationSession:
    def __init__(self, case, profile):
        self.case_id = case.id
        self.request = AgentTurnRequest.model_validate(profile).model_copy(deep=True)
        self.request.conversation.summary = json.dumps(case.initial_summary, ensure_ascii=False)
        self.request.conversation.recentMessages = []
        payload = self.request.model_dump(mode="json")
        terminal = {'COMPLETED', 'CANCELLED', 'FAILED'}
        payload["activeOperations"] = [o for o in case.initial_operations if o['lifecycle'] not in terminal]
        payload["recentOperations"] = [o for o in case.initial_operations if o['lifecycle'] in terminal]
        self.request = AgentTurnRequest.model_validate(payload)
        self.request.guest.preferredLanguage = case.locale
        self.language = {"version": 1, "source": case.language_source,
                         "explicit": case.language_source == "EXPLICIT"}
        self.index = 0
        self.pending = False
        self.blocked_by_tools = False
        self.pending_tools = []
        self.synthetic_operations = []
        self.responses = []
        self.room_button_bindings = {}
        self.pending_room_token = None

    def receive(self, event):
        if event.kind == "tool_result":
            return self.receive_synthetic_tool_result(event)
        if event.kind != "guest":
            raise NotImplementedError("Backend and tool events need the integration adapter")
        if self.pending or self.blocked_by_tools:
            raise RuntimeError("Previous turn must finish, including its tool results")
        reply_id = event.reply_id
        if event.reply_from_turn is not None:
            candidates = (reversed(self.responses) if event.reply_from_turn == -1 else
                          [self.responses[event.reply_from_turn - 1]])
            prefix, action = reply_id.rsplit(":", 1)
            reply_id = next((option.id for response in candidates for message in response.messages
                             if message.interaction for option in message.interaction.options
                             if option.id.startswith(prefix + ":") and option.id.endswith(":" + action)), None)
            if reply_id is None:
                raise ValueError("Referenced confirmation option was never emitted")
        self.index += 1
        self.request.createdAt += timedelta(seconds=10)
        message_id = uuid5(NAMESPACE_URL, f"synthetic:{self.case_id}:inbound:{self.index}")
        self.request.agentTurnId = uuid5(NAMESPACE_URL, f"synthetic:{self.case_id}:turn:{self.index}")
        self.request.traceId = f"synthetic:{self.case_id}:{self.index}"
        self.request.previousToolResults = []
        self.request.trigger = TurnTrigger(type="INBOUND_MESSAGE", messageId=message_id,
                                           eventPayload={"languageContext": dict(self.language)})
        parts = (reply_id or '').split(':')
        if len(parts) == 4 and parts[:2] == ['confirmation', 'ROOM_SERVICE'] and parts[2] in self.room_button_bindings:
            self.request.trigger.eventPayload['roomServiceButtonContext'] = {'operationId': self.room_button_bindings[parts[2]]}
        self.request.conversation.recentMessages.append(ConversationMessage(
            messageId=message_id, direction="INBOUND", actor="GUEST", text=event.text,
            interactionReplyId=reply_id, createdAt=self.request.createdAt,
        ))
        # Match the production snapshot window, never the expected state in fixtures.
        self.request.conversation.recentMessages = self.request.conversation.recentMessages[-100:]
        self.request = AgentTurnRequest.model_validate(self.request.model_dump())
        self.pending = True
        return self.request.model_copy(deep=True)

    def receive_synthetic_tool_result(self, event):
        """Explicit simulation of search/start results; never executes domain services.

        Input and IDs derive from the actual proposed call, not the case's assertions.
        Backend persistence is checked separately by the Spring integration runner.
        """
        if self.pending or len(self.pending_tools) != 1 or not self.blocked_by_tools:
            raise ValueError("A synthetic result requires exactly one outstanding tool")
        call = self.pending_tools[0]
        name = call.toolName.value
        if event.action == "reply_to_last_tool" and name == "SEARCH_KNOWLEDGE":
            if event.data != {"status": "SUCCEEDED", "result": {"status": "NO_MATCH", "matches": []}}:
                raise ValueError("Only the explicit empty-knowledge simulation is supported")
            result = {"query": call.arguments["query"], "matchStatus": "SEMANTIC_REVIEW_REQUIRED", "confidence": 0,
                      "matches": [], "retrievalMode": "SEMANTIC_V1", "candidateSetComplete": True}
        elif event.action == "fail_service_start" and name == "START_SERVICE" and not event.data:
            self.request.previousToolResults.append(ToolResult(toolCallId=call.toolCallId, toolName=name,
                status="FAILED", error={"code": "TOOL_EXECUTION_FAILED", "message": "Synthetic failure", "retryable": False}))
            self.request.trigger = self.request.trigger.model_copy(update={"type": TurnTriggerType.TOOL_RESULTS})
            self.pending_tools = []
            self.blocked_by_tools = False
            self.pending = True
            return AgentTurnRequest.model_validate(self.request.model_dump())
        elif event.action in {"complete_faq_handoff", "complete_service_start"} and name == "START_SERVICE" and not event.data:
            code = call.arguments["offeringCode"]
            if event.action == "complete_faq_handoff" and code != "FAQ":
                raise ValueError("FAQ handoff result cannot complete a different offering")
            result = {"operationId": str(uuid5(NAMESPACE_URL, f"synthetic:{self.case_id}:operation:{len(self.synthetic_operations)}")),
                      "offeringCode": code, "referenceCode": f"TEST-{len(self.synthetic_operations)+1}",
                      "lifecycle": "ACTIVE", "detailedStatus": "STARTED", "input": call.arguments["input"]}
            self.synthetic_operations.append(result)
            if code == 'ROOM_SERVICE' and self.pending_room_token:
                self.room_button_bindings[self.pending_room_token] = result['operationId']
            self.request.activeOperations.append(OperationSnapshot(**result, summary="Synthetic started operation",
                availableActions=[], pendingConversationTasks=[], version=1))
        else:
            raise ValueError(f"Unsupported synthetic tool result: {event.action}/{name}")
        self.request.previousToolResults.append(ToolResult(toolCallId=call.toolCallId, toolName=name,
                                                           status="SUCCEEDED", result=result))
        self.request.trigger = self.request.trigger.model_copy(update={"type": TurnTriggerType.TOOL_RESULTS})
        self.pending_tools = []
        self.blocked_by_tools = False
        self.pending = True
        return AgentTurnRequest.model_validate(self.request.model_dump())

    def accept(self, response):
        response = AgentTurnResponse.model_validate(response)
        if not self.pending or response.agentTurnId != self.request.agentTurnId:
            raise ValueError("Response does not belong to the pending turn")
        if response.languageDecision and response.languageDecision.messageId != self.request.trigger.messageId:
            raise ValueError("Language decision is not supported by the current message")
        self.pending = False
        self.responses.append(response.model_copy(deep=True))
        if any(c.toolName.value == 'START_SERVICE' and c.arguments.get('offeringCode') == 'ROOM_SERVICE' for c in response.toolCalls):
            self.pending_room_token = structured_summary(self.request.conversation.summary).get('roomServiceConfirmation', {}).get('token')
        # Spring ignores null/blank summaries. '{}' is an explicit, valid empty state.
        summary = response.updatedConversationSummary
        completes_task = any(c.toolName.value == "COMPLETE_CONVERSATION_TASK" for c in response.toolCalls)
        if summary and summary.strip() and not completes_task:
            self.request.conversation.summary = summary.strip()
        if response.languageDecision:
            decision = response.languageDecision
            self.request.guest.preferredLanguage = decision.locale
            self.language.update(source=decision.source, explicit=decision.source == "EXPLICIT")
        for i, message in enumerate(response.messages):
            self.request.conversation.recentMessages.append(ConversationMessage(
                messageId=uuid5(NAMESPACE_URL, f"synthetic:{self.case_id}:outbound:{self.index}:{i}"),
                direction="OUTBOUND", actor="ASSISTANT", text=message.text,
                operationIds=message.operationIds, conversationTaskIds=message.conversationTaskIds,
                createdAt=self.request.createdAt + timedelta(seconds=1),
            ))
        self.blocked_by_tools = bool(response.toolCalls)
        self.pending_tools = list(response.toolCalls)
        return {
            "state": structured_summary(self.request.conversation.summary),
            "response": response.model_dump(mode="json"),
            "guest": self.request.guest.model_dump(mode="json"),
            "text": "\n".join(m.text for m in response.messages),
            "tool_names": [c.toolName.value for c in response.toolCalls],
            "tools": [c.model_dump(mode="json") for c in response.toolCalls],
            "message_purposes": [m.purpose for m in response.messages],
            "environment": {"source": "synthetic-tool-results", "createdOperations": {code: sum(o["offeringCode"] == code for o in self.synthetic_operations)
                                      for code in {o["offeringCode"] for o in self.synthetic_operations}},
                            "operationInputs": {o["offeringCode"]: o["input"] for o in self.synthetic_operations}},
        }


MISSING = object()


def pointer(value, path):
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value.get(part, MISSING)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(strict_equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def check_observation(observed, checks):
    errors = []
    for check in checks:
        actual = pointer(observed, check.path)
        if check.op == "exists":
            ok = actual is not MISSING
        elif check.op == "absent":
            ok = actual is MISSING
        elif actual is MISSING:
            ok = False  # Missing data must not satisfy a negative assertion.
        elif check.op == "equals":
            ok = strict_equal(actual, check.value)
        elif check.op in {"contains", "not_contains"}:
            try:
                present = (any(strict_equal(item, check.value) for item in actual) if isinstance(actual, list)
                           else check.value in actual if isinstance(actual, (dict, str)) else None)
            except TypeError:
                present = None
            ok = present is not None and (present if check.op == "contains" else not present)
        if not ok:
            errors.append({"path": check.path, "op": check.op, "expected": check.value,
                           "actual": "<missing>" if actual is MISSING else actual})
    return errors
