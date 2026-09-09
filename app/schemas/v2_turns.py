from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurnTriggerType(str, Enum):
    INBOUND_MESSAGE = "INBOUND_MESSAGE"
    CONVERSATION_TASK_CREATED = "CONVERSATION_TASK_CREATED"
    PROCESS_STATUS_CHANGED = "PROCESS_STATUS_CHANGED"
    STAFF_MESSAGE_READY = "STAFF_MESSAGE_READY"
    TIMER = "TIMER"
    TOOL_RESULTS = "TOOL_RESULTS"


class DomainToolName(str, Enum):
    LIST_AVAILABLE_OFFERINGS = "LIST_AVAILABLE_OFFERINGS"
    GET_OFFERING_DEFINITION = "GET_OFFERING_DEFINITION"
    SEARCH_CATALOG = "SEARCH_CATALOG"
    SEARCH_KNOWLEDGE = "SEARCH_KNOWLEDGE"
    LIST_ACTIVE_OPERATIONS = "LIST_ACTIVE_OPERATIONS"
    GET_OPERATION = "GET_OPERATION"
    GET_OPERATION_STATUS = "GET_OPERATION_STATUS"
    START_SERVICE = "START_SERVICE"
    SAVE_CONVERSATION_TASK_PROGRESS = "SAVE_CONVERSATION_TASK_PROGRESS"
    COMPLETE_CONVERSATION_TASK = "COMPLETE_CONVERSATION_TASK"
    EXECUTE_SERVICE_ACTION = "EXECUTE_SERVICE_ACTION"


class TurnTrigger(StrictModel):
    type: TurnTriggerType
    messageId: UUID | None = None
    conversationTaskId: UUID | None = None
    operationId: UUID | None = None
    eventCode: str | None = Field(default=None, max_length=100)
    eventPayload: dict[str, Any] = Field(default_factory=dict)


class HotelContext(StrictModel):
    hotelId: UUID
    hotelCode: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    timeZone: str = Field(min_length=1, max_length=100)
    defaultLanguage: str = Field(min_length=2, max_length=35)


class GuestContext(StrictModel):
    guestId: UUID
    stayId: UUID
    displayName: str = Field(min_length=1, max_length=255)
    roomNumber: str = Field(min_length=1, max_length=100)
    preferredLanguage: str = Field(min_length=2, max_length=35)
    openAccount: bool = False


class ConversationMessage(StrictModel):
    messageId: UUID
    direction: Literal["INBOUND", "OUTBOUND", "INTERNAL"]
    actor: Literal["GUEST", "ASSISTANT", "STAFF", "SYSTEM"]
    text: str = Field(max_length=20000)
    interactionReplyId: str | None = Field(default=None, max_length=256)
    operationIds: list[UUID] = Field(default_factory=list)
    conversationTaskIds: list[UUID] = Field(default_factory=list)
    createdAt: datetime


class ConversationContext(StrictModel):
    contactThreadId: UUID | None = None
    conversationId: UUID
    conversationRouteId: UUID
    channel: Literal["WHATSAPP", "EMAIL", "CONSOLE"]
    status: Literal["OPEN", "CLOSED"]
    focusedConversationTaskId: UUID | None = None
    summary: str = Field(max_length=20000)
    recentMessages: list[ConversationMessage] = Field(max_length=200)


class AvailableAction(StrictModel):
    actionCode: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    inputSchema: dict[str, Any]
    requiresExplicitGuestConfirmation: bool = False


class ConversationTaskSnapshot(StrictModel):
    conversationTaskId: UUID
    operationId: UUID
    processInstanceId: str | None = Field(default=None, min_length=1)
    activityId: str | None = Field(default=None, min_length=1, max_length=255)
    taskType: str = Field(min_length=1, max_length=100)
    status: Literal["OPEN"]
    requiredOutputSchema: dict[str, Any]
    partialResult: dict[str, Any]
    context: dict[str, Any]
    priority: Literal["LOW", "NORMAL", "HIGH", "URGENT"]
    expiresAt: datetime | None = None
    version: int = Field(ge=0)
    createdAt: datetime


class OperationSnapshot(StrictModel):
    operationId: UUID
    referenceCode: str | None = Field(default=None, max_length=32)
    offeringCode: str = Field(min_length=1, max_length=100)
    lifecycle: Literal[
        "ACTIVE", "WAITING_FOR_GUEST", "WAITING_FOR_STAFF", "COMPLETED", "CANCELLED", "FAILED"
    ]
    detailedStatus: str = Field(min_length=1, max_length=100)
    summary: str = Field(max_length=4000)
    input: dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
    completedAt: datetime | None = None
    processInstanceIds: list[str] = Field(default_factory=list)
    availableActions: list[AvailableAction]
    pendingConversationTasks: list[ConversationTaskSnapshot]
    version: int = Field(ge=0)


class OfferingCapability(StrictModel):
    offeringCode: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=4000)
    executionMode: Literal["KNOWLEDGE", "PROCESS", "INTEGRATION"]
    inputSchema: dict[str, Any]
    catalogCodes: list[str] = Field(default_factory=list)
    requiresExplicitGuestConfirmation: bool = False


class ToolPolicy(StrictModel):
    allowedTools: list[DomainToolName]
    maxToolCalls: int = Field(ge=0, le=20)
    allowMultipleConversationTaskCompletions: bool = True


class ToolError(StrictModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    toolCallId: UUID
    toolName: str = Field(min_length=1, max_length=100)
    status: Literal["SUCCEEDED", "REJECTED", "FAILED"]
    result: dict[str, Any] | None = None
    error: ToolError | None = None


class AgentTurnRequest(StrictModel):
    schemaVersion: Literal["2.0"]
    agentTurnId: UUID
    traceId: str = Field(min_length=1, max_length=100)
    trigger: TurnTrigger
    hotel: HotelContext
    guest: GuestContext
    conversation: ConversationContext
    activeOperations: list[OperationSnapshot] = Field(max_length=100)
    recentOperations: list[OperationSnapshot] = Field(default_factory=list, max_length=100)
    availableOfferings: list[OfferingCapability] = Field(max_length=100)
    toolPolicy: ToolPolicy
    previousToolResults: list[ToolResult] = Field(max_length=50)
    createdAt: datetime


class DomainToolCall(StrictModel):
    toolCallId: UUID
    toolName: DomainToolName
    targetOperationId: UUID | None = None
    targetConversationTaskId: UUID | None = None
    arguments: dict[str, Any]
    confidence: float = Field(ge=0, le=1)
    evidenceMessageIds: list[UUID]


class InteractionOption(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=24)


class AgentInteraction(StrictModel):
    type: Literal["BUTTONS", "LIST"]
    title: str | None = Field(default=None, max_length=60)
    body: str = Field(min_length=1, max_length=1024)
    buttonText: str | None = Field(default=None, max_length=20)
    options: list[InteractionOption] = Field(min_length=1, max_length=10)


class AgentMessage(StrictModel):
    messageDraftId: UUID
    purpose: Literal["ANSWER", "CLARIFICATION", "CONFIRMATION", "STATUS_UPDATE", "HANDOFF", "REMINDER", "CLOSURE"]
    text: str = Field(min_length=1, max_length=20000)
    language: str = Field(min_length=2, max_length=35)
    operationIds: list[UUID]
    conversationTaskIds: list[UUID]
    interaction: AgentInteraction | None = None


class AgentUsage(StrictModel):
    model: str = Field(min_length=1, max_length=255)
    inputTokens: int = Field(ge=0)
    cachedInputTokens: int = Field(ge=0)
    outputTokens: int = Field(ge=0)
    reasoningTokens: int = Field(ge=0)
    totalTokens: int = Field(ge=0)
    latencyMs: int | None = Field(default=None, ge=0)


class AgentTurnResponse(StrictModel):
    schemaVersion: Literal["2.0"]
    agentTurnId: UUID
    disposition: Literal["RESPONSE_READY", "TOOL_CALLS_REQUIRED", "NO_ACTION", "HANDOFF_REQUIRED"]
    detectedLanguage: str | None = Field(default=None, max_length=35)
    messages: list[AgentMessage] = Field(max_length=10)
    toolCalls: list[DomainToolCall] = Field(max_length=20)
    updatedConversationSummary: str | None = Field(default=None, max_length=20000)
    usage: AgentUsage
    warnings: list[str] = Field(max_length=20)
