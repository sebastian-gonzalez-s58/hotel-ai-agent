from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.hotel import ConversationMessage, Interaction


class AgentTaskType(str, Enum):
    CLASSIFY_OFFERING = "CLASSIFY_OFFERING"
    EXTRACT_REQUIREMENT_VALUES = "EXTRACT_REQUIREMENT_VALUES"
    MATCH_CATALOG_ITEMS = "MATCH_CATALOG_ITEMS"
    GENERATE_CLARIFICATION = "GENERATE_CLARIFICATION"
    ANSWER_KNOWLEDGE_QUERY = "ANSWER_KNOWLEDGE_QUERY"
    GENERATE_GUEST_CONFIRMATION = "GENERATE_GUEST_CONFIRMATION"
    EVALUATE_GUEST_DECISION = "EVALUATE_GUEST_DECISION"
    REWRITE_STAFF_RESPONSE = "REWRITE_STAFF_RESPONSE"
    GENERATE_OPERATION_UPDATE = "GENERATE_OPERATION_UPDATE"
    GENERATE_HANDOFF_MESSAGE = "GENERATE_HANDOFF_MESSAGE"
    GENERATE_INACTIVITY_MESSAGE = "GENERATE_INACTIVITY_MESSAGE"
    SUMMARIZE_CONVERSATION = "SUMMARIZE_CONVERSATION"


class AgentToolName(str, Enum):
    LIST_AVAILABLE_OFFERINGS = "LIST_AVAILABLE_OFFERINGS"
    GET_OFFERING_DEFINITION = "GET_OFFERING_DEFINITION"
    GET_OFFERING_CATALOGS = "GET_OFFERING_CATALOGS"
    SEARCH_CATALOG_ITEMS = "SEARCH_CATALOG_ITEMS"
    SEARCH_KNOWLEDGE = "SEARCH_KNOWLEDGE"


class AgentTaskContext(BaseModel):
    conversationId: str | None = None
    operationId: str | None = None
    messageId: str | None = None
    unmatchedMessageId: str | None = None
    processInstanceId: str | None = None
    processActivityId: str | None = None
    requestId: str | None = None
    language: str | None = None
    channel: str = "WHATSAPP"
    hotelTimeZone: str | None = None


class AgentToolPolicy(BaseModel):
    allowedTools: list[AgentToolName] = Field(default_factory=list)
    allowedCatalogIds: list[str] = Field(default_factory=list)
    allowedCatalogCodes: list[str] = Field(default_factory=list)
    maxCatalogResults: int = Field(default=20, ge=1, le=100)


class AgentTaskRequest(BaseModel):
    taskType: AgentTaskType
    schemaVersion: int = Field(default=1, ge=1)
    taskId: str | None = None
    context: AgentTaskContext = Field(default_factory=AgentTaskContext)
    latestMessage: str | None = None
    staffMessage: str | None = None
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    offeringCode: str | None = None
    offering: dict[str, Any] | None = None
    allowedOfferings: list[dict[str, Any]] = Field(default_factory=list)
    operation: dict[str, Any] = Field(default_factory=dict)
    taskConfig: dict[str, Any] = Field(default_factory=dict)
    toolPolicy: AgentToolPolicy = Field(default_factory=AgentToolPolicy)


class ExtractedRequirementValue(BaseModel):
    fieldCode: str
    value: Any
    confidence: float | None = Field(default=None, ge=0, le=1)


class AgentCatalogSelection(BaseModel):
    catalogId: str | None = None
    catalogCode: str | None = None
    itemId: str
    itemCode: str | None = None
    quantity: int | float = Field(default=1, gt=0)
    optionIds: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class AgentEvidence(BaseModel):
    type: str
    id: str | None = None
    code: str | None = None
    title: str | None = None


class GeneratedTaskMessage(BaseModel):
    text: str
    interaction: Interaction | None = None


class AgentTokenUsage(BaseModel):
    inputTokens: int = Field(default=0, ge=0)
    cachedInputTokens: int = Field(default=0, ge=0)
    outputTokens: int = Field(default=0, ge=0)
    reasoningTokens: int = Field(default=0, ge=0)
    totalTokens: int = Field(default=0, ge=0)


class AgentTaskResponse(BaseModel):
    taskType: AgentTaskType
    schemaVersion: int = 1
    status: Literal[
        "COMPLETED",
        "NEEDS_CLARIFICATION",
        "NEEDS_HUMAN",
        "UNRESOLVED",
    ]
    confidence: float | None = Field(default=None, ge=0, le=1)
    offeringCode: str | None = None
    containsEmergency: bool = False
    extractedValues: list[ExtractedRequirementValue] = Field(default_factory=list)
    catalogSelections: list[AgentCatalogSelection] = Field(default_factory=list)
    missingFieldCodes: list[str] = Field(default_factory=list)
    decision: str | None = None
    complete: bool = False
    message: GeneratedTaskMessage | None = None
    summary: str | None = None
    evidence: list[AgentEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: AgentTokenUsage = Field(default_factory=AgentTokenUsage)


class AgentCapabilitiesResponse(BaseModel):
    schemaVersion: int = 1
    taskTypes: list[AgentTaskType]
    tools: list[AgentToolName]
