from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["guest", "assistant", "staff", "system"]
    content: str


class HotelConversationRequest(BaseModel):
    conversationId: str | None = None
    guestMessage: str
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class HotelExtractionResponse(BaseModel):
    intent: Literal[
        "FAQ",
        "ROOM_SERVICE",
        "MAINTENANCE",
        "HOUSEKEEPING",
        "RESTAURANT_RESERVATION",
        "COMPLAINT",
        "HUMAN_REVIEW",
        "UNKNOWN",
    ]
    confidence: float
    containsEmergency: bool
    roomNumber: str | None = None
    language: str | None = None
    missingFields: list[str]
    extractedEntities: dict[str, Any]
    requestComplete: bool


class ClarificationRequest(BaseModel):
    extraction: dict[str, Any]
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)


class ClarificationResponse(BaseModel):
    message: str


class FaqResponseRequest(BaseModel):
    guestMessage: str
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class FaqResponse(BaseModel):
    message: str
    answered: bool = True
    needsHumanAnswer: bool = False
    category: str | None = None


class RoomServiceConfirmationRequest(BaseModel):
    extraction: dict[str, Any]
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)


class RoomServiceConfirmationResponse(BaseModel):
    message: str
    pendingOrder: dict[str, Any]


class RoomServiceConfirmationEvaluationRequest(BaseModel):
    guestMessage: str
    pendingOrder: dict[str, Any]
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)


class RoomServiceConfirmationEvaluationResponse(BaseModel):
    confirmationAction: Literal[
        "CONFIRMED",
        "CHANGE_REQUESTED",
        "CANCELLED",
        "UNCLEAR",
    ]
    updatedOrder: dict[str, Any]
    message: str


class GenericMessageRequest(BaseModel):
    guestMessage: str | None = None
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class GenericMessageResponse(BaseModel):
    message: str


class SpaMenuResponseRequest(BaseModel):
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class SpaReservationConfirmationRequest(BaseModel):
    guestMessage: str | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class SpaReservationConfirmationResponse(BaseModel):
    message: str
    pendingReservation: dict[str, Any]


class SpaReservationConfirmationEvaluationRequest(BaseModel):
    guestMessage: str
    pendingReservation: dict[str, Any]
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)


class SpaReservationConfirmationEvaluationResponse(BaseModel):
    confirmationAction: Literal[
        "CONFIRMED",
        "CHANGE_REQUESTED",
        "CANCELLED",
        "UNCLEAR",
    ]
    updatedReservation: dict[str, Any]
    message: str


class MaintenanceInitialResponseRequest(BaseModel):
    extraction: dict[str, Any] = Field(default_factory=dict)
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class MaintenanceStaffUpdateResponseRequest(BaseModel):
    staffStatus: Literal["SOLVED", "FURTHER_STEPS_REQUIRED"]
    staffMessage: str | None = None
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class MaintenanceGuestResolutionEvaluationRequest(BaseModel):
    guestMessage: str
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class MaintenanceGuestResolutionEvaluationResponse(BaseModel):
    guestConfirmedResolved: bool
    message: str


class UnmatchedGuestResponseRequest(BaseModel):
    guestMessage: str
    fromPhoneNumber: str | None = None
    knownContext: dict[str, Any] = Field(default_factory=dict)
