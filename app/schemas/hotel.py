from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["guest", "assistant", "staff", "system"]
    content: str


class SuggestedAction(BaseModel):
    id: str
    label: str


class Interaction(BaseModel):
    type: str = "BUTTONS"
    title: str | None = None
    body: str | None = None
    buttonText: str | None = None
    actions: list[SuggestedAction] = Field(default_factory=list)


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
        "SPA",
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
    interaction: Interaction | None = None


class FaqResponseRequest(BaseModel):
    guestMessage: str
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class FaqResponse(BaseModel):
    message: str
    answered: bool = True
    needsHumanAnswer: bool = False
    category: str | None = None
    interaction: Interaction | None = None


class RoomServiceConfirmationRequest(BaseModel):
    extraction: dict[str, Any]
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)


class RoomServiceConfirmationResponse(BaseModel):
    message: str
    pendingOrder: dict[str, Any]
    missingFields: list[str] = Field(default_factory=list)
    requestComplete: bool = False
    interaction: Interaction | None = None


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
    interaction: Interaction | None = None


class GenericMessageRequest(BaseModel):
    guestMessage: str | None = None
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class GenericMessageResponse(BaseModel):
    message: str
    interaction: Interaction | None = None


class SpaMenuResponseRequest(BaseModel):
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class SpaReservationDetailsRequest(BaseModel):
    guestMessage: str
    pendingReservation: dict[str, Any] = Field(default_factory=dict)
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class SpaReservationDetailsResponse(BaseModel):
    message: str
    pendingReservation: dict[str, Any]
    missingFields: list[
        Literal["serviceName", "reservationDate", "reservationTime"]
    ] = Field(default_factory=list)
    requestComplete: bool
    interaction: Interaction | None = None


class SpaReservationConfirmationRequest(BaseModel):
    guestMessage: str | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)
    conversationHistory: list[ConversationMessage] = Field(default_factory=list)
    knownContext: dict[str, Any] = Field(default_factory=dict)


class SpaReservationConfirmationResponse(BaseModel):
    message: str
    pendingReservation: dict[str, Any]
    interaction: Interaction | None = None


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
    interaction: Interaction | None = None


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
    interaction: Interaction | None = None


class UnmatchedGuestResponseRequest(BaseModel):
    guestMessage: str
    fromPhoneNumber: str | None = None
    knownContext: dict[str, Any] = Field(default_factory=dict)
