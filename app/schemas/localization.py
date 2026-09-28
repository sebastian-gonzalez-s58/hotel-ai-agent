from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.conversation_language import normalize_locale


class PresentationText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000)
    maxLength: int = Field(ge=1, le=20000)


class LocalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messageId: UUID
    namespace: str = Field(min_length=1, max_length=100)
    locale: str = Field(max_length=35)
    texts: list[PresentationText] = Field(min_length=1, max_length=25)
    protectedValues: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("locale")
    @classmethod
    def valid_locale(cls, value):
        locale = normalize_locale(value)
        if locale is None:
            raise ValueError("Invalid target locale")
        return locale

    @model_validator(mode="after")
    def bounded_payload(self):
        if sum(len(item.text) for item in self.texts) > 30000:
            raise ValueError("Localization text budget exceeded")
        if sum(len(value) for value in self.protectedValues) > 12000:
            raise ValueError("Protected parameter budget exceeded")
        return self


class LocalizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    locale: str
    version: str = "1"
    texts: list[str]
