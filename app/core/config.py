import os

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    app_name: str
    app_version: str
    environment: str
    openai_api_key: str | None
    openai_model: str
    openai_timeout_seconds: float
    agent_internal_token: str | None
    max_guest_message_chars: int
    max_history_messages: int
    max_context_chars: int
    request_timeout_seconds: float
    max_concurrent_requests: int
    enable_debug_endpoints: bool
    chatbotinn_api_base_url: str | None
    chatbotinn_api_internal_token: str | None
    chatbotinn_api_timeout_seconds: float
    knowledge_cache_ttl_seconds: int

    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "chatbotinn-agent")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        self.environment = os.getenv("APP_ENV", "local")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.openai_timeout_seconds = _get_float("OPENAI_TIMEOUT_SECONDS", 20.0)
        self.agent_internal_token = os.getenv("AGENT_INTERNAL_TOKEN")
        self.max_guest_message_chars = _get_int("MAX_GUEST_MESSAGE_CHARS", 4000)
        self.max_history_messages = _get_int("MAX_HISTORY_MESSAGES", 40)
        self.max_context_chars = _get_int("MAX_CONTEXT_CHARS", 12000)
        self.request_timeout_seconds = _get_float("REQUEST_TIMEOUT_SECONDS", 25.0)
        self.max_concurrent_requests = _get_int("MAX_CONCURRENT_REQUESTS", 20)
        self.enable_debug_endpoints = _get_bool("ENABLE_DEBUG_ENDPOINTS", False)
        self.chatbotinn_api_base_url = os.getenv("CHATBOTINN_API_BASE_URL")
        self.chatbotinn_api_internal_token = os.getenv("CHATBOTINN_API_INTERNAL_TOKEN") or self.agent_internal_token
        self.chatbotinn_api_timeout_seconds = _get_float("CHATBOTINN_API_TIMEOUT_SECONDS", 10.0)
        self.knowledge_cache_ttl_seconds = _get_int("KNOWLEDGE_CACHE_TTL_SECONDS", 60)

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def is_internal_token_configured(self) -> bool:
        return bool(self.agent_internal_token and self.agent_internal_token.strip())

    @property
    def is_chatbotinn_api_configured(self) -> bool:
        return bool(
            self.chatbotinn_api_base_url
            and self.chatbotinn_api_base_url.strip()
            and self.chatbotinn_api_internal_token
            and self.chatbotinn_api_internal_token.strip()
        )


settings = Settings()
