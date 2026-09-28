"""Explicit model boundary for offline and opt-in synthetic live evaluations."""
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import time
from unittest.mock import patch

import httpx
from openai import OpenAI

from app.core.config import settings
from app.services import openai_client
from tests.conversation_regression.replay import MODEL_MODULES, ScriptedModel, prohibit_external_effect


class EvaluationLimit(AssertionError):
    pass


class LiveModel:
    def __init__(self, api_key, model, max_calls=60, max_tokens=100000, output_limit=4096):
        if not api_key or not model or min(max_calls, max_tokens, output_limit) < 1:
            raise ValueError("Explicit evaluation credential, model and positive limits required")
        self.model = model
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls, self.tokens = [], 0
        self.stop_reason = None
        self.original = openai_client.call_openai_json_result
        # No ambient API base URL, proxy, hotel endpoint or production telemetry.
        transport = httpx.HTTPTransport(retries=0)
        def destination_guard(request):
            if request.url.scheme != "https" or request.url.host != "api.openai.com":
                raise EvaluationLimit("Evaluation permits only the official OpenAI API")
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                             max_retries=0, timeout=30,
                             http_client=httpx.Client(transport=transport, trust_env=False,
                                                       event_hooks={"request": [destination_guard]}))
        self.output_limit = output_limit

    def __call__(self, prompt, *, purpose=None, **kwargs):
        if self.stop_reason or len(self.calls) >= self.max_calls or self.tokens >= self.max_tokens:
            self.stop_reason = "Evaluation call/token budget exhausted"
            raise EvaluationLimit(self.stop_reason)
        entry = {"purpose": purpose, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "status": "STARTED"}
        self.calls.append(entry)
        started = time.perf_counter()
        try:
            result = self.original(prompt, purpose=purpose, **kwargs)
            entry.update(status="SUCCEEDED", usage=result.usage.as_api_dict(),
                         response_id=result.response_id, output=result.payload)
            self.tokens += result.usage.total_tokens
            return result
        except Exception as error:
            # Store type, never SDK exceptions that can include sensitive request headers.
            entry.update(status="ERROR", error_type=type(error).__name__)
            cause = error.__cause__
            if cause:
                entry["cause_type"] = type(cause).__name__
                entry["http_status"] = getattr(cause, "status_code", None)
            raise
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000)

    @contextmanager
    def activate(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "openai_model", self.model))
            # Recompute parameters for the selected candidate, not the ambient model.
            generation = {**openai_client._generation_parameters(), "max_output_tokens": self.output_limit}
            stack.enter_context(patch.object(openai_client, "_generation_parameters", return_value=generation))
            stack.enter_context(patch.object(openai_client, "get_openai_client", return_value=self.client))
            stack.enter_context(patch.object(openai_client, "record_model_call", return_value=None))
            stack.enter_context(patch.object(settings, "chatbotinn_api_base_url", None))
            stack.enter_context(patch.object(settings, "chatbotinn_api_internal_token", None))
            for name in MODEL_MODULES:
                module = importlib.import_module(name)
                if hasattr(module, "call_openai_json_result"):
                    stack.enter_context(patch.object(module, "call_openai_json_result", side_effect=self))
            yield self

    def configuration(self):
        with patch.object(settings, "openai_model", self.model):
            return {"requested_model": self.model,
                    "generation": {**openai_client._generation_parameters(),
                                   "max_output_tokens": self.output_limit},
                    "api_base": "https://api.openai.com/v1", "sdk_retries": 0,
                    "response_localization": "enabled",
                    "translation_cache": "shared within run; model interpretation still reevaluated each turn"}

    def close(self):
        self.client.close()


@contextmanager
def scripted_runtime(replies, *, localize=False):
    import socket
    model = ScriptedModel(replies)
    with ExitStack() as stack:
        for method in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, method, prohibit_external_effect))
        stack.enter_context(patch.object(openai_client, "get_openai_client", side_effect=prohibit_external_effect))
        for name in MODEL_MODULES:
            module = importlib.import_module(name)
            if hasattr(module, "call_openai_json_result"):
                stack.enter_context(patch.object(module, "call_openai_json_result", side_effect=model))
        if not localize:
            stack.enter_context(patch("app.agents.v2_turn_planner.localize_response",
                                      side_effect=lambda request, response, started_at: response))
        yield model
        model.assert_consumed()
