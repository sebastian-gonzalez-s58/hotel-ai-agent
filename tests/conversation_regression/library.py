"""Strict fixture loading and contract links for the sandbox regression library."""
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.v2_turns import AgentTurnRequest, DomainToolCall

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "fixtures"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Check(Strict):
    path: str = Field(pattern=r"^/")
    op: Literal["equals", "contains", "not_contains", "exists", "absent"]
    value: Any = None


class ModelReply(Strict):
    purpose: str
    payload: dict[str, Any] | None = None
    error: Literal["TIMEOUT"] | None = None

    @model_validator(mode="after")
    def one_result(self):
        if (self.payload is None) == (self.error is None):
            raise ValueError("Provide exactly one recorded payload or error")
        return self


class Event(Strict):
    kind: Literal["guest", "backend", "tool_result", "proposed_plan"]
    text: str | None = None
    reply_id: str | None = None
    # Select an actual emitted option by action suffix. -1 means latest matching
    # confirmation; positive values identify a prior response (one based).
    reply_from_turn: int | None = None
    action: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_event(self):
        if self.kind == "guest":
            if not self.text or self.action is not None or self.data:
                raise ValueError("Guest events contain only text and optional reply_id")
            if self.reply_from_turn is not None and (not self.reply_id or self.reply_from_turn == 0 or self.reply_from_turn < -1):
                raise ValueError("Button references require reply_id and a positive turn or -1")
        elif not self.action or self.text is not None or self.reply_id is not None or self.reply_from_turn is not None:
            raise ValueError("Environment events require an action and optional data")
        return self


class Step(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    event: Event
    checks: list[Check] = Field(min_length=1)
    review: list[str] = Field(default_factory=list)
    # Synthetic scripts, not recordings. None means unavailable; [] forbids calls.
    model_replies: list[ModelReply] | None = None


class Case(Strict):
    schema_version: Literal[1]
    id: str = Field(pattern=r"^SC-\d{3}-[a-z0-9-]+$")
    scenario_id: str = Field(pattern=r"^SC-\d{3}$")
    title: str
    origin: Literal["anonymized_incident", "successful_path", "boundary"]
    contract_version: Literal["1.0.0"]
    profile: str
    initial_summary: dict[str, Any] = Field(default_factory=dict)
    initial_operations: list[dict[str, Any]] = Field(default_factory=list)
    locale: str = "es-MX"
    language_source: Literal["INHERITED", "DETECTED", "EXPLICIT"] = "INHERITED"
    required_levels: list[Literal["offline", "live_model", "integration"]] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_steps(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate step IDs")
        if not self.id.startswith(self.scenario_id + "-"):
            raise ValueError("Case ID must belong to its contract scenario")
        return self

    @property
    def offline_ready(self):
        return self.integration_ready and all(s.model_replies is not None and (
            s.event.kind == 'backend' or self.model_copy(update={'steps': [s]}).agent_adapter_ready
        ) for s in self.steps)

    @property
    def integration_ready(self):
        return all(self.model_copy(update={"steps": [s]}).agent_adapter_ready or
                   s.event.kind == "backend" and s.event.action in {
                       "redeliver_same_event_and_tool_ids", "replay_tool_with_changed_arguments",
                       "lose_result_after_service_commit", "retry_interrupted_turn",
                       "execute_proposed_tools_then_fail_localization", "retry_localization_successfully",
                       "change_route_same_guest_and_stay", "reassign_contact_to_new_guest",
                       "reject_reassigned_inbound", "start_new_stay_same_contact",
                       "set_offering_requirements", "activate_synthetic_catalog_offering",
                       "localize_overlong_button", "persist_two_inbounds_before_processing",
                       "process_first_inbound", "process_second_inbound", "advance_room_order",
                       "close_maintenance", "inspect_maintenance_recurrence"}
                   or s.event.kind == 'proposed_plan' and s.event.action == 'submit_invalid_plan'
                   or s.event.kind == 'tool_result' and s.event.action == 'provide_approved_faq'
                   for s in self.steps)

    @property
    def agent_adapter_ready(self):
        return all(s.event.kind == "guest" or s.event.kind == "tool_result" and (
            s.event.action == "reply_to_last_tool" and s.event.data == {
                "status": "SUCCEEDED", "result": {"status": "NO_MATCH", "matches": []}}
            or s.event.action in {"complete_faq_handoff", "complete_service_start"} and not s.event.data
            or s.event.action == "fail_service_start" and not s.event.data
        ) for s in self.steps)


def profiles():
    return json.loads((DATA / "profiles.json").read_text(encoding="utf-8"))


def load_cases():
    cases = [Case.model_validate_json(p.read_text(encoding="utf-8"))
             for p in sorted((DATA / "cases").glob("*.json"))]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Empty library or duplicate case IDs")
    contract = json.loads((ROOT / "contracts/behavior/conversation-behavior.v1.json").read_text(encoding="utf-8"))
    scenarios = {s["id"]: s for s in contract["scenarios"]}
    if {c.scenario_id for c in cases} != set(scenarios):
        raise ValueError("Library must cover every contract scenario and no unknown IDs")
    available_profiles = profiles()
    for scenario_id, scenario in scenarios.items():
        expected_paths = {f"tests/conversation_regression/fixtures/cases/{c.id}.json"
                          for c in cases if c.scenario_id == scenario_id}
        if set(scenario["fixturePaths"]) != expected_paths:
            raise ValueError(f"Contract fixture links are out of date: {scenario_id}")
    for case in cases:
        if case.profile not in available_profiles:
            raise ValueError(f"Unknown profile: {case.profile}")
        if set(case.required_levels) != set(scenarios[case.scenario_id]["requiredLevels"]):
            raise ValueError(f"Required verification levels changed: {case.id}")
        request = AgentTurnRequest.model_validate(available_profiles[case.profile])
        if request.hotel.hotelCode != "SYNTHETIC_REGRESSION" or request.guest.displayName != "Test Guest":
            raise ValueError("Only synthetic profiles are allowed")
        # Validate operation schema as well, without creating any remote operation.
        payload = request.model_dump(mode="json")
        payload["activeOperations"] = case.initial_operations
        AgentTurnRequest.model_validate(payload)
        for step in case.steps:
            if step.event.kind == "proposed_plan":
                # Invalid business plans must still have a well-formed tool envelope.
                DomainToolCall.model_validate(step.event.data["tool_call"])
    return cases
