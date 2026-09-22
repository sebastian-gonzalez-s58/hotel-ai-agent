import json
import hashlib
from copy import deepcopy
import logging
import re
import time
import unicodedata
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AgentModelError
from app.agents.v2_scope_router import ScopeDecision, classify_hotel_scope
from app.agents.maintenance_recurrence import plan_maintenance_recurrence
from app.agents.reservation_actions import plan_reservation_action
from app.agents.room_service_status import existing_order_message, resolve_order, status_message
from app.agents.social_opening import classify_social_opening
from app.agents.schema_validation import satisfies_schema
from app.agents.spa_turns import plan_spa_turn, preserve_spa_state, started_spa_summary, validate_spa_call
from app.prompts.v2_turn import build_v2_turn_prompt
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from app.schemas.v2_turns import DomainToolName
from app.services.openai_client import call_openai_json_result
from app.services.conversation_language import language_enabled, greeting_language, resolve_language
from app.services.localized_content import localize_response, template
from app.services.catalog_selection import pending_catalog_selection, ordered_capture_fields as _ordered_guest_capture_fields
from app.services.catalog_orders import (catalog_for, normalize_order, pending_choice, apply_choice,
                                         clarification as catalog_clarification)
from app.services.faq_grounding import is_semantic_search, resolve_semantic_faq, restore_grounded_faq
from app.services.input_understanding import (
    understanding_turn, understanding_enabled, record_scope_action, semantic_action, semantic_action_offering, understand_order,
    order_understanding_issue, order_understanding_failed,
)


logger = logging.getLogger("chatbotinn-agent.v2-turn-planner")
AGENT_TURN_RESPONSE_SCHEMA = AgentTurnResponse.model_json_schema()
AGENT_TURN_RESPONSE_SCHEMA["properties"].pop("languageDecision", None)
AGENT_TURN_RESPONSE_SCHEMA["properties"].pop("roomServiceDraftEvent", None)
MAX_PLAN_ATTEMPTS = 3


def plan_v2_turn(request: AgentTurnRequest) -> AgentTurnResponse:
    with understanding_turn(request) as understanding:
        response = _plan_v2_turn(request)
        for name, value in understanding.usage.items():
            setattr(response.usage, name, getattr(response.usage, name) + value)
        if understanding.last_order_issue:
            response.warnings = (response.warnings + ["ORDER_INPUT_" + understanding.last_order_issue])[-20:]
        response.usage.latencyMs = round((time.perf_counter() - understanding.started_at) * 1000)
        return response


def _plan_v2_turn(request: AgentTurnRequest) -> AgentTurnResponse:
    started_at = time.perf_counter()
    # Bind every helper (including deterministic captures) to the triggering inbound message.
    request = _request_for_trigger(request)
    latest = _latest_inbound_message(request)
    original_request = request
    language_decision = None
    if language_enabled(request):
        request, language_decision = resolve_language(request, latest)
    if (request.trigger.type == "INBOUND_MESSAGE" and not request.previousToolResults
            and latest is not None and _is_room_confirmation_button(latest)
            and (not _room_confirmation_matches(request, latest)
                 or request.trigger.eventPayload.get('confirmationQueuedBeforePrompt') is True)):
        response = _stale_room_confirmation(request, started_at)
        response.languageDecision = language_decision
        if language_enabled(request):
            response.detectedLanguage = language_decision.locale if language_decision else None
            response = localize_response(request, response, started_at)
        return _bind_room_confirmation(request, response)
    reservation = plan_reservation_action(request, latest)
    if reservation is not None:
        response = _deterministic_turn_response(request, started_at, **reservation)
        response.languageDecision = language_decision
        if language_enabled(request):
            response.detectedLanguage = language_decision.locale if language_decision else None
            response = localize_response(request, response, started_at)
        return response
    maintenance = plan_maintenance_recurrence(request, _latest_capture_state(request), latest)
    if maintenance is not None:
        response = _deterministic_turn_response(request, started_at, **maintenance)
        response.languageDecision = language_decision
        if language_enabled(request):
            response.detectedLanguage = language_decision.locale if language_decision else None
            response = localize_response(request, response, started_at)
        return response
    pending_selection = pending_catalog_selection(request, _latest_capture_state(request))
    if pending_selection and latest is not None and not latest.interactionReplyId:
        code = pending_selection.exact_code(latest.text)
        if code is not None:
            request = _with_catalog_selection(request, pending_selection, code)
            latest = _latest_inbound_message(request)
    scope = None
    scope_usage = None
    opening_usage = None
    if (request.trigger.type == "INBOUND_MESSAGE" and not request.previousToolResults
            and latest is not None and not latest.interactionReplyId
            and not _pending_catalog_option_answer(request, latest)
            and not (not understanding_enabled() and _has_pending_maintenance_resolution_task(request)
                     and _maintenance_resolution_value(latest) is not None)
            and not _is_greeting_turn(request) and _capture_selection(request, latest) is None):
        scope, scope_usage = classify_hotel_scope(request, latest, _latest_capture_state(request))
        record_scope_action(latest, scope)
        if language_enabled(request):
            request, language_decision = resolve_language(original_request, latest, scope)
        opening = False
        if (language_enabled(request) and scope.kind == "SOCIAL" and not scope.languageChangeOnly
                and request.availableOfferings and not scope.containsUnrelatedTopic):
            opening, opening_usage = classify_social_opening(latest.text)
        maintenance = plan_maintenance_recurrence(request, _latest_capture_state(request), latest, scope)
        reservation = plan_reservation_action(request, latest, scope)
        existing_message = existing_order_message(request, latest, scope, _latest_capture_state(request))
        if reservation is not None:
            response = _deterministic_turn_response(request, started_at, **reservation)
        elif maintenance is not None:
            response = _deterministic_turn_response(request, started_at, **maintenance)
        elif existing_message is not None:
            response = _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                messages=[existing_message], updated_summary=request.conversation.summary)
        elif (scope.replyAction == 'CONFIRM' and request.trigger.eventPayload.get('confirmationQueuedBeforePrompt') is True
                and _latest_capture_state(request).get('pendingOffering') == 'ROOM_SERVICE'):
            response = _stale_room_confirmation(request, started_at)
        elif language_enabled(request) and scope.languageChangeOnly and language_decision is not None:
            language_template = ("language.changed.pending" if _latest_capture_state(request).get("pendingOffering")
                                 or request.activeOperations else "language.changed")
            response = _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY", messages=[{
                "purpose": "ANSWER", "text": template(language_template, request.guest.preferredLanguage),
                "language": request.guest.preferredLanguage, "operationIds": [], "conversationTaskIds": [],
            }], updated_summary=request.conversation.summary)
        elif scope.kind in {"OUT_OF_SCOPE", "UNCLEAR"}:
            response = _scope_clarification(request, scope.kind, started_at)
        elif opening:
            messages = []
            _ensure_personalized_service_menu(request, messages, force=True)
            response = _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                                    messages=messages, updated_summary=request.conversation.summary)
        else:
            scoped = request.model_copy(deep=True)
            _latest_inbound_message(scoped).text = scope.relevantText
            code = pending_selection.semantic_code(scope, latest.text) if pending_selection else None
            if code is not None:
                scoped = _with_catalog_selection(scoped, pending_selection, code)
                response = _plan_hotel_turn(scoped, started_at, scope)
            elif (pending_selection and scope.kind == "CONTEXT_REPLY" and scope.replyAction in {"NONE", "AMBIGUOUS"}
                  and (scope.selectionAttempted or scope.selectionCode is not None)
                  and not scope.containsUnrelatedTopic):
                field = pending_selection.field_schema
                message = _capture_message(request, pending_selection.offering.offeringCode,
                                           pending_selection.field_code, field, field["x-chatbotinn-capture"])
                summary = request.conversation.summary
                if pending_selection.current_value is not None:
                    # BC-003 / BC-006: retain the draft, but do not allow an unresolved
                    # location edit to confirm the previous delivery location.
                    summary = _room_service_summary(_latest_capture_state(request)["capturedFields"],
                                                    False, "NEEDS_LOCATION_CLARIFICATION")
                response = _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                        messages=[message], updated_summary=summary)
            else:
                response = _plan_hotel_turn(scoped, started_at, scope)
        if (scope.containsUnrelatedTopic and scope.kind not in {"OUT_OF_SCOPE", "UNCLEAR"}
                and response.messages):
            notice = _scope_refusal(request)
            response.messages[0].text = notice + "\n\n" + response.messages[0].text
            if response.messages[0].interaction is not None:
                response.messages[0].interaction.body = response.messages[0].text[:1024]
        for usage in (scope_usage, opening_usage):
            if usage is not None:
                for name, count in usage.as_api_dict().items():
                    setattr(response.usage, name, getattr(response.usage, name) + count)
        response.usage.latencyMs = round((time.perf_counter() - started_at) * 1000)
        response = preserve_spa_state(request, response, scope)
    else:
        if language_enabled(request) and _is_greeting_turn(request) and request.availableOfferings and not request.previousToolResults:
            messages = []
            _ensure_personalized_service_menu(request, messages, force=True)
            response = _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                                    messages=messages, updated_summary=request.conversation.summary)
        else:
            response = preserve_spa_state(request, _plan_hotel_turn(request, started_at, scope))
    response = _preserve_room_service_draft(original_request, response)
    response = _preserve_start_failure(original_request, response)
    # Language is runtime-owned metadata; a planner may not invent a preference change.
    response.languageDecision = language_decision
    if language_enabled(request):
        response.detectedLanguage = language_decision.locale if language_decision else None
        if not _is_verified_faq_response(request, response):
            response = localize_response(request, response, started_at)
    return _bind_room_confirmation(request, response, new_draft=bool(
        scope and scope.separateRequest and scope.offeringCode == 'ROOM_SERVICE'))


def _pending_catalog_option_answer(request, message):
    """A reply to an offered option is not a global cancellation or order approval."""
    state = _latest_capture_state(request)
    pending = [state.get('catalogPending'), (state.get('spaDraft') or {}).get('catalogPending')]
    drafts = state.get('catalogReplacementTasks', {})
    for operation in request.activeOperations:
        for task in operation.pendingConversationTasks:
            draft = drafts.get(str(task.conversationTaskId), {})
            if draft.get('version') == task.version:
                pending.append(draft.get('pending'))
    return any(p and p.get('kind') == 'OPTION' and pending_choice(p, message) for p in pending)


def _with_catalog_selection(request, selection, code):
    # Only the internal request gains a canonical selection; text and message ID remain original evidence.
    resolved = request.model_copy(deep=True)
    _latest_inbound_message(resolved).interactionReplyId = selection.reply_id(code)
    return resolved


def _is_room_confirmation_button(message):
    reply = (message.interactionReplyId or "").strip().upper()
    # Older menus have no draft capability. Treat them as stale, never as fresh model intent.
    return reply in {"CONFIRM_ORDER", "CHANGE_ORDER", "CANCEL_ORDER"} or reply.startswith(
        ("CONFIRMATION:ROOM_SERVICE:", "ROOM-SERVICE:"))


def _room_order_digest(request, captured):
    payload = [str(request.conversation.conversationId), captured]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _room_confirmation_matches(request, latest):
    if latest is None:
        return False
    if not latest.interactionReplyId:
        return True  # Free-text approval still goes through semantic validation.
    state = _latest_capture_state(request)
    confirmation = state.get("roomServiceConfirmation")
    if (state.get("pendingOffering") != "ROOM_SERVICE" or not state.get("awaitingExplicitConfirmation")
            or state.get("phase") == "STARTING" or not isinstance(confirmation, dict)):
        return False
    token = confirmation.get("token")
    return (isinstance(token, str) and bool(token)
            and confirmation.get("orderDigest") == _room_order_digest(request, state.get("capturedFields"))
            and latest.interactionReplyId in {
                f"confirmation:ROOM_SERVICE:{token}:{action}" for action in ("CONFIRM", "CHANGE", "CANCEL")})


def _bind_room_confirmation(request, response, new_draft=False):
    """Runtime-owned capability: each displayed menu authorizes exactly its draft.

    A fresh token also prevents an A -> B -> A edit from reviving A's old menu.
    The model cannot preserve/invent a token for a different captured order.
    """
    # Persist only runtime-verified transitions, never planner-invented history.
    response.roomServiceDraftEvent = None
    summary = response.updatedConversationSummary
    if not summary:
        return response
    probe = request.model_copy(deep=True)
    probe.conversation.summary = summary
    state = _latest_capture_state(probe)
    previous_state = _latest_capture_state(request)
    latest = _latest_inbound_message(request)
    was_room = previous_state.get('pendingOffering') == 'ROOM_SERVICE'
    is_room = state.get('pendingOffering') == 'ROOM_SERVICE'
    if is_room and (not was_room or new_draft):
        response.roomServiceDraftEvent = 'NEW'
    elif (was_room and not is_room and not response.toolCalls and latest is not None
          and request.trigger.type == 'INBOUND_MESSAGE' and not request.previousToolResults
          and (not latest.interactionReplyId or _room_confirmation_matches(request, latest))
          and (_room_service_confirmation_action(latest) == 'CANCEL' or _is_free_text_cancel(latest.text))):
        response.roomServiceDraftEvent = 'CANCELLED'
    messages = [m for m in response.messages if m.purpose == "CONFIRMATION" and m.interaction
                and any(o.id.startswith("confirmation:ROOM_SERVICE:") for o in m.interaction.options)]
    if (state.get("pendingOffering") == "ROOM_SERVICE" and state.get("awaitingExplicitConfirmation")
            and state.get("phase") != "STARTING"):
        if messages:
            token = uuid4().hex
            state["roomServiceConfirmation"] = {
                "token": token, "orderDigest": _room_order_digest(request, state.get("capturedFields"))}
            for message in messages:
                for option in message.interaction.options:
                    if option.id.startswith("confirmation:ROOM_SERVICE:"):
                        option.id = f"confirmation:ROOM_SERVICE:{token}:{option.id.rsplit(':', 1)[-1]}"
        else:
            # Only carry an unchanged capability from the incoming runtime state.
            previous = _latest_capture_state(request).get("roomServiceConfirmation")
            state.pop("roomServiceConfirmation", None)
            if (isinstance(previous, dict)
                    and previous.get("orderDigest") == _room_order_digest(request, state.get("capturedFields"))):
                state["roomServiceConfirmation"] = previous
    else:
        state.pop("roomServiceConfirmation", None)
    # Preserve prose summaries; replace only their final structured snapshot.
    if messages or state != _latest_capture_state(probe):
        encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        lines = summary.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            try:
                if isinstance(json.loads(lines[index]), dict):
                    lines[index] = encoded
                    break
            except ValueError:
                continue
        else:
            lines = [encoded]
        response.updatedConversationSummary = "\n".join(lines)
    return response


def _stale_room_confirmation(request, started_at):
    state = _latest_capture_state(request)
    latest = _latest_inbound_message(request)
    operation = resolve_order(request, latest, button=True) if latest else None
    action = (latest.interactionReplyId or '').upper().rsplit(':', 1)[-1].removesuffix('_ORDER') if latest else 'CONFIRM'
    history = request.trigger.eventPayload.get('roomServiceButtonContext', {})
    history_status = history.get('menuStatus') if isinstance(history, dict) else None
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    notice = ("Ese botón corresponde a una confirmación anterior. Revisa el resumen actualizado antes de confirmar."
              if spanish else "That button belongs to an earlier confirmation. Review the updated summary before confirming.")
    offering = next((o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE"), None)
    if history_status == 'CANCELLED_DRAFT':
        message = {'purpose': 'CLARIFICATION', 'text': (
            'Cancelaste este borrador antes de enviarlo a cocina. Esa confirmación ya no puede utilizarse; no envié ningún pedido.'
            if spanish else 'You cancelled this draft before sending it to the kitchen. That confirmation can no longer be used; I did not submit an order.'),
            'language': request.guest.preferredLanguage, 'operationIds': [], 'conversationTaskIds': []}
    elif operation is not None:
        message = status_message(request, operation, action, button=True)
        if history_status == 'REPLACED':
            message['text'] = ('Ese botón corresponde a una versión anterior de tu pedido y no puede confirmarla. ' if spanish else
                               'That button belongs to an earlier version of your order and cannot confirm it. ') + message['text']
    elif (offering and state.get("pendingOffering") == "ROOM_SERVICE"
            and state.get("awaitingExplicitConfirmation") and state.get("phase") != "STARTING"):
        message = _room_service_confirmation_message(request, offering, state.get("capturedFields", {}))
        message["text"] = notice + "\n\n" + message["text"]
        if message.get("interaction"):
            if len(message["text"]) > 1024:
                message["interaction"] = None
            else:
                message["interaction"]["body"] = message["text"]
    elif offering and state.get("pendingOffering") == "ROOM_SERVICE" and state.get("phase") == "NEEDS_LOCATION_CLARIFICATION":
        field = offering.inputSchema.get("properties", {}).get("deliveryLocation", {})
        message = _capture_message(request, offering.offeringCode, "deliveryLocation",
                                   field, field.get("x-chatbotinn-capture", {}))
        message = message or _order_clarification_message(request)
    elif history_status == 'REPLACED':
        message = {'purpose': 'CLARIFICATION', 'text': (
            'Modificaste este borrador y esa confirmación corresponde a una versión anterior. No envié ni modifiqué ningún pedido.'
            if spanish else 'You changed this draft and that confirmation belongs to an earlier version. I did not submit or modify an order.'),
            'language': request.guest.preferredLanguage, 'operationIds': [], 'conversationTaskIds': []}
    else:
        message = status_message(request, None, action, button=True)
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                       messages=[message], updated_summary=request.conversation.summary)


def _service_start_failure(request, started_at, scope=None):
    state = _latest_capture_state(request)
    failed = [r for r in request.previousToolResults if r.toolName == "START_SERVICE" and r.status == "FAILED"]
    unresolved = state.get("serviceStartFailures", {})
    if failed:
        unresolved = dict(unresolved)
        for receipt in failed:
            code = receipt.error.details.get("offeringCode") if receipt.error else None
            unresolved[str(receipt.toolCallId)] = {"offeringCode": code or (
                state.get("pendingOffering", "UNKNOWN") if len(failed) == 1 else "UNKNOWN")}
        state["serviceStartFailures"] = unresolved
        state.update(phase="START_UNCERTAIN", awaitingExplicitConfirmation=False)
        state.pop("roomServiceConfirmation", None)
    elif unresolved and request.trigger.type == "INBOUND_MESSAGE" and not request.previousToolResults:
        latest = _latest_inbound_message(request)
        selected = (latest.interactionReplyId or "") if latest else ""
        codes = {value.get("offeringCode", "UNKNOWN") for value in unresolved.values()}
        # Allow independent services; retain the unresolved receipt across their turns.
        if "UNKNOWN" not in codes and selected.startswith("offering:") and selected.split(":", 1)[1] not in codes:
            return None
        if "UNKNOWN" not in codes and scope and scope.offeringCode and scope.offeringCode not in codes:
            return None
        if ("UNKNOWN" not in codes and state.get("pendingOffering") not in codes
                and not (scope and scope.offeringCode in codes)
                and selected not in {f"offering:{code}" for code in codes}):
            return None
    else:
        return None
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    text = ("No pude confirmar el inicio de tu solicitud. Conservé los datos. "
            "Es necesario revisar el resultado anterior antes de volver a iniciarla." if spanish else
            "I could not confirm that your request started. Your details have been saved. "
            "The previous outcome needs to be checked before starting it again.")
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
        messages=[{"purpose": "CLARIFICATION", "text": text, "language": request.guest.preferredLanguage,
                   "operationIds": [], "conversationTaskIds": []}],
        updated_summary=json.dumps(state, ensure_ascii=False, separators=(",", ":")))


def _preserve_start_failure(request, response):
    unresolved = _latest_capture_state(request).get("serviceStartFailures")
    if not unresolved:
        return response
    # Runtime-owned receipt; model summaries and switching services cannot erase it.
    probe = request.model_copy(deep=True)
    probe.conversation.summary = response.updatedConversationSummary or request.conversation.summary
    state = _latest_capture_state(probe)
    state["serviceStartFailures"] = {**state.get("serviceStartFailures", {}), **unresolved}
    response.updatedConversationSummary = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return response


def _preserve_room_service_draft(request, response):
    """BC-004: another service's summary cannot discard an independent order."""
    old = _latest_capture_state(request)
    if response.updatedConversationSummary is None:
        return response
    updated = request.model_copy(deep=True)
    updated.conversation.summary = response.updatedConversationSummary
    new = _latest_capture_state(updated)
    if new.get("pendingOffering") == "ROOM_SERVICE":
        # There is a single active draft: never leave a second, stale resumable copy.
        if "roomServiceDraft" in new:
            new.pop("roomServiceDraft")
            response.updatedConversationSummary = json.dumps(new, ensure_ascii=False)
        return response
    draft = old.get("roomServiceDraft")
    if old.get("pendingOffering") == "ROOM_SERVICE":
        latest = _latest_inbound_message(request)
        cancelled = (not request.previousToolResults and latest is not None
                     and (_room_service_confirmation_action(latest) == "CANCEL" or _is_free_text_cancel(latest.text)))
        submitted = any(c.toolName == DomainToolName.START_SERVICE and c.arguments.get("offeringCode") == "ROOM_SERVICE"
                        for c in response.toolCalls)
        started = any(r.toolName == "START_SERVICE" and r.status == "SUCCEEDED" and isinstance(r.result, dict)
                      and r.result.get("offeringCode") == "ROOM_SERVICE" for r in request.previousToolResults)
        if cancelled or submitted or started:
            return response
        draft = {key: deepcopy(old[key]) for key in ("pendingOffering", "capturedFields", "phase",
                "awaitingExplicitConfirmation", "readyToStart") if key in old}
        # Resuming must present the current summary before it can be confirmed again.
        draft["awaitingExplicitConfirmation"] = False
        draft["readyToStart"] = False
    if isinstance(draft, dict) and draft.get("pendingOffering") == "ROOM_SERVICE":
        new["roomServiceDraft"] = deepcopy(draft)
        response.updatedConversationSummary = json.dumps(new, ensure_ascii=False)
    return response


def _resume_room_service_draft(request, scope, started_at):
    if request.previousToolResults or request.trigger.type != "INBOUND_MESSAGE":
        return None
    state = _latest_capture_state(request)
    draft = state.get("roomServiceDraft")
    if not isinstance(draft, dict) or draft.get("pendingOffering") != "ROOM_SERVICE":
        return None
    latest = _latest_inbound_message(request)
    if latest is None:
        return None
    explicit = latest.interactionReplyId == "offering:ROOM_SERVICE"
    scoped = scope is not None and scope.offeringCode == "ROOM_SERVICE" and not scope.separateRequest
    sole_context = (scope is not None and scope.kind == "CONTEXT_REPLY" and not scope.separateRequest
                    and scope.offeringCode is None and not state.get("pendingOffering") and not state.get("spaDraft")
                    and not any(o.pendingConversationTasks for o in request.activeOperations))
    if not (explicit or scoped and scope.kind in {"SERVICE_REQUEST", "CONTEXT_REPLY"} or sole_context):
        return None
    # A decision from an earlier confirmation cannot submit a suspended draft.
    if scope is not None and scope.replyAction in {"CONFIRM", "CANCEL"}:
        return None
    offering = next((o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE"), None)
    if offering is None:
        return None
    captured = deepcopy(draft.get("capturedFields", {}))
    resumed = request.model_copy(deep=True)
    resumed.conversation.summary = _room_service_summary(captured, False, draft.get("phase", "CAPTURING_ITEMS"))
    if scope is not None and scope.hasRequestDetails:
        return _room_service_draft_plan(resumed, started_at)
    if not captured.get("items"):
        field_code = "items" if captured.get("deliveryLocation") else "deliveryLocation"
        field = offering.inputSchema.get("properties", {}).get(field_code, {})
        message = _capture_message(resumed, offering.offeringCode, field_code, field,
                                   field.get("x-chatbotinn-capture", {}))
        return _deterministic_turn_response(resumed, started_at, disposition="RESPONSE_READY",
            messages=[message or _order_clarification_message(resumed)],
            updated_summary=_room_service_summary(captured, False,
                "CAPTURING_ITEMS" if field_code == "items" else "CAPTURING_LOCATION"))
    return _deterministic_turn_response(resumed, started_at, disposition="RESPONSE_READY",
        **_room_service_capture_output(resumed, offering, captured, draft.get("phase") == "NEEDS_LOCATION_CLARIFICATION"))


def _plan_hotel_turn(request: AgentTurnRequest, started_at: float,
                     scope: ScopeDecision | None = None) -> AgentTurnResponse:
    maintenance_resolution = _maintenance_resolution_task_plan(request, started_at, scope)
    if maintenance_resolution is not None:
        return maintenance_resolution
    failure = _service_start_failure(request, started_at, scope)
    if failure is not None:
        return failure
    started = _acknowledgeable_service_starts(request)
    if started:
        messages = []
        _ensure_service_start_acknowledgements(request, messages)
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                            messages=messages,
                                            updated_summary=_started_service_summary(request, started))
    latest = _latest_inbound_message(request)
    resumed_order = _resume_room_service_draft(request, scope, started_at)
    if resumed_order is not None:
        return resumed_order
    greeting = (request.trigger.type == "INBOUND_MESSAGE" and not request.previousToolResults
                and latest is not None and not latest.interactionReplyId and _is_greeting_turn(request))
    front_desk_plan = _front_desk_start_plan(request, started_at, scope)
    if front_desk_plan is not None:
        return front_desk_plan
    spa_plan = None if greeting else plan_spa_turn(request, scope)
    if spa_plan is not None:
        return _deterministic_turn_response(request, started_at, **spa_plan)
    if scope is not None and scope.kind == "NAVIGATION":
        if not request.availableOfferings:
            return _scope_clarification(request, "UNCLEAR", started_at)
        messages = []
        _ensure_personalized_service_menu(request, messages, force=True)
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                            messages=messages, updated_summary=request.conversation.summary)
    if scope is not None and scope.kind == "HOTEL_QUESTION":
        deterministic = _faq_knowledge_lookup_plan(request, started_at, direct_question=True)
        if deterministic is not None:
            return deterministic
        return _scope_clarification(request, "UNCLEAR", started_at)
    if scope is not None and scope.kind == "SERVICE_REQUEST":
        offering = next(o for o in request.availableOfferings
                        if o.offeringCode == scope.offeringCode)
        if understanding_enabled() and offering.offeringCode == "ROOM_SERVICE" and scope.hasRequestDetails:
            initial = _initial_offering_capture_plan(request, offering, started_at)
            if initial is not None:
                items = understand_order(request, latest, [], allow_partial=True)
                if order_understanding_failed():
                    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                            messages=[_order_clarification_message(request)],
                            updated_summary=request.conversation.summary)
                if items and any(item.get("quantity") is None for item in items):
                    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                            **_room_service_capture_output(request, offering, {"items": items}))
                initial.updatedConversationSummary = _room_service_summary(
                    {"items": items} if items else {}, False, "CAPTURING_LOCATION")
                return initial
        deterministic = _single_free_text_service_start_plan(
            request,
            offering,
            scope,
            started_at,
        )
        if deterministic is not None:
            return deterministic
        if not scope.hasRequestDetails:
            deterministic = _initial_offering_capture_plan(request, offering, started_at)
            if deterministic is not None:
                return deterministic
        # A new explicit request must not be consumed as an answer to an older task/draft.
        if _latest_capture_state(request).get("pendingOffering") != offering.offeringCode:
            request = request.model_copy(deep=True)
            request.conversation.summary = _capture_summary(request, offering.offeringCode, {}, False)
    skip_capture = (not request.previousToolResults and _is_greeting_turn(request)) or (
        scope is not None and scope.kind in {
        "SERVICE_REQUEST", "STATUS_REQUEST", "NAVIGATION", "SOCIAL"
    })
    deterministic = None if skip_capture else _room_service_operation_task_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    deterministic = None if skip_capture else _room_service_draft_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    deterministic = None if skip_capture else _faq_knowledge_lookup_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    deterministic = _faq_service_start_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    deterministic = _faq_started_response_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    deterministic = None if skip_capture else _configured_capture_plan(request, started_at)
    if deterministic is not None:
        return deterministic
    prompt = build_v2_turn_prompt(request)
    if not skip_capture:
        prompt += _build_capture_turn_instruction(request)
    if scope is not None:
        prompt += ("\nThe hotel scope router classified this current message as "
                   + scope.model_dump_json() + ". Respect that route. A specific request is not "
                   "a greeting or a request for the main menu. Ask only for missing service data. "
                   "Do not turn a status question or a thank-you into a capture field.")
    accumulated_usage = {name: 0 for name in (
        "inputTokens", "cachedInputTokens", "outputTokens", "reasoningTokens", "totalTokens"
    )}
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        result = call_openai_json_result(
            prompt,
            purpose="V2_AGENT_TURN",
            response_schema=AGENT_TURN_RESPONSE_SCHEMA,
            response_schema_name="agent_turn_response_v2",
        )
        payload = _normalize_response_envelope(request, result.payload)
        payload["languageDecision"] = None
        payload = _normalize_guest_experience(request, payload, skip_capture=skip_capture)
        for name, count in result.usage.as_api_dict().items():
            accumulated_usage[name] += count
        payload["usage"] = {
            "model": settings.openai_model,
            **accumulated_usage,
            "latencyMs": round((time.perf_counter() - started_at) * 1000),
        }
        try:
            response = AgentTurnResponse.model_validate(payload)
            _validate_plan(request, response, enforce_faq_rewrite=attempt == 1)
            if scope is not None and scope.kind == "SERVICE_REQUEST" and any(
                m.interaction and any(o.id.startswith("offering:") for o in m.interaction.options)
                for m in response.messages
            ):
                raise AgentModelError("A specific service request must not return the main service menu")
            return response
        except ValidationError as exc:
            error = AgentModelError(
                f"OpenAI returned an invalid V2 agent turn schema response_id={result.response_id}"
            )
            details = exc.errors(include_url=False, include_input=False)
        except AgentModelError as exc:
            error = exc
            details = str(exc)

        logger.warning(
            "Invalid V2 agent plan; retrying when possible agent_turn_id=%s response_id=%s "
            "attempt=%s/%s error=%s",
            request.agentTurnId,
            result.response_id,
            attempt,
            MAX_PLAN_ATTEMPTS,
            details,
        )
        if attempt == MAX_PLAN_ATTEMPTS:
            raise error
        prompt += (
            "\n\nThe previous plan was invalid and must not be repeated. "
            f"Validation error: {details}. Return a corrected plan."
            + _build_focused_task_repair_instruction(request)
        )

    raise AgentModelError("OpenAI did not return a valid V2 agent plan")


def _request_for_trigger(request: AgentTurnRequest) -> AgentTurnRequest:
    message_id = request.trigger.messageId
    if message_id is None:
        return request
    index = next((i for i, m in enumerate(request.conversation.recentMessages)
                  if m.messageId == message_id and m.direction == "INBOUND"), None)
    if index is None:
        if request.trigger.type == "INBOUND_MESSAGE":
            raise AgentModelError("Trigger message is missing from the turn context")
        return request
    scoped = request.model_copy(deep=True)
    scoped.conversation.recentMessages = [
        m for i, m in enumerate(scoped.conversation.recentMessages)
        if i <= index or m.direction != "INBOUND"
    ]
    return scoped


def _scope_refusal(request: AgentTurnRequest) -> str:
    if language_enabled(request):
        return template("scope.refusal", request.guest.preferredLanguage)
    return ("Lo siento, solo puedo ayudarte con los servicios del hotel y tu estancia."
            if request.guest.preferredLanguage.lower().startswith("es") else
            "Sorry, I can only help with hotel services and your stay.")


def _scope_clarification(request: AgentTurnRequest, kind: str,
                         started_at: float) -> AgentTurnResponse:
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    text = (_scope_refusal(request) if kind == "OUT_OF_SCOPE" else
            "¿En qué servicio del hotel necesitas ayuda?" if spanish else
            "Which hotel service do you need help with?")
    if kind == "OUT_OF_SCOPE":
        text += (" ¿Necesitas ayuda con algún servicio?" if spanish else
                 " Do you need help with a hotel service?")
    if language_enabled(request):
        locale = request.guest.preferredLanguage
        text = (template("scope.refusal", locale) + " " + template("scope.invitation", locale)
                if kind == "OUT_OF_SCOPE" else template("scope.clarify", locale))
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY", messages=[{
        "purpose": "CLARIFICATION", "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [], "conversationTaskIds": [], "interaction": None,
    }], updated_summary=request.conversation.summary)


def _front_desk_start_plan(request: AgentTurnRequest, started_at: float,
                           scope: ScopeDecision | None) -> AgentTurnResponse | None:
    if request.trigger.type != "INBOUND_MESSAGE" or request.previousToolResults:
        return None
    latest = _latest_inbound_message(request)
    if latest is None:
        return None
    selection = _capture_selection(request, latest)
    selected = selection is not None and selection[0].offeringCode == "FRONT_DESK" and selection[1] is None
    requested = scope is not None and scope.kind == "SERVICE_REQUEST" and scope.offeringCode == "FRONT_DESK"
    if not selected and not requested:
        return None
    offering = next((o for o in request.availableOfferings if o.offeringCode == "FRONT_DESK"), None)
    if (offering is None or offering.executionMode != "PROCESS"
            or offering.requiresExplicitGuestConfirmation
            or not satisfies_schema({}, offering.inputSchema)
            or DomainToolName.START_SERVICE not in request.toolPolicy.allowedTools
            or request.toolPolicy.maxToolCalls < 1):
        return None
    # The current selection is the evidence. Other service drafts/tasks stay intact.
    return _deterministic_turn_response(
        request, started_at, disposition="TOOL_CALLS_REQUIRED", messages=[],
        tool_calls=[{
            "toolCallId": str(uuid4()), "toolName": DomainToolName.START_SERVICE.value,
            "targetOperationId": None, "targetConversationTaskId": None,
            "arguments": {"offeringCode": "FRONT_DESK", "input": {}},
            "confidence": 1.0, "evidenceMessageIds": [str(latest.messageId)],
        }], updated_summary=request.conversation.summary,
    )


def _initial_offering_capture_plan(request: AgentTurnRequest, offering,
                                  started_at: float) -> AgentTurnResponse | None:
    fields = _ordered_guest_capture_fields(offering)
    if not fields:
        return None
    field_code, field_schema = fields[0]
    capture = field_schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return None
    message = _capture_message(request, offering.offeringCode, field_code, field_schema, capture)
    if message is None:
        return None
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                                       messages=[message], updated_summary=_capture_summary(
                                           request, offering.offeringCode, {}, False))


def _single_free_text_service_start_plan(
    request: AgentTurnRequest,
    offering,
    scope: ScopeDecision,
    started_at: float,
) -> AgentTurnResponse | None:
    """Start a one-field service without asking the model to rebuild known input."""
    if (
        not scope.hasRequestDetails
        or request.previousToolResults
        or offering.executionMode != "PROCESS"
        or offering.requiresExplicitGuestConfirmation
        or DomainToolName.START_SERVICE not in request.toolPolicy.allowedTools
        or request.toolPolicy.maxToolCalls < 1
    ):
        return None

    fields = _ordered_guest_capture_fields(offering)
    if len(fields) != 1:
        return None
    field_code, field_schema = fields[0]
    capture = field_schema.get("x-chatbotinn-capture")
    if (
        not isinstance(capture, dict)
        or str(capture.get("inputMode") or "AUTO").upper() != "FREE_TEXT"
    ):
        return None

    latest = _latest_inbound_message(request)
    value = " ".join(scope.relevantText.strip().split()).strip()
    service_input = {field_code: value}
    if latest is None or not value or not satisfies_schema(service_input, offering.inputSchema):
        return None

    evidence_message_id = str(latest.messageId)
    return _deterministic_turn_response(
        request,
        started_at,
        disposition="TOOL_CALLS_REQUIRED",
        messages=[],
        tool_calls=[{
            "toolCallId": str(uuid4()),
            "toolName": DomainToolName.START_SERVICE.value,
            "targetOperationId": None,
            "targetConversationTaskId": None,
            "arguments": {
                "offeringCode": offering.offeringCode,
                "input": service_input,
            },
            "confidence": scope.confidence,
            "evidenceMessageIds": [evidence_message_id],
        }],
        updated_summary=_capture_summary(
            request,
            offering.offeringCode,
            service_input,
            True,
        ),
    )


def _faq_knowledge_lookup_plan(
    request: AgentTurnRequest,
    started_at: float,
    direct_question: bool = False,
) -> AgentTurnResponse | None:
    if request.previousToolResults or DomainToolName.SEARCH_KNOWLEDGE not in request.toolPolicy.allowedTools:
        return None

    if direct_question:
        offering = next((o for o in request.availableOfferings if o.offeringCode == "FAQ"), None)
        latest = _latest_inbound_message(request)
        context = {"offering": offering, "latestInbound": latest} if offering and latest else None
    else:
        context = _pending_free_text_capture_context(request)
    if context is None or context["offering"].offeringCode != "FAQ":
        return None

    question = " ".join(context["latestInbound"].text.strip().split()).strip()
    if not question:
        return None

    payload = _normalize_response_envelope(request, {
        "disposition": "TOOL_CALLS_REQUIRED",
        "messages": [],
        "toolCalls": [{
            "toolName": DomainToolName.SEARCH_KNOWLEDGE.value,
            "targetOperationId": None,
            "targetConversationTaskId": None,
            "arguments": {
                "offeringCode": "FAQ",
                "query": question,
                **({"semantic": True} if language_enabled(request) else {}),
                "limit": 10,
            },
            "confidence": 1.0,
            "evidenceMessageIds": [str(context["latestInbound"].messageId)],
        }],
        "updatedConversationSummary": _capture_summary(
            request,
            "FAQ",
            {"question": question},
            False,
        ),
        "warnings": [],
    })
    payload["usage"] = {
        "model": settings.openai_model,
        "inputTokens": 0,
        "cachedInputTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "totalTokens": 0,
        "latencyMs": round((time.perf_counter() - started_at) * 1000),
    }
    response = AgentTurnResponse.model_validate(payload)
    _validate_plan(request, response)
    return response


def _configured_capture_plan(
    request: AgentTurnRequest,
    started_at: float,
) -> AgentTurnResponse | None:
    latest_inbound = _latest_inbound_message(request)
    if latest_inbound is None or request.previousToolResults:
        return None

    selection = _capture_selection(request, latest_inbound)
    context = _pending_free_text_capture_context(request)
    if selection is None and context is None:
        return None
    if context is not None:
        capture = context["fieldSchema"].get("x-chatbotinn-capture")
        explicit_catalog_confirmation = (
            context["offering"].requiresExplicitGuestConfirmation
            and isinstance(capture, dict)
            and str(capture.get("inputMode") or "").upper() == "CATALOG_ITEMS"
        )
        if not explicit_catalog_confirmation and not _supports_sequential_capture(context["offering"]):
            return None

    payload = _normalize_response_envelope(request, {
        "disposition": "RESPONSE_READY",
        "messages": [],
        "toolCalls": [],
        "updatedConversationSummary": None,
        "warnings": [],
    })
    payload = _normalize_guest_experience(request, payload)
    if not payload.get("messages") and not payload.get("toolCalls"):
        return None
    payload["usage"] = {
        "model": settings.openai_model,
        "inputTokens": 0,
        "cachedInputTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "totalTokens": 0,
        "latencyMs": round((time.perf_counter() - started_at) * 1000),
    }
    response = AgentTurnResponse.model_validate(payload)
    _validate_plan(request, response)
    return response


_ROOM_SERVICE_CHANGE_TASK_TYPES = {
    "ROOM_SERVICE_KITCHEN_CHANGE_DECISION",
    "ROOM_SERVICE_ORDER_CHANGE_DETAILS",
}

_MAINTENANCE_RESOLUTION_TASK_TYPE = "MAINTENANCE_RESOLUTION_CONFIRMATION"
_MAINTENANCE_RESOLUTION_REPLY = re.compile(
    r"^maintenance-resolution:([0-9a-f-]{36}):(RESOLVED|NOT_RESOLVED)$",
    re.IGNORECASE,
)


def _maintenance_resolution_task_plan(
    request: AgentTurnRequest,
    started_at: float,
    scope: ScopeDecision | None,
) -> AgentTurnResponse | None:
    """Consume a maintenance resolution reply once and keep it out of service capture."""
    completed_result = next(
        (
            result.result
            for result in request.previousToolResults
            if result.status == "SUCCEEDED"
            and result.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK.value
            and isinstance(result.result, dict)
            and result.result.get("taskType") == _MAINTENANCE_RESOLUTION_TASK_TYPE
        ),
        None,
    )
    if completed_result is not None:
        completion = completed_result.get("completionResult")
        resolved = completion.get("resolved") if isinstance(completion, dict) else None
        if resolved is not True:
            return _deterministic_turn_response(
                request,
                started_at,
                disposition="NO_ACTION",
                messages=[],
                updated_summary=request.conversation.summary,
            )
        task_id = str(completed_result.get("conversationTaskId") or "")
        operation_id = str(completed_result.get("operationId") or "")
        spanish = request.guest.preferredLanguage.lower().startswith("es")
        text = (
            "Gracias por confirmar. Nos alegra saber que el problema quedó resuelto. "
            "¿Hay algo más en lo que podamos ayudarte?"
            if spanish else
            "Thank you for confirming. We are glad the issue was resolved. "
            "Is there anything else we can help you with?"
        )
        return _deterministic_turn_response(
            request,
            started_at,
            disposition="RESPONSE_READY",
            messages=[{
                "messageDraftId": str(uuid4()),
                "purpose": "CLOSURE",
                "text": text,
                "language": request.guest.preferredLanguage,
                "operationIds": [operation_id] if operation_id else [],
                "conversationTaskIds": [task_id] if task_id else [],
                "interaction": None,
            }],
            updated_summary=request.conversation.summary,
        )

    if request.previousToolResults:
        return None
    if DomainToolName.COMPLETE_CONVERSATION_TASK not in request.toolPolicy.allowedTools:
        return None

    latest = _latest_inbound_message(request)
    if latest is None:
        return None
    candidates = _maintenance_resolution_tasks(request)
    if not candidates:
        return None

    reply_match = _MAINTENANCE_RESOLUTION_REPLY.fullmatch(
        (latest.interactionReplyId or "").strip()
    )
    task = None
    resolved = None
    if reply_match is not None:
        reply_task_id = UUID(reply_match.group(1))
        task = next(
            (candidate for candidate in candidates
             if candidate.conversationTaskId == reply_task_id),
            None,
        )
        if task is None:
            return None
        resolved = reply_match.group(2).upper() == "RESOLVED"
    else:
        # An explicit request for another service remains independent of this open task.
        if scope is not None and scope.kind == "SERVICE_REQUEST":
            return None
        if understanding_enabled():
            if latest.interactionReplyId:
                return None
            pending = _latest_capture_state(request).get("pendingOffering")
            if pending not in {None, "MAINTENANCE"} and not (
                    request.trigger.conversationTaskId or latest.conversationTaskIds
                    or request.trigger.operationId or latest.operationIds):
                return None
            task = _select_understood_task(request, latest, candidates)
            addressed_ids = ([request.trigger.conversationTaskId] if request.trigger.conversationTaskId else
                             latest.conversationTaskIds or [request.conversation.focusedConversationTaskId])
            if task is None and (any(addressed_ids) or request.trigger.operationId or latest.operationIds):
                return None
        resolved = _maintenance_resolution_value(latest)
        if understanding_enabled() and scope and scope.kind == "CONTEXT_REPLY":
            if task is None or resolved is None:
                return _task_decision_clarification(request, started_at, task)
        if resolved is None:
            return None
        if not understanding_enabled():
            focused_id = request.conversation.focusedConversationTaskId
            task = next(
                (candidate for candidate in candidates if candidate.conversationTaskId == focused_id),
                candidates[0] if len(candidates) == 1 else None,
            )
        if task is None:
            return None

    evidence_message_id = str(latest.messageId)
    return _deterministic_turn_response(
        request,
        started_at,
        disposition="TOOL_CALLS_REQUIRED",
        messages=[],
        tool_calls=[{
            "toolCallId": str(uuid4()),
            "toolName": DomainToolName.COMPLETE_CONVERSATION_TASK.value,
            "targetOperationId": str(task.operationId),
            "targetConversationTaskId": str(task.conversationTaskId),
            "arguments": {
                "conversationTaskId": str(task.conversationTaskId),
                "expectedVersion": task.version,
                "result": {"resolved": resolved},
            },
            "confidence": 1.0,
            "evidenceMessageIds": [evidence_message_id],
        }],
        updated_summary=request.conversation.summary,
    )


def _select_understood_task(request, message, candidates):
    task_ids = ([request.trigger.conversationTaskId] if request.trigger.conversationTaskId
                else message.conversationTaskIds)
    operation_ids = ([request.trigger.operationId] if request.trigger.operationId else message.operationIds)
    if task_ids:
        selected = [task for task in candidates if task.conversationTaskId in task_ids]
    elif operation_ids:
        selected = [task for task in candidates if task.operationId in operation_ids]
    elif semantic_action_offering(message.text):
        # Current, evidenced service selection takes precedence over historical focus.
        offering = semantic_action_offering(message.text)
        operation_ids = {operation.operationId for operation in request.activeOperations if operation.offeringCode == offering}
        selected = [task for task in candidates if task.operationId in operation_ids]
    elif request.conversation.focusedConversationTaskId:
        selected = [task for task in candidates
                    if task.conversationTaskId == request.conversation.focusedConversationTaskId]
    else:
        all_tasks = [task for operation in request.activeOperations for task in operation.pendingConversationTasks]
        selected = candidates if len(all_tasks) == 1 else []
    return selected[0] if len(selected) == 1 else None


def _task_decision_clarification(request, started_at, task=None):
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    text = (("¿El problema quedó resuelto o sigue pendiente?" if spanish else
             "Was the issue resolved, or is it still unresolved?") if task else
            ("¿A qué solicitud te refieres? Indica el folio o responde al mensaje de esa solicitud."
             if spanish else "Which request do you mean? Provide its reference or reply to that request's message."))
    options = ([{"id": f"maintenance-resolution:{task.conversationTaskId}:{action}", "label": label}
                for action, label in (("RESOLVED", "Resuelto" if spanish else "Resolved"),
                                      ("NOT_RESOLVED", "Sigue pendiente" if spanish else "Not resolved"))] if task else [])
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY", messages=[{
        "messageDraftId": str(uuid4()), "purpose": "CLARIFICATION", "text": text,
        "language": request.guest.preferredLanguage, "operationIds": [str(task.operationId)] if task else [],
        "conversationTaskIds": [], "interaction": {"type": "BUTTONS", "body": text, "options": options} if task else None,
    }], updated_summary=request.conversation.summary)


def _maintenance_resolution_tasks(request: AgentTurnRequest) -> list:
    return [
        task
        for operation in request.activeOperations
        for task in operation.pendingConversationTasks
        if task.taskType == _MAINTENANCE_RESOLUTION_TASK_TYPE
    ]


def _has_pending_maintenance_resolution_task(request: AgentTurnRequest) -> bool:
    return bool(_maintenance_resolution_tasks(request))


def _maintenance_resolution_value(message) -> bool | None:
    reply_match = _MAINTENANCE_RESOLUTION_REPLY.fullmatch(
        (message.interactionReplyId or "").strip()
    )
    if reply_match is not None:
        return reply_match.group(2).upper() == "RESOLVED"

    if understanding_enabled():
        action = semantic_action(message.text)
        return True if action in {"RESOLVED", "CONFIRM"} else False if action == "NOT_RESOLVED" else None

    text = _fold_text(message.text)
    negative_phrases = {
        "no", "no se resolvio", "no se soluciono", "no quedo resuelto",
        "no quedo solucionado", "no esta resuelto", "no esta solucionado", "sigue sin resolver",
        "todavia no", "aun no", "no funciona", "sigue igual",
        "el problema continua", "el problema sigue",
    }
    if text in negative_phrases or any(
        phrase in text for phrase in (
            "sigue sin resolver", "no se resolvio", "no se soluciono",
            "no quedo resuelto", "no quedo solucionado",
            "todavia no funciona", "aun no funciona",
        )
    ):
        return False

    positive_phrases = {
        "si", "confirmo", "resuelto", "solucionado", "ya quedo",
        "ya funciona", "todo bien", "quedo resuelto", "quedo solucionado",
        "el problema quedo resuelto", "el problema quedo solucionado",
    }
    if text in positive_phrases or any(
        phrase in text for phrase in (
            "quedo resuelto", "quedo solucionado", "ya se resolvio",
            "ya esta resuelto", "ya esta solucionado",
        )
    ):
        return True
    return None


def _room_service_operation_task_plan(
    request: AgentTurnRequest,
    started_at: float,
) -> AgentTurnResponse | None:
    """Resolve kitchen-requested order changes without a multi-tool model loop."""
    completed_result = next(
        (
            result.result
            for result in request.previousToolResults
            if result.status == "SUCCEEDED"
            and result.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK.value
            and isinstance(result.result, dict)
            and result.result.get("taskType") in _ROOM_SERVICE_CHANGE_TASK_TYPES
        ),
        None,
    )
    if completed_result is not None:
        return _deterministic_turn_response(
            request,
            started_at,
            disposition="NO_ACTION",
            messages=[],
            updated_summary=request.conversation.summary,
        )
    if request.previousToolResults:
        return None
    if DomainToolName.COMPLETE_CONVERSATION_TASK not in request.toolPolicy.allowedTools:
        return None

    focused_id = request.conversation.focusedConversationTaskId
    candidates = [
        task
        for operation in request.activeOperations
        for task in operation.pendingConversationTasks
        if task.taskType in _ROOM_SERVICE_CHANGE_TASK_TYPES
    ]
    task = next(
        (candidate for candidate in candidates if candidate.conversationTaskId == focused_id),
        candidates[0] if len(candidates) == 1 else None,
    )
    latest_inbound = _latest_inbound_message(request)
    if understanding_enabled() and latest_inbound:
        reply_id = latest_inbound.interactionReplyId or ""
        if reply_id and not reply_id.startswith(("room-service-change:", "catalog-choice:", "catalog-replacement:")):
            return None
        task = _select_understood_task(request, latest_inbound, candidates)
        if candidates and task is None and not _latest_capture_state(request).get("pendingOffering"):
            return _task_decision_clarification(request, started_at)
    if task is None or latest_inbound is None:
        return None

    offering = next((o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE"), None)
    if task.taskType == "ROOM_SERVICE_ORDER_CHANGE_DETAILS" and catalog_for(offering, "items"):
        return _catalog_replacement_plan(request, started_at, task, offering, latest_inbound)

    if task.taskType == "ROOM_SERVICE_KITCHEN_CHANGE_DECISION":
        decision = _room_service_confirmation_action(latest_inbound)
        if decision not in {"CHANGE", "CANCEL"}:
            if _is_free_text_change(latest_inbound.text):
                decision = "CHANGE"
            elif _is_free_text_cancel(latest_inbound.text):
                decision = "CANCEL"
        if decision not in {"CHANGE", "CANCEL"}:
            return _deterministic_turn_response(
                request,
                started_at,
                disposition="RESPONSE_READY",
                messages=[_room_service_kitchen_change_decision_message(request, task)],
                updated_summary=request.conversation.summary,
            )
        result = {"decision": decision}
    else:
        items = (understand_order(request, latest_inbound, [], require_full=True) if understanding_enabled()
                 else _parse_order_items(latest_inbound.text, []))
        if not items:
            return _deterministic_turn_response(
                request,
                started_at,
                disposition="RESPONSE_READY",
                messages=[_room_service_replacement_order_prompt(request, task)],
                updated_summary=request.conversation.summary,
            )
        result = {"items": items}

    evidence_message_id = str(latest_inbound.messageId)
    return _deterministic_turn_response(
        request,
        started_at,
        disposition="TOOL_CALLS_REQUIRED",
        messages=[],
        tool_calls=[{
            "toolCallId": str(uuid4()),
            "toolName": DomainToolName.COMPLETE_CONVERSATION_TASK.value,
            "targetOperationId": str(task.operationId),
            "targetConversationTaskId": str(task.conversationTaskId),
            "arguments": {
                "conversationTaskId": str(task.conversationTaskId),
                "expectedVersion": task.version,
                "result": result,
            },
            "confidence": 1.0,
            "evidenceMessageIds": [evidence_message_id],
        }],
        updated_summary=request.conversation.summary,
    )


def _catalog_replacement_plan(request, started_at, task, offering, message):
    state = deepcopy(_latest_capture_state(request))
    drafts = state.setdefault("catalogReplacementTasks", {})
    key = str(task.conversationTaskId)
    draft = drafts.get(key, {})
    if draft.get("version") != task.version:
        draft = {"version": task.version, "items": []}
    drafts[key] = draft
    old = draft.get("items", [])
    reply = message.interactionReplyId or ""
    expected = f"catalog-replacement:{key}:{draft.get('token')}:CONFIRM"
    choice = pending_choice(draft.get("pending"), message)
    if reply == expected and draft.get("token"):
        items, pending = normalize_order(offering, old, semantic=False)
        if items and not pending and items == old:
            return _deterministic_turn_response(request, started_at, disposition="TOOL_CALLS_REQUIRED", messages=[],
                tool_calls=[{"toolCallId": str(uuid4()), "toolName": "COMPLETE_CONVERSATION_TASK",
                    "targetOperationId": str(task.operationId), "targetConversationTaskId": key,
                    "arguments": {"conversationTaskId": key, "expectedVersion": task.version, "result": {"items": items}},
                    "confidence": 1.0, "evidenceMessageIds": [str(message.messageId)]}], updated_summary=request.conversation.summary)
    else:
        items = deepcopy(old)
        if choice:
            index = draft["pending"]["itemIndex"]
            items[index] = apply_choice(items[index], draft["pending"], choice)
        elif not reply:
            extracted = understand_order(request, message, items, allow_partial=True, require_full=not bool(items))
            if extracted is not None:
                items = extracted
        items, pending = normalize_order(offering, items)
    draft.update(items=items, pending=pending)
    draft.pop("token", None)
    if pending:
        output = catalog_clarification(request, offering, "items", pending)
    elif not items or any(type(i.get("quantity")) is not int or i["quantity"] < 1 for i in items):
        output = _room_service_replacement_order_prompt(request, task)
    else:
        draft["token"] = str(uuid4())
        output = _room_service_confirmation_message(request, offering, {"items": items})
        spanish = request.guest.preferredLanguage.lower().startswith("es")
        output["text"] = output["text"].rsplit("\n", 1)[0] + ("\nConfirma el pedido actualizado o escribe los cambios." if spanish else
                       "\nConfirm the updated order or type your changes.")
        output["interaction"] = {"type": "BUTTONS", "body": output["text"][:1024], "options": [{
            "id": f"catalog-replacement:{key}:{draft['token']}:CONFIRM", "label": "Confirmar" if spanish else "Confirm"}]}
    output["operationIds"] = [str(task.operationId)]
    output["conversationTaskIds"] = []
    messages = [output]
    if output.get("interaction") and len(output["text"]) > 1024:
        # Send the complete summary before a separate confirmation control.
        summary = {**output, "interaction": None}
        output = deepcopy(output)
        output["text"] = "Confirma el pedido actualizado." if request.guest.preferredLanguage.lower().startswith("es") else "Confirm the updated order."
        output["interaction"]["body"] = output["text"]
        messages = [summary, output]
    return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY", messages=messages,
                                       updated_summary=json.dumps(state, ensure_ascii=False))


def _room_service_kitchen_change_decision_message(request, task) -> dict:
    locale = request.guest.preferredLanguage
    text = template("order.kitchen.change", locale)
    offering = next((o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE"), None)
    if offering:
        catalog = (offering.inputSchema.get("properties", {}).get("items", {})
                   .get("x-chatbotinn-capture", {}).get("catalog", {}))
        if catalog.get("externalUrl"):
            text += "\n" + catalog["externalUrl"]
    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CLARIFICATION",
        "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [str(task.operationId)],
        "conversationTaskIds": [],
        "interaction": {
            "type": "BUTTONS",
            "title": template("order.kitchen.title", locale),
            "body": text,
            "buttonText": template("options.open", locale),
            "options": [
                {"id": "room-service-change:CHANGE", "label": template("action.change", locale)},
                {"id": "room-service-change:CANCEL", "label": template("order.cancel", locale)},
            ],
        },
    }


def _room_service_replacement_order_prompt(request, task) -> dict:
    if order_understanding_failed():
        message = _order_clarification_message(request)
        message["operationIds"] = [str(task.operationId)]
        return message
    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CLARIFICATION",
        "text": template("order.replacement", request.guest.preferredLanguage),
        "language": request.guest.preferredLanguage,
        "operationIds": [str(task.operationId)],
        "conversationTaskIds": [],
        "interaction": None,
    }


def _room_service_draft_plan(
    request: AgentTurnRequest,
    started_at: float,
) -> AgentTurnResponse | None:
    """Keep a room-service order draft deterministic across guest turns."""
    if request.previousToolResults:
        return None

    latest_inbound = _latest_inbound_message(request)
    state = _latest_capture_state(request)
    if latest_inbound is None or state.get("pendingOffering") != "ROOM_SERVICE":
        return None

    selection = _capture_selection(request, latest_inbound)
    if selection is not None:
        # Configured offering and field selections are handled by the generic capture flow.
        return None
    if _is_greeting_turn(request):
        return None

    offering = next(
        (
            candidate
            for candidate in request.availableOfferings
            if candidate.offeringCode == "ROOM_SERVICE"
        ),
        None,
    )
    if offering is None:
        return None

    captured = state.get("capturedFields")
    captured = dict(captured) if isinstance(captured, dict) else {}
    action = _room_service_confirmation_action(latest_inbound)
    awaiting_confirmation = bool(state.get("awaitingExplicitConfirmation"))
    unresolved_location = state.get("phase") == "NEEDS_LOCATION_CLARIFICATION"
    location_prompt = None
    if unresolved_location:
        field = offering.inputSchema.get("properties", {}).get("deliveryLocation", {})
        location_prompt = _capture_message(request, offering.offeringCode, "deliveryLocation",
                                           field, field.get("x-chatbotinn-capture", {}))

    if action == "CANCEL" or _is_free_text_cancel(latest_inbound.text):
        return _deterministic_turn_response(
            request,
            started_at,
            disposition="RESPONSE_READY",
            messages=[_room_service_cancellation_message(request)],
            updated_summary="{}",
        )

    catalog_pending = state.get("catalogPending")
    choice = pending_choice(catalog_pending, latest_inbound)
    if choice and catalog_for(offering, "items"):
        index = catalog_pending.get("itemIndex")
        if type(index) is int and 0 <= index < len(captured.get("items", [])):
            captured = deepcopy(captured)
            captured["items"][index] = apply_choice(captured["items"][index], catalog_pending, choice)
            return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                    **_room_service_capture_output(request, offering, captured, unresolved_location))
    if (latest_inbound.interactionReplyId or "").startswith("catalog-choice:"):
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                **_room_service_capture_output(request, offering, captured, unresolved_location))

    if unresolved_location and action in {"CHANGE", "CONFIRM"}:
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                messages=[location_prompt or _order_clarification_message(request)],
                updated_summary=_room_service_summary(captured, False, "NEEDS_LOCATION_CLARIFICATION"))

    if action == "CHANGE" or (
        awaiting_confirmation and _is_free_text_change(latest_inbound.text)
    ):
        if not understanding_enabled():
            captured.pop("items", None)
        return _deterministic_turn_response(
            request,
            started_at,
            disposition="RESPONSE_READY",
            messages=[_room_service_change_prompt(request)],
            updated_summary=_room_service_summary(captured, False, "CAPTURING_ITEMS"),
        )

    if action == "CONFIRM" or (
        awaiting_confirmation and _is_free_text_confirmation(latest_inbound.text)
    ):
        if understanding_enabled() and not awaiting_confirmation:
            return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                    messages=[_order_clarification_message(request)], updated_summary=request.conversation.summary)
        if DomainToolName.START_SERVICE not in request.toolPolicy.allowedTools:
            return None
        items = _coerce_order_items(captured.get("items"))
        delivery_location = captured.get("deliveryLocation")
        if not items or not isinstance(delivery_location, str) or not delivery_location:
            return None
        checked, pending = normalize_order(offering, captured.get("items", []), semantic=False)
        if pending or checked != captured.get("items"):
            captured = {**captured, "items": checked}
            return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                    **_room_service_capture_output(request, offering, captured, unresolved_location))
        evidence_message_id = str(latest_inbound.messageId)
        return _deterministic_turn_response(
            request,
            started_at,
            disposition="TOOL_CALLS_REQUIRED",
            messages=[],
            tool_calls=[{
                "toolCallId": str(uuid4()),
                "toolName": DomainToolName.START_SERVICE.value,
                "targetOperationId": None,
                "targetConversationTaskId": None,
                "arguments": {
                    "offeringCode": offering.offeringCode,
                    # BC-006 / BC-013: confirm the stored draft verbatim. Coercion
                    # above checks completeness; it must not rewrite business data
                    # after the guest has reviewed it.
                    "input": captured if understanding_enabled() else {
                        "deliveryLocation": delivery_location,
                        "items": items,
                    },
                    "guestConfirmationEvidenceMessageId": evidence_message_id,
                },
                "confidence": 1.0,
                "evidenceMessageIds": [evidence_message_id],
            }],
            updated_summary=_room_service_summary(captured, True, "STARTING"),
        )

    if (latest_inbound.interactionReplyId or "").strip():
        return None

    existing_items = _coerce_order_items(captured.get("items"), allow_partial=understanding_enabled())
    items = (understand_order(request, latest_inbound, existing_items, allow_partial=True) if understanding_enabled()
             else _parse_order_items(latest_inbound.text, existing_items))
    if not items:
        if understanding_enabled():
            return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                    messages=[_order_clarification_message(request)],
                    updated_summary=_room_service_summary(captured, False,
                            "NEEDS_LOCATION_CLARIFICATION" if unresolved_location else
                            "INPUT_RETRY" if order_understanding_failed() else "NEEDS_CLARIFICATION"))
        return None
    captured["items"] = items
    if understanding_enabled() or catalog_for(offering, "items"):
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                **_room_service_capture_output(request, offering, captured, unresolved_location))
    if unresolved_location:
        # An item edit does not resolve an earlier ambiguous delivery change.
        return _deterministic_turn_response(request, started_at, disposition="RESPONSE_READY",
                messages=[location_prompt or _order_clarification_message(request)],
                updated_summary=_room_service_summary(captured, False, "NEEDS_LOCATION_CLARIFICATION"))
    return _deterministic_turn_response(
        request,
        started_at,
        disposition="RESPONSE_READY",
        messages=[_room_service_confirmation_message(request, offering, captured)],
        updated_summary=_room_service_summary(captured, True, "AWAITING_CONFIRMATION"),
    )


def _deterministic_turn_response(
    request: AgentTurnRequest,
    started_at: float,
    *,
    disposition: str,
    messages: list[dict],
    updated_summary: str,
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
) -> AgentTurnResponse:
    payload = _normalize_response_envelope(request, {
        "disposition": disposition,
        "messages": messages,
        "toolCalls": tool_calls or [],
        "updatedConversationSummary": updated_summary,
        "warnings": [],
    })
    payload["usage"] = {
        "model": settings.openai_model,
        "inputTokens": 0,
        "cachedInputTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "totalTokens": 0,
        "latencyMs": round((time.perf_counter() - started_at) * 1000),
    }
    if usage:
        payload["usage"].update(usage)
    response = AgentTurnResponse.model_validate(payload)
    _validate_plan(request, response)
    return response


def _faq_service_start_plan(
    request: AgentTurnRequest,
    started_at: float,
) -> AgentTurnResponse | None:
    if DomainToolName.START_SERVICE not in request.toolPolicy.allowedTools:
        return None
    search_result = _successful_faq_search_result(request)
    if search_result is None or _successful_faq_start_result(request) is not None:
        return None

    latest_inbound = _latest_inbound_message(request)
    if latest_inbound is None:
        return None
    question = " ".join(str(search_result.get("query") or latest_inbound.text).split()).strip()
    if not question:
        return None

    semantic = language_enabled(request) and is_semantic_search(search_result)
    grounding, usage = None, None
    stale_source = _has_stale_faq_source(request)
    if semantic and not stale_source:
        grounding, usage = resolve_semantic_faq(request, search_result, started_at)
        match = restore_grounded_faq(request, search_result, grounding)
    else:
        match = None if stale_source else _exact_faq_match(search_result)
    service_input = {
        "question": question,
        "resolutionMode": "AUTOMATIC" if match is not None else "HUMAN_REQUIRED",
    }
    if match is not None:
        service_input.update({
            "knowledgeAnswer": str(match["answer"]).strip(),
            "knowledgeQuestion": str(match.get("question") or question).strip(),
            "knowledgeItemId": str(match.get("catalogItemId") or "").strip(),
        })

    summary = (_faq_capture_summary(question, grounding) if semantic else _capture_summary(
        request, "FAQ", {"question": question}, False))
    payload = _normalize_response_envelope(request, {
        "disposition": "TOOL_CALLS_REQUIRED",
        "messages": [],
        "toolCalls": [{
            "toolName": DomainToolName.START_SERVICE.value,
            "targetOperationId": None,
            "targetConversationTaskId": None,
            "arguments": {
                "offeringCode": "FAQ",
                "input": service_input,
            },
            "confidence": (float(match.get("confidence") or 0.0) if match else 0.0) if semantic
                          else float(search_result.get("confidence") or 0.0),
            "evidenceMessageIds": [str(latest_inbound.messageId)],
        }],
        "updatedConversationSummary": summary,
        "warnings": [],
    })
    return _zero_usage_response(request, payload, started_at, usage)


def _faq_capture_summary(question, grounding):
    return json.dumps({"pendingOffering": "FAQ", "capturedFields": {"question": question},
                       "readyToStart": False, "faqGrounding": grounding}, ensure_ascii=False)


def _has_stale_faq_source(request):
    return any(result.toolName == "START_SERVICE" and result.status == "REJECTED"
               and result.error and result.error.code == "FAQ_KNOWLEDGE_STALE"
               for result in request.previousToolResults)


def _is_verified_faq_response(request, response):
    search = _successful_faq_search_result(request)
    operation = _successful_faq_start_result(request)
    if not is_semantic_search(search) or operation is None or len(response.messages) != 1:
        return False
    match = restore_grounded_faq(request, search, _latest_capture_state(request).get("faqGrounding"))
    message = response.messages[0]
    # The checked guest-language answer must not be rewritten by presentation localization.
    return (match is not None and message.purpose == "ANSWER" and message.interaction is None
            and message.text == match["guestAnswer"] and message.language == request.guest.preferredLanguage
            and [str(value) for value in message.operationIds] == [str(operation.get("operationId"))])


def _faq_started_response_plan(
    request: AgentTurnRequest,
    started_at: float,
) -> AgentTurnResponse | None:
    operation = _successful_faq_start_result(request)
    search_result = _successful_faq_search_result(request)
    if operation is None or search_result is None:
        return None

    operation_id = str(operation.get("operationId") or "").strip()
    semantic = language_enabled(request) and is_semantic_search(search_result)
    match = (restore_grounded_faq(request, search_result, _latest_capture_state(request).get("faqGrounding"))
             if semantic else _exact_faq_match(search_result))
    if _has_stale_faq_source(request):
        match = None
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    if match is not None:
        text = match["guestAnswer"] if semantic else _compose_known_faq_answer(
            str(search_result.get("query") or ""),
            str(match["answer"]),
            request.guest.preferredLanguage,
        )
        purpose = "ANSWER"
    else:
        text = (
            "Lo siento, como asistente virtual no tengo información suficiente para responder "
            "tu pregunta. La compartiré con el equipo del hotel para que te respondan a la brevedad."
            if spanish
            else "I’m sorry, but I do not have enough information to answer your question. "
            "I’ll share it with the hotel team so they can reply shortly."
        )
        purpose = "HANDOFF"

    payload = _normalize_response_envelope(request, {
        "disposition": "RESPONSE_READY",
        "messages": [{
            "purpose": purpose,
            "text": text,
            "language": request.guest.preferredLanguage,
            "operationIds": [operation_id] if operation_id else [],
            "conversationTaskIds": [],
            "interaction": None,
        }],
        "toolCalls": [],
        "updatedConversationSummary": "{}",
        "warnings": [],
    })
    return _zero_usage_response(request, payload, started_at)


def _zero_usage_response(
    request: AgentTurnRequest,
    payload: dict,
    started_at: float,
    usage: dict | None = None,
) -> AgentTurnResponse:
    payload["usage"] = {
        "model": settings.openai_model,
        "inputTokens": 0,
        "cachedInputTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "totalTokens": 0,
        **(usage or {}),
        "latencyMs": round((time.perf_counter() - started_at) * 1000),
    }
    response = AgentTurnResponse.model_validate(payload)
    _validate_plan(request, response)
    return response


def _successful_faq_search_result(request: AgentTurnRequest) -> dict | None:
    return next((
        result.result
        for result in reversed(request.previousToolResults)
        if result.status == "SUCCEEDED"
        and result.toolName == DomainToolName.SEARCH_KNOWLEDGE.value
        and isinstance(result.result, dict)
    ), None)


def _successful_faq_start_result(request: AgentTurnRequest) -> dict | None:
    return next((
        result.result
        for result in reversed(request.previousToolResults)
        if result.status == "SUCCEEDED"
        and result.toolName == DomainToolName.START_SERVICE.value
        and isinstance(result.result, dict)
        and (
            result.result.get("offeringCode") is None
            or str(result.result.get("offeringCode")).upper() == "FAQ"
        )
    ), None)


def _exact_faq_match(search_result: dict) -> dict | None:
    if str(search_result.get("matchStatus") or "").upper() != "EXACT_MATCH":
        return None
    matches = search_result.get("matches")
    if not isinstance(matches, list) or not matches or not isinstance(matches[0], dict):
        return None
    answer = matches[0].get("answer")
    return matches[0] if isinstance(answer, str) and answer.strip() else None


def _compose_known_faq_answer(question: str, source_answer: str, language: str) -> str:
    answer = " ".join(source_answer.replace("\n", " ").split()).strip()
    answer = re.sub(r"^(?:respuesta|answer)\s*:\s*", "", answer, flags=re.IGNORECASE)
    spanish = language.lower().startswith("es")
    folded_question = _fold_text(question)

    if spanish and any(word in folded_question for word in ("cierra", "cierre", "cierran")):
        times = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", answer)
        subject = _spanish_faq_subject(question, ("cierra", "cierre", "cierran"))
        if times and subject:
            cadence = " todos los días" if "todos los dias" in _fold_text(answer) else ""
            answer = f"{subject} cierra{cadence} a las {_display_time(times[-1])}"
    elif spanish and any(word in folded_question for word in ("abre", "abren", "apertura")):
        times = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", answer)
        subject = _spanish_faq_subject(question, ("abre", "abren"))
        if times and subject:
            cadence = " todos los días" if "todos los dias" in _fold_text(answer) else ""
            answer = f"{subject} abre{cadence} a las {_display_time(times[0])}"

    answer = _deduplicate_sentences(answer)
    follow_up = (
        "¿Hay algo más en lo que pueda ayudarte?"
        if spanish
        else "Is there anything else I can help you with?"
    )
    if not _contains_faq_follow_up(answer, language):
        answer = f"{answer.rstrip()} {follow_up}"
    return answer


def _spanish_faq_subject(question: str, verbs: tuple[str, ...]) -> str | None:
    pattern = r"\b(?:" + "|".join(re.escape(verb) for verb in verbs) + r")\s+((?:el|la|los|las)\s+[^?.,]+)"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    subject = " ".join(match.group(1).split()).strip()
    return subject[:1].upper() + subject[1:]


def _display_time(value: str) -> str:
    hour, minute = (int(part) for part in value.split(":", 1))
    suffix = "a.m." if hour < 12 else "p.m."
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def _deduplicate_sentences(value: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", value) if sentence.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = _normalized_phrase(sentence)
        if normalized and normalized not in seen:
            unique.append(sentence)
            seen.add(normalized)
    return " ".join(unique).strip()


def _build_focused_task_repair_instruction(request: AgentTurnRequest) -> str:
    focused_task_id = request.conversation.focusedConversationTaskId
    if focused_task_id is None:
        return ""
    if any(
        result.status == "SUCCEEDED"
        and result.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK.value
        and isinstance(result.result, dict)
        and str(result.result.get("conversationTaskId")) == str(focused_task_id)
        for result in request.previousToolResults
    ):
        return ""

    focused_task = next(
        (
            task
            for operation in request.activeOperations
            for task in operation.pendingConversationTasks
            if task.conversationTaskId == focused_task_id
        ),
        None,
    )
    if focused_task is None:
        return ""

    latest_inbound = next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )
    if latest_inbound is None:
        return ""

    required_schema = json.dumps(
        focused_task.requiredOutputSchema,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\nThe turn is answering the focused conversation task. Do not send a guest-facing "
        "acknowledgement yet. Return disposition TOOL_CALLS_REQUIRED, messages [], and exactly "
        "one COMPLETE_CONVERSATION_TASK tool call. Use "
        f"targetOperationId={focused_task.operationId}, "
        f"targetConversationTaskId={focused_task.conversationTaskId}, "
        f"arguments.conversationTaskId={focused_task.conversationTaskId}, "
        f"arguments.expectedVersion={focused_task.version}, and "
        f"evidenceMessageIds=[{latest_inbound.messageId}]. "
        "Infer arguments.result from the guest's latest message and make it satisfy this exact "
        f"requiredOutputSchema: {required_schema}."
    )


def _build_capture_turn_instruction(request: AgentTurnRequest) -> str:
    context = _pending_free_text_capture_context(request)
    if context is None:
        return ""

    offering = context["offering"]
    field_code = context["fieldCode"]
    completed_values = json.dumps(
        context["completedValues"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    latest_inbound = context["latestInbound"]
    return (
        "\n\nAuthoritative capture state for this turn: "
        f"offeringCode={offering.offeringCode}, currentField={field_code}, "
        f"completedStructuredFields={completed_values}. "
        f"The latest inbound message {latest_inbound.messageId} is the guest's free-text answer "
        f"for currentField={field_code}. Extract and normalize its value. Do not repeat that "
        "field's introMessage and do not ask for information already present. If every required "
        "field is now available and the offering requires explicit guest confirmation, return a "
        "concise confirmation summary with Confirm, Change, and Cancel choices; do not call "
        "START_SERVICE until the guest confirms."
    )


def _normalize_response_envelope(
    request: AgentTurnRequest,
    model_payload: dict,
) -> dict:
    payload = dict(model_payload)
    # These fields are protocol metadata, not model decisions.
    payload["schemaVersion"] = "2.0"
    payload["agentTurnId"] = str(request.agentTurnId)
    payload.setdefault("detectedLanguage", None)
    payload.setdefault("messages", [])
    payload.setdefault("toolCalls", [])
    payload.setdefault("updatedConversationSummary", None)
    payload.setdefault("warnings", [])
    for message in payload["messages"]:
        if isinstance(message, dict):
            message["messageDraftId"] = str(uuid4())
            message.setdefault("operationIds", [])
            message.setdefault("conversationTaskIds", [])
    for tool_call in payload["toolCalls"]:
        if isinstance(tool_call, dict):
            tool_call["toolCallId"] = str(uuid4())
    return payload


def _validate_plan(
    request: AgentTurnRequest,
    response: AgentTurnResponse,
    *,
    enforce_faq_rewrite: bool = False,
) -> None:
    if response.agentTurnId != request.agentTurnId:
        raise AgentModelError("Agent turn ID does not match the request")

    allowed_tools = set(request.toolPolicy.allowedTools)
    message_ids = {message.messageId for message in request.conversation.recentMessages}
    all_operations = _all_operations(request)
    operation_ids = {operation.operationId for operation in all_operations}
    tasks_by_id = {
        task.conversationTaskId: task
        for operation in request.activeOperations
        for task in operation.pendingConversationTasks
    }
    task_ids = set(tasks_by_id)
    offerings = {offering.offeringCode: offering for offering in request.availableOfferings}
    operations_by_id = {operation.operationId: operation for operation in all_operations}

    if len(response.toolCalls) > request.toolPolicy.maxToolCalls:
        raise AgentModelError("Agent exceeded the tool-call limit")
    if response.disposition == "TOOL_CALLS_REQUIRED" and not response.toolCalls:
        raise AgentModelError("TOOL_CALLS_REQUIRED requires tool calls")
    if response.disposition != "TOOL_CALLS_REQUIRED" and response.toolCalls:
        raise AgentModelError("Only TOOL_CALLS_REQUIRED may contain tool calls")
    if response.disposition in {"RESPONSE_READY", "HANDOFF_REQUIRED"} and not response.messages:
        raise AgentModelError("The disposition requires a guest-facing message")
    if response.disposition == "NO_ACTION" and response.messages:
        raise AgentModelError("NO_ACTION cannot contain messages")

    _validate_faq_answer_style(request, response, enforce_rewrite=enforce_faq_rewrite)

    completed_tasks: set[UUID] = set()
    for call in response.toolCalls:
        unresolved = _latest_capture_state(request).get("serviceStartFailures", {})
        if (call.toolName == DomainToolName.START_SERVICE and unresolved
                and any(value.get("offeringCode") in {None, "UNKNOWN", call.arguments.get("offeringCode")}
                        for value in unresolved.values())):
            raise AgentModelError("Resolve the previous service-start failure before proposing another start")
        if call.toolName not in allowed_tools:
            raise AgentModelError(f"Tool {call.toolName.value} is not allowed")
        if call.targetOperationId is not None and call.targetOperationId not in operation_ids:
            raise AgentModelError("Tool call targets an operation outside the turn context")
        if call.targetConversationTaskId is not None and call.targetConversationTaskId not in task_ids:
            raise AgentModelError("Tool call targets a conversation task outside the turn context")
        if not set(call.evidenceMessageIds).issubset(message_ids):
            raise AgentModelError("Tool call contains evidence outside the turn context")
        _validate_lifecycle_call(call, offerings, operations_by_id)
        _validate_status_call(call, offerings)
        _validate_conversation_task_call(call, tasks_by_id)
        if call.toolName == DomainToolName.START_SERVICE and call.arguments.get("offeringCode") == "ROOM_SERVICE":
            item_input = call.arguments.get("input", {}).get("items", [])
            checked, pending = normalize_order(offerings.get("ROOM_SERVICE"), item_input, semantic=False)
            if pending or checked != item_input:
                raise AgentModelError("Room-service selection does not match the current catalog and prices")
        if call.toolName == DomainToolName.START_SERVICE and call.arguments.get("offeringCode") == "ROOM_SERVICE":
            if request.trigger.eventPayload.get('confirmationQueuedBeforePrompt') is True:
                raise AgentModelError('Queued confirmation predates the presented room-service summary')
            latest = _latest_inbound_message(request)
            if (not _room_confirmation_matches(request, latest)
                    or latest.interactionReplyId and (
                        _room_service_confirmation_action(latest) != "CONFIRM"
                        or call.arguments.get("input") != _latest_capture_state(request).get("capturedFields"))):
                raise AgentModelError("Room-service button does not authorize the current captured order")
        target_task = tasks_by_id.get(call.targetConversationTaskId)
        strict_replacement = (target_task and target_task.taskType == "ROOM_SERVICE_ORDER_CHANGE_DETAILS"
                              and catalog_for(offerings.get("ROOM_SERVICE"), "items"))
        if understanding_enabled() or strict_replacement:
            _validate_understood_call(request, call, tasks_by_id)
        validate_spa_call(request, call, tasks_by_id)
        if call.toolName.value == "COMPLETE_CONVERSATION_TASK" and call.targetConversationTaskId:
            completed_tasks.add(call.targetConversationTaskId)

    if not request.toolPolicy.allowMultipleConversationTaskCompletions and len(completed_tasks) > 1:
        raise AgentModelError("Multiple conversation-task completions are not allowed")

    successfully_completed_tasks = {
        UUID(str(result.result["conversationTaskId"]))
        for result in request.previousToolResults
        if result.status == "SUCCEEDED"
        and result.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK.value
        and isinstance(result.result, dict)
        and result.result.get("conversationTaskId")
    }
    known_task_ids = task_ids | successfully_completed_tasks
    for message in response.messages:
        referenced_tasks = set(message.conversationTaskIds)
        if not referenced_tasks.issubset(known_task_ids):
            raise AgentModelError(
                "Agent message references a conversation task outside the turn context"
            )
        if not referenced_tasks.issubset(successfully_completed_tasks):
            raise AgentModelError(
                "Complete the referenced conversation task before acknowledging it"
            )


def _validate_understood_call(request, call, tasks_by_id):
    latest = _latest_inbound_message(request)
    task = tasks_by_id.get(call.targetConversationTaskId)
    room_start = call.toolName == DomainToolName.START_SERVICE and call.arguments.get("offeringCode") == "ROOM_SERVICE"
    supported_task = (call.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK and task
                      and task.taskType in _ROOM_SERVICE_CHANGE_TASK_TYPES | {_MAINTENANCE_RESOLUTION_TASK_TYPE})
    if not room_start and not supported_task:
        return
    if latest is None or call.evidenceMessageIds != [latest.messageId]:
        raise AgentModelError("This decision requires evidence from the current guest message")
    if room_start:
        state = _latest_capture_state(request)
        if (state.get("pendingOffering") != "ROOM_SERVICE" or not state.get("awaitingExplicitConfirmation")
                or _room_service_confirmation_action(latest) != "CONFIRM"
                or call.arguments.get("input") != state.get("capturedFields")):
            raise AgentModelError("Room service requires explicit confirmation of the current captured order")
        return
    result = call.arguments.get("result", {})
    if task.taskType == _MAINTENANCE_RESOLUTION_TASK_TYPE:
        expected = _maintenance_resolution_value(latest)
        button = _MAINTENANCE_RESOLUTION_REPLY.fullmatch(latest.interactionReplyId or "")
        selected = (str(task.conversationTaskId) == button.group(1) if button else
                    _select_understood_task(request, latest, _maintenance_resolution_tasks(request)) == task)
        if not selected or expected is None or result.get("resolved") is not expected:
            raise AgentModelError("Maintenance resolution does not match the current guest decision")
    else:
        candidates = [t for op in request.activeOperations for t in op.pendingConversationTasks
                      if t.taskType in _ROOM_SERVICE_CHANGE_TASK_TYPES]
        if _select_understood_task(request, latest, candidates) != task:
            raise AgentModelError("Room-service task target is ambiguous")
        if task.taskType == "ROOM_SERVICE_KITCHEN_CHANGE_DECISION":
            expected = _room_service_confirmation_action(latest)
            if expected not in {"CHANGE", "CANCEL"} or result.get("decision") != expected:
                raise AgentModelError("Kitchen-change decision does not match current guest evidence")
        else:
            offering = next((o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE"), None)
            if catalog_for(offering, "items"):
                draft = _latest_capture_state(request).get("catalogReplacementTasks", {}).get(str(task.conversationTaskId), {})
                checked, pending = normalize_order(offering, draft.get("items", []), semantic=False)
                expected_reply = f"catalog-replacement:{task.conversationTaskId}:{draft.get('token')}:CONFIRM"
                if (draft.get("version") != task.version or not draft.get("token") or pending
                        or not checked or checked != draft.get("items") or result.get("items") != checked
                        or latest.interactionReplyId != expected_reply):
                    raise AgentModelError("Replacement requires confirmation of the current catalog selection")
                return
            items = understand_order(request, latest, [], require_full=True)
            if items is None or result.get("items") != items:
                raise AgentModelError("Replacement order does not match current guest evidence")


def _validate_faq_answer_style(
    request: AgentTurnRequest,
    response: AgentTurnResponse,
    *,
    enforce_rewrite: bool,
) -> None:
    source_answers = _successful_faq_source_answers(request)
    if not source_answers or response.disposition != "RESPONSE_READY":
        return

    answer_messages = [
        message
        for message in response.messages
        if message.purpose == "ANSWER"
    ]
    if not answer_messages:
        raise AgentModelError("An approved FAQ result requires a guest-facing answer")

    answer_text = " ".join(message.text.strip() for message in answer_messages).strip()
    if enforce_rewrite:
        normalized_answer = _normalized_phrase(answer_text)
        for source_answer in source_answers:
            normalized_source = _normalized_phrase(source_answer)
            if len(normalized_source.split()) >= 7 and normalized_source in normalized_answer:
                raise AgentModelError(
                    "Rewrite the approved FAQ facts naturally instead of copying the catalog answer verbatim"
                )


def _successful_faq_source_answers(request: AgentTurnRequest) -> list[str]:
    if _has_stale_faq_source(request):
        return []
    answers: list[str] = []
    for result in request.previousToolResults:
        if (
            result.status != "SUCCEEDED"
            or result.toolName != DomainToolName.SEARCH_KNOWLEDGE.value
            or not isinstance(result.result, dict)
        ):
            continue
        matches = result.result.get("matches")
        if isinstance(matches, list):
            if str(result.result.get("matchStatus") or "").upper() != "EXACT_MATCH":
                continue
            for match in matches:
                if isinstance(match, dict) and isinstance(match.get("answer"), str):
                    answers.append(match["answer"].strip())
        else:
            _collect_approved_faq_answers(result.result, answers)
    return list(dict.fromkeys(answers))


def _collect_approved_faq_answers(value, answers: list[str]) -> None:
    if isinstance(value, dict):
        configuration = value.get("faqConfiguration")
        if isinstance(configuration, dict) and configuration.get("approved") is True:
            answer = configuration.get("answer")
            if isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())
        for nested in value.values():
            _collect_approved_faq_answers(nested, answers)
    elif isinstance(value, list):
        for nested in value:
            _collect_approved_faq_answers(nested, answers)


def _contains_faq_follow_up(text: str, language: str) -> bool:
    normalized = _fold_text(text)
    if language.lower().startswith("es"):
        return (
            "algo mas" in normalized
            or "otra cosa" in normalized
            or "alguna otra" in normalized
        ) and (
            "ayud" in normalized
            or "necesit" in normalized
            or "gustaria" in normalized
        )
    if language.lower().startswith("en"):
        return ("anything else" in normalized or "something else" in normalized) and (
            "help" in normalized or "need" in normalized
        )
    return text.rstrip().endswith("?")


def _ensure_faq_follow_up(request: AgentTurnRequest, messages: list[dict]) -> None:
    if not _successful_faq_source_answers(request):
        return

    answer = next(
        (message for message in messages if message.get("purpose") == "ANSWER"),
        None,
    )
    if answer is None:
        return

    text = " ".join(str(answer.get("text") or "").strip().split()).strip()
    if not text:
        return
    if _contains_faq_follow_up(text, request.guest.preferredLanguage) or text.endswith("?"):
        answer["text"] = text
        return

    follow_up = (
        "¿Hay algo más en lo que pueda ayudarte?"
        if request.guest.preferredLanguage.lower().startswith("es")
        else "Is there anything else I can help you with?"
    )
    answer["text"] = f"{text} {follow_up}"


def _validate_lifecycle_call(call, offerings, operations_by_id) -> None:
    if call.toolName == DomainToolName.START_SERVICE:
        if call.targetOperationId is not None:
            raise AgentModelError("START_SERVICE cannot target an existing operation")
        offering_code = call.arguments.get("offeringCode")
        offering = offerings.get(offering_code)
        if offering is None:
            raise AgentModelError("START_SERVICE references an unavailable offering")
        if not isinstance(call.arguments.get("input"), dict):
            raise AgentModelError("START_SERVICE input must be an object")
        _validate_offering_input(
            call.arguments["input"],
            offering.inputSchema,
            offering.offeringCode,
        )
        if not call.evidenceMessageIds:
            raise AgentModelError("START_SERVICE requires guest evidence")
        if offering.requiresExplicitGuestConfirmation:
            confirmation = _argument_uuid(
                call.arguments.get("guestConfirmationEvidenceMessageId"),
                "START_SERVICE confirmation evidence",
            )
            if confirmation not in set(call.evidenceMessageIds):
                raise AgentModelError("START_SERVICE confirmation evidence is not declared")

    if call.toolName == DomainToolName.EXECUTE_SERVICE_ACTION:
        if call.targetOperationId is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION requires targetOperationId")
        operation = operations_by_id.get(call.targetOperationId)
        if operation is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION targets an unavailable operation")
        argument_operation_id = _argument_uuid(
            call.arguments.get("operationId"),
            "EXECUTE_SERVICE_ACTION operationId",
        )
        if argument_operation_id != call.targetOperationId:
            raise AgentModelError("EXECUTE_SERVICE_ACTION operation IDs do not match")
        if call.arguments.get("expectedVersion") != operation.version:
            raise AgentModelError("EXECUTE_SERVICE_ACTION uses a stale operation version")
        if not isinstance(call.arguments.get("input", {}), dict):
            raise AgentModelError("EXECUTE_SERVICE_ACTION input must be an object")
        action_code = call.arguments.get("actionCode")
        action = next(
            (candidate for candidate in operation.availableActions
             if candidate.actionCode == action_code),
            None,
        )
        if action is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION references an unavailable action")
        argument_evidence = {
            _argument_uuid(value, "EXECUTE_SERVICE_ACTION evidence")
            for value in call.arguments.get("evidenceMessageIds", [])
        }
        if argument_evidence != set(call.evidenceMessageIds):
            raise AgentModelError("EXECUTE_SERVICE_ACTION evidence declarations do not match")
        if action.requiresExplicitGuestConfirmation and not call.evidenceMessageIds:
            raise AgentModelError("EXECUTE_SERVICE_ACTION requires explicit guest evidence")


def _validate_offering_input(value: dict, schema: dict, offering_code: str) -> None:
    if schema.get("type") not in {None, "object"}:
        raise AgentModelError(f"Offering {offering_code} input schema must describe an object")

    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    for field in required:
        field_value = value.get(field)
        if _is_missing_required_value(field_value):
            raise AgentModelError(
                f"START_SERVICE requires a non-empty value for offering field {field}"
            )
        field_schema = properties.get(field)
        if isinstance(field_schema, dict):
            _validate_schema_value(field_value, field_schema, field)


def _validate_schema_value(value, schema: dict, field: str) -> None:
    expected = schema.get("type")
    valid = {
        "string": isinstance(value, str) and bool(value.strip()),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list) and bool(value),
        "object": isinstance(value, dict) and bool(value),
    }.get(expected, True)
    if not valid:
        raise AgentModelError(f"START_SERVICE offering field {field} must be {expected}")

    capture = schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return
    input_mode = str(capture.get("inputMode") or "AUTO").upper()
    allowed_codes = _capture_option_codes(capture)
    if input_mode == "SINGLE_SELECT" and allowed_codes and value not in allowed_codes:
        raise AgentModelError(
            f"START_SERVICE offering field {field} must use a configured option code"
        )
    if input_mode == "MULTI_SELECT" and allowed_codes:
        if not isinstance(value, list) or any(item not in allowed_codes for item in value):
            raise AgentModelError(
                f"START_SERVICE offering field {field} must use configured option codes"
            )


def _is_missing_required_value(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, dict)) and not value
    )


def _validate_status_call(call, offerings) -> None:
    if call.toolName != DomainToolName.GET_OPERATION_STATUS:
        return
    reference_code = call.arguments.get("referenceCode")
    offering_code = call.arguments.get("offeringCode")
    if not isinstance(reference_code, str) and not isinstance(offering_code, str):
        raise AgentModelError("GET_OPERATION_STATUS requires referenceCode or offeringCode")
    if isinstance(offering_code, str) and offering_code not in offerings:
        raise AgentModelError("GET_OPERATION_STATUS references an unavailable offering")


def _all_operations(request: AgentTurnRequest):
    operations = {operation.operationId: operation for operation in request.recentOperations}
    operations.update({operation.operationId: operation for operation in request.activeOperations})
    return list(operations.values())


def _normalize_guest_experience(request: AgentTurnRequest, payload: dict,
                                skip_capture: bool = False) -> dict:
    normalized = dict(payload)
    messages = [dict(message) for message in normalized.get("messages", []) if isinstance(message, dict)]
    normalized["messages"] = messages
    _remove_maintenance_issue_interactions(request, messages)
    _ensure_faq_follow_up(request, messages)
    if not skip_capture and _ensure_explicit_capture_confirmation(request, normalized, messages):
        return normalized
    if not skip_capture and _ensure_sequential_field_capture(request, normalized, messages):
        return normalized
    if not skip_capture and _ensure_configured_field_capture(request, normalized, messages):
        return normalized
    if _is_greeting_turn(request) and not request.previousToolResults:
        # A greeting opens a navigation turn. Historical operations remain available
        # for later status questions, but they can never trigger side effects here.
        normalized["disposition"] = "RESPONSE_READY"
        normalized["toolCalls"] = []
        normalized["updatedConversationSummary"] = "{}"
        _ensure_personalized_service_menu(request, messages)
        return normalized
    started = _acknowledgeable_service_starts(request)
    if started:
        # A successful lifecycle result is authoritative. The same guest turn must
        # acknowledge it instead of proposing START_SERVICE again.
        normalized["disposition"] = "RESPONSE_READY"
        normalized["toolCalls"] = []
        normalized["updatedConversationSummary"] = _started_service_summary(request, started)
        _ensure_service_start_acknowledgements(request, messages)
        return normalized
    if normalized.get("disposition") == "RESPONSE_READY":
        _ensure_personalized_service_menu(request, messages)
        _ensure_service_start_acknowledgements(request, messages)
        if messages:
            normalized["disposition"] = "RESPONSE_READY"
    return normalized


def _ensure_sequential_field_capture(
    request: AgentTurnRequest,
    normalized: dict,
    messages: list[dict],
) -> bool:
    context = _pending_free_text_capture_context(request)
    if context is None:
        return False
    offering = context["offering"]
    if offering.requiresExplicitGuestConfirmation or not _supports_sequential_capture(offering):
        return False

    value = " ".join(context["latestInbound"].text.strip().split()).strip()
    if not value:
        return False
    completed_values = {
        **context["completedValues"],
        context["fieldCode"]: value,
    }
    remaining = [
        (field_code, field_schema)
        for field_code, field_schema in _ordered_guest_capture_fields(offering)
        if field_code not in completed_values
    ]
    if remaining:
        field_code, field_schema = remaining[0]
        capture = field_schema.get("x-chatbotinn-capture")
        if not isinstance(capture, dict):
            return False
        message = _capture_message(
            request,
            offering.offeringCode,
            field_code,
            field_schema,
            capture,
        )
        if message is None:
            return False
        messages[:] = [message]
        normalized["disposition"] = "RESPONSE_READY"
        normalized["toolCalls"] = []
        normalized["updatedConversationSummary"] = _capture_summary(
            request,
            offering.offeringCode,
            completed_values,
            False,
        )
        return True

    latest_message_id = str(context["latestInbound"].messageId)
    messages[:] = []
    normalized["disposition"] = "TOOL_CALLS_REQUIRED"
    normalized["toolCalls"] = [{
        "toolCallId": str(uuid4()),
        "toolName": DomainToolName.START_SERVICE.value,
        "targetOperationId": None,
        "targetConversationTaskId": None,
        "arguments": {
            "offeringCode": offering.offeringCode,
            "input": completed_values,
        },
        "confidence": 1.0,
        "evidenceMessageIds": [latest_message_id],
    }]
    normalized["updatedConversationSummary"] = _capture_summary(
        request,
        offering.offeringCode,
        completed_values,
        True,
    )
    return True


def _supports_sequential_capture(offering) -> bool:
    fields = _ordered_guest_capture_fields(offering)
    if not fields:
        return False
    modes = {
        str(field_schema.get("x-chatbotinn-capture", {}).get("inputMode") or "AUTO").upper()
        for _, field_schema in fields
    }
    return modes.issubset({"FREE_TEXT", "DATE", "TIME", "CATALOG_ITEMS"})


def _capture_summary(
    request: AgentTurnRequest,
    offering_code: str,
    completed_values: dict[str, str],
    ready_to_start: bool,
) -> str:
    state = json.dumps(
        {
            "pendingOffering": offering_code,
            "capturedFields": completed_values,
            "readyToStart": ready_to_start,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    existing = request.conversation.summary.strip()
    return f"{existing}\n{state}" if existing else state


def _latest_capture_state(request: AgentTurnRequest) -> dict:
    summary = request.conversation.summary.strip()
    if not summary:
        return {}
    for candidate in reversed(summary.splitlines()):
        try:
            value = json.loads(candidate.strip())
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    try:
        value = json.loads(summary)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _room_service_summary(
    captured_fields: dict,
    awaiting_confirmation: bool,
    phase: str,
) -> str:
    return json.dumps(
        {
            "pendingOffering": "ROOM_SERVICE",
            "phase": phase,
            "capturedFields": captured_fields,
            "awaitingExplicitConfirmation": awaiting_confirmation,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _room_service_confirmation_action(latest_inbound) -> str | None:
    reply_id = (latest_inbound.interactionReplyId or "").strip().upper()
    if understanding_enabled():
        if not reply_id:
            return semantic_action(latest_inbound.text)
        if not reply_id.startswith(("CONFIRMATION:ROOM_SERVICE:", "ROOM-SERVICE:", "ROOM-SERVICE-CHANGE:")):
            return None
    for action in ("CONFIRM", "CHANGE", "CANCEL"):
        if reply_id.endswith(f":{action}"):
            return action
    legacy = {
        "ROOM-SERVICE:CONFIRM": "CONFIRM",
        "ROOM-SERVICE:CHANGE": "CHANGE",
        "ROOM-SERVICE:CANCEL": "CANCEL",
    }
    return legacy.get(reply_id)


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.strip().split()).strip("!?., ")


def _is_free_text_cancel(value: str) -> bool:
    if understanding_enabled():
        return semantic_action(value) == "CANCEL"
    text = _fold_text(value)
    return text in {
        "cancelar",
        "cancela",
        "cancelalo",
        "cancela mi pedido",
        "cancelar mi pedido",
        "ya no quiero el pedido",
        "cancel",
        "cancel order",
        "cancel my order",
    }


def _is_free_text_change(value: str) -> bool:
    if understanding_enabled():
        return semantic_action(value) == "CHANGE"
    text = _fold_text(value)
    return text in {
        "cambiar",
        "cambia",
        "cambiar pedido",
        "cambiar mi pedido",
        "quiero cambiar",
        "quiero cambiar el pedido",
        "modificar",
        "change",
        "change order",
    }


def _is_free_text_confirmation(value: str) -> bool:
    if understanding_enabled():
        return semantic_action(value) == "CONFIRM"
    text = _fold_text(value)
    return text in {
        "confirmar",
        "confirmo",
        "confirmado",
        "si",
        "si confirmo",
        "es correcto",
        "correcto",
        "confirm",
        "confirmed",
        "yes",
    }


def _coerce_order_items(value, *, allow_partial=False) -> list[dict]:
    strict = understanding_enabled()
    if isinstance(value, str):
        return [] if strict else _parse_order_items(value, [])
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, dict):
            if strict:
                return []
            continue
        name = " ".join(str(item.get("name") or "").split()).strip(" .,;")
        quantity = item.get("quantity", None if strict else 1)
        missing_quantity = strict and allow_partial and quantity is None
        if not name or (not missing_quantity and (not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1)):
            if strict:
                return []
            continue
        modifications = item.get("modifications")
        if not isinstance(modifications, list):
            modifications = []
        items.append({
            "name": name,
            **({} if missing_quantity else {"quantity": quantity}),
            "modifications": [str(entry) for entry in modifications if str(entry).strip()],
            **({"catalogSelection": deepcopy(item["catalogSelection"])} if isinstance(item.get("catalogSelection"), dict) else {}),
        })
    return items


_QUANTITY_WORDS = {
    "un": 1,
    "una": 1,
    "uno": 1,
    "unos": 1,
    "unas": 1,
    "one": 1,
    "dos": 2,
    "two": 2,
    "tres": 3,
    "three": 3,
    "cuatro": 4,
    "four": 4,
    "cinco": 5,
    "five": 5,
}


def _parse_order_items(value: str, existing_items: list[dict]) -> list[dict]:
    text = " ".join(value.strip().split()).strip(" .,;")
    if not text:
        return []

    folded = _fold_text(text)
    if existing_items and "cada uno" in folded:
        quantities = _extract_quantities(folded)
        quantity = quantities[0] if quantities else 1
        return [{**item, "quantity": quantity} for item in existing_items]

    quantity_only_parts = re.split(r"\s*(?:,|\by\b|\band\b)\s*", folded)
    if existing_items and len(quantity_only_parts) == len(existing_items):
        quantities = [_parse_quantity(part) for part in quantity_only_parts]
        if all(quantity is not None for quantity in quantities):
            return [
                {**item, "quantity": int(quantity)}
                for item, quantity in zip(existing_items, quantities)
            ]

    text = re.sub(r"^trame\s+(?:mejor\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:(?:por favor|please)\s+)?(?:trae(?:me)?|tráe(?:me)?|quiero|quisiera|"
        r"dame|ponme|mejor|cambia(?:me)?(?:\s+mejor)?)(?:\s+por favor)?\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^mejor\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+con\s+(?=(?:\d+|un|una|uno|unos|unas|one|two|dos|tres|three)\b)",
        " y ",
        text,
        flags=re.IGNORECASE,
    )
    parts = [
        part.strip(" .,;")
        for part in re.split(r"\s*(?:,|\by\b|\band\b)\s*", text, flags=re.IGNORECASE)
        if part.strip(" .,;")
    ]
    items = []
    quantity_pattern = "|".join(sorted(_QUANTITY_WORDS, key=len, reverse=True))
    for part in parts:
        part = re.sub(r"^(?:por favor|please)\s+", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\s+(?:por favor|please)$", "", part, flags=re.IGNORECASE)
        match = re.match(
            rf"^(?P<quantity>\d+|{quantity_pattern})\s+(?:de\s+)?(?P<name>.+)$",
            part,
            flags=re.IGNORECASE,
        )
        if match:
            quantity = _parse_quantity(match.group("quantity")) or 1
            name = match.group("name").strip(" .,;")
        else:
            quantity = 1
            name = part.strip(" .,;")
        if name:
            items.append({"name": name, "quantity": quantity, "modifications": []})
    return items


def _parse_quantity(value: str) -> int | None:
    folded = _fold_text(value)
    if folded.isdigit():
        quantity = int(folded)
        return quantity if quantity > 0 else None
    return _QUANTITY_WORDS.get(folded)


def _extract_quantities(value: str) -> list[int]:
    return [
        quantity
        for token in re.findall(r"\d+|[a-záéíóúñ]+", value.casefold())
        if (quantity := _parse_quantity(token)) is not None
    ]


def _room_service_confirmation_message(request: AgentTurnRequest, offering, captured: dict) -> dict:
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    items = _coerce_order_items(captured.get("items"))
    item_lines = [f"- {item['quantity']} x {item['name']}"
                  + (" — " + "; ".join(item["catalogSelection"].get("optionNames", []))
                     if item.get("catalogSelection", {}).get("optionNames") else "")
                  + (" (" + "; ".join(item["modifications"]) + ")" if item["modifications"] else "")
                  + (f" — {item['catalogSelection']['currency']} {item['catalogSelection']['lineTotal']}"
                     if item.get("catalogSelection", {}).get("lineTotal") else "")
                  for item in items]
    if items and all(i.get("catalogSelection", {}).get("lineTotal") for i in items):
        currencies = {i["catalogSelection"]["currency"] for i in items}
        if len(currencies) == 1:
            from decimal import Decimal
            total = sum((Decimal(i["catalogSelection"]["lineTotal"]) for i in items), Decimal(0))
            item_lines.append(f"\nTotal: {next(iter(currencies))} {total:.2f}")
    destination = _capture_value_label(
        offering,
        "deliveryLocation",
        captured.get("deliveryLocation"),
    )
    if spanish:
        lines = ["Confirmación de pedido", "", "Artículos:"]
        lines.extend(item_lines)
        if destination:
            lines.extend(["", f"Lugar de entrega: {destination}"])
        lines.extend(["", "¿Deseas confirmar, cambiar o cancelar el pedido?"])
        title = "Confirmación de pedido"
        labels = ("Confirmar", "Cambiar", "Cancelar")
    else:
        lines = ["Order confirmation", "", "Items:"]
        lines.extend(item_lines)
        if destination:
            lines.extend(["", f"Delivery location: {destination}"])
        lines.extend(["", "Would you like to confirm, change, or cancel the order?"])
        title = "Order confirmation"
        labels = ("Confirm", "Change", "Cancel")
    text = "\n".join(lines)
    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CONFIRMATION",
        "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [],
        "conversationTaskIds": [],
        "interaction": None if understanding_enabled() and len(text) > 1024 else {
            "type": "BUTTONS",
            "title": title,
            "body": text[:1024],
            "buttonText": "",
            "options": [
                {"id": "confirmation:ROOM_SERVICE:CONFIRM", "label": labels[0]},
                {"id": "confirmation:ROOM_SERVICE:CHANGE", "label": labels[1]},
                {"id": "confirmation:ROOM_SERVICE:CANCEL", "label": labels[2]},
            ],
        },
    }


def _order_clarification_message(request: AgentTurnRequest) -> dict:
    key = "order.input.clarify"
    if order_understanding_failed():
        key = "order.input.unavailable"
    elif order_understanding_issue() == "MISSING_QUANTITY":
        key = "order.input.quantity"
    elif order_understanding_issue() == "COMPLETE_ORDER_REQUIRED":
        key = "order.replacement"
    text = template(key, request.guest.preferredLanguage)
    return {"messageDraftId": str(uuid4()), "purpose": "CLARIFICATION", "text": text,
            "language": request.guest.preferredLanguage, "operationIds": [],
            "conversationTaskIds": [], "interaction": None}


def _room_service_capture_output(request, offering, captured, unresolved_location=False):
    """BC-001/002/006: keep partial drafts; confirmation needs quantities and location."""
    checked, catalog_pending = normalize_order(offering, captured.get("items", []))
    captured = {**captured, "items": checked}
    if catalog_pending:
        summary = json.loads(_room_service_summary(captured, False, "NEEDS_CATALOG_SELECTION"))
        summary["catalogPending"] = catalog_pending
        return {"messages": [catalog_clarification(request, offering, "items", catalog_pending)],
                "updated_summary": json.dumps(summary, ensure_ascii=False, separators=(",", ":"))}
    pending = [item for item in captured.get("items", []) if item.get("quantity") is None]
    if pending:
        message = _order_clarification_message(request)
        message["text"] = template("order.input.pending_quantity", request.guest.preferredLanguage)
        message["text"] += "\n" + "\n".join("- " + item["name"] for item in pending)
        phase = "NEEDS_LOCATION_CLARIFICATION" if unresolved_location else "NEEDS_QUANTITY"
        awaiting = False
    elif unresolved_location or not captured.get("deliveryLocation"):
        field = offering.inputSchema.get("properties", {}).get("deliveryLocation", {})
        message = _capture_message(request, offering.offeringCode, "deliveryLocation",
                                   field, field.get("x-chatbotinn-capture", {}))
        message = message or _order_clarification_message(request)
        phase = "NEEDS_LOCATION_CLARIFICATION" if unresolved_location else "CAPTURING_LOCATION"
        awaiting = False
    else:
        message = _room_service_confirmation_message(request, offering, captured)
        phase, awaiting = "AWAITING_CONFIRMATION", True
    return {"messages": [message], "updated_summary": _room_service_summary(captured, awaiting, phase)}


def _room_service_change_prompt(request: AgentTurnRequest) -> dict:
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    text = (
        "Indícame nuevamente el pedido completo, incluyendo productos, cantidades y modificaciones."
        if spanish
        else "Please provide the complete order again, including items, quantities, and changes."
    )
    if understanding_enabled():
        text = template("order.change", request.guest.preferredLanguage)
    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CLARIFICATION",
        "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [],
        "conversationTaskIds": [],
        "interaction": None,
    }


def _room_service_cancellation_message(request: AgentTurnRequest) -> dict:
    return {
        "messageDraftId": str(uuid4()),
        "purpose": "ANSWER",
        "text": template("order.cancelled", request.guest.preferredLanguage),
        "language": request.guest.preferredLanguage,
        "operationIds": [],
        "conversationTaskIds": [],
        "interaction": None,
    }


def _ensure_explicit_capture_confirmation(
    request: AgentTurnRequest,
    normalized: dict,
    messages: list[dict],
) -> bool:
    context = _pending_free_text_capture_context(request)
    if context is None:
        return False

    offering = context["offering"]
    field_schema = context["fieldSchema"]
    capture = field_schema.get("x-chatbotinn-capture")
    if (
        not offering.requiresExplicitGuestConfirmation
        or not isinstance(capture, dict)
        or str(capture.get("inputMode") or "").upper() != "CATALOG_ITEMS"
        or offering.offeringCode != "ROOM_SERVICE"
    ):
        return False

    latest_inbound = context["latestInbound"]
    existing_items = _coerce_order_items(context["completedValues"].get(context["fieldCode"]), allow_partial=understanding_enabled())
    items = (understand_order(request, latest_inbound, existing_items, allow_partial=True) if understanding_enabled()
             else _parse_order_items(latest_inbound.text, existing_items))
    if not items:
        if understanding_enabled():
            messages[:] = [_order_clarification_message(request)]
            normalized.update(disposition="RESPONSE_READY", toolCalls=[], updatedConversationSummary=
                              _room_service_summary(context["completedValues"], False,
                                  "INPUT_RETRY" if order_understanding_failed() else "NEEDS_CLARIFICATION"))
            return True
        return False
    completed_values = {
        **context["completedValues"],
        context["fieldCode"]: items,
    }
    if understanding_enabled() or catalog_for(offering, "items"):
        output = _room_service_capture_output(request, offering, completed_values)
        messages[:] = output["messages"]
        normalized.update(disposition="RESPONSE_READY", toolCalls=[], updatedConversationSummary=output["updated_summary"])
        return True
    messages[:] = [_room_service_confirmation_message(request, offering, completed_values)]
    normalized["disposition"] = "RESPONSE_READY"
    normalized["toolCalls"] = []
    normalized["updatedConversationSummary"] = _room_service_summary(
        completed_values,
        True,
        "AWAITING_CONFIRMATION",
    )
    return True


def _pending_free_text_capture_context(request: AgentTurnRequest) -> dict | None:
    latest_inbound = _latest_inbound_message(request)
    if latest_inbound is None or (latest_inbound.interactionReplyId or "").strip():
        return None
    if not latest_inbound.text.strip():
        return None

    state = _latest_capture_state(request)
    pending_offering_code = state.get("pendingOffering")
    captured_state = state.get("capturedFields")
    if isinstance(pending_offering_code, str) and isinstance(captured_state, dict):
        offering = next(
            (
                candidate
                for candidate in request.availableOfferings
                if candidate.offeringCode == pending_offering_code
            ),
            None,
        )
        if offering is not None and not state.get("awaitingExplicitConfirmation"):
            for field_code, field_schema in _ordered_guest_capture_fields(offering):
                if field_code in captured_state:
                    continue
                capture = field_schema.get("x-chatbotinn-capture")
                input_mode = (
                    str(capture.get("inputMode") or "AUTO").upper()
                    if isinstance(capture, dict)
                    else "AUTO"
                )
                if input_mode in {"FREE_TEXT", "DATE", "TIME", "MULTI_SELECT", "CATALOG_ITEMS"}:
                    return {
                        "offering": offering,
                        "fieldCode": field_code,
                        "fieldSchema": field_schema,
                        "completedValues": dict(captured_state),
                        "latestInbound": latest_inbound,
                    }
                break

        # Explicit draft state takes precedence over an older menu selection in history.
        return None

    selected_offering = None
    selected_at = -1
    completed_values: dict[str, str] = {}
    for index, message in enumerate(request.conversation.recentMessages):
        if message.direction != "INBOUND":
            continue
        selection = _capture_selection(request, message)
        if selection is None:
            continue
        offering, field_code = selection
        if offering is None:
            continue
        if field_code is None:
            selected_offering = offering
            selected_at = index
            completed_values = {}
            continue
        if selected_offering is None or selected_offering.offeringCode != offering.offeringCode:
            selected_offering = offering
            selected_at = index
            completed_values = {}
        field_value = _structured_capture_value(message.interactionReplyId)
        if field_value is not None:
            completed_values[field_code] = field_value

    if selected_offering is None:
        return None
    latest_index = next(
        (
            index
            for index in range(len(request.conversation.recentMessages) - 1, -1, -1)
            if request.conversation.recentMessages[index].messageId == latest_inbound.messageId
        ),
        -1,
    )
    if latest_index <= selected_at:
        return None

    completed_values.update(_completed_free_text_capture_values(
        request,
        selected_offering,
        selected_at,
        latest_index,
        completed_values,
    ))

    for field_code, field_schema in _ordered_guest_capture_fields(selected_offering):
        if field_code in completed_values:
            continue
        capture = field_schema.get("x-chatbotinn-capture")
        input_mode = str(capture.get("inputMode") or "AUTO").upper() if isinstance(capture, dict) else "AUTO"
        if input_mode not in {"FREE_TEXT", "DATE", "TIME", "MULTI_SELECT", "CATALOG_ITEMS"}:
            return None
        previous_message = next(
            (
                request.conversation.recentMessages[index]
                for index in range(latest_index - 1, selected_at - 1, -1)
                if request.conversation.recentMessages[index].direction != "INTERNAL"
            ),
            None,
        )
        if not _is_capture_prompt_message(previous_message, field_schema):
            return None
        return {
            "offering": selected_offering,
            "fieldCode": field_code,
            "fieldSchema": field_schema,
            "completedValues": completed_values,
            "latestInbound": latest_inbound,
        }
    return None


def _completed_free_text_capture_values(
    request: AgentTurnRequest,
    offering,
    selected_at: int,
    latest_index: int,
    structured_values: dict[str, str],
) -> dict[str, str]:
    completed = dict(structured_values)
    pending_field = None
    fields = _ordered_guest_capture_fields(offering)
    for message in request.conversation.recentMessages[selected_at + 1:latest_index]:
        if message.direction == "OUTBOUND":
            pending_field = next(
                (
                    field_code
                    for field_code, field_schema in fields
                    if field_code not in completed
                    and _is_capture_prompt_message(message, field_schema)
                ),
                None,
            )
            continue
        if message.direction != "INBOUND" or pending_field is None:
            continue
        if (message.interactionReplyId or "").strip():
            pending_field = None
            continue
        value = " ".join(message.text.strip().split()).strip()
        if value:
            completed[pending_field] = value
        pending_field = None
    return completed


def _is_capture_prompt_message(message, field_schema: dict) -> bool:
    if message is None or message.direction != "OUTBOUND":
        return False
    prompt_text = _normalized_phrase(message.text)
    if not prompt_text:
        return False
    capture = field_schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return False
    intro_message = capture.get("introMessage")
    if isinstance(intro_message, str):
        normalized_intro = _normalized_phrase(intro_message)
        if normalized_intro and normalized_intro in prompt_text:
            return True
        intro_tokens = {
            token.strip("!?.,;:")
            for token in normalized_intro.split()
            if len(token.strip("!?.,;:")) >= 4
        }
        prompt_tokens = {
            token.strip("!?.,;:")
            for token in prompt_text.split()
            if len(token.strip("!?.,;:")) >= 4
        }
        smaller_count = min(len(intro_tokens), len(prompt_tokens))
        if smaller_count and len(intro_tokens & prompt_tokens) / smaller_count >= 0.4:
            return True
    catalog = capture.get("catalog")
    external_url = catalog.get("externalUrl") if isinstance(catalog, dict) else None
    return isinstance(external_url, str) and external_url.strip() in message.text


def _structured_capture_value(reply_id: str | None) -> str | None:
    if not reply_id or not reply_id.startswith("field:"):
        return None
    parts = reply_id.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[3] else None


def _capture_value_label(offering, field_code: str, value: str | None) -> str | None:
    if not value:
        return None
    properties = offering.inputSchema.get("properties")
    field_schema = properties.get(field_code) if isinstance(properties, dict) else None
    capture = field_schema.get("x-chatbotinn-capture") if isinstance(field_schema, dict) else None
    if isinstance(capture, dict):
        option = next(
            (option for option in _capture_options(capture) if option["code"] == value),
            None,
        )
        if option is not None:
            return str(option["label"])
    return value


def _ensure_configured_field_capture(
    request: AgentTurnRequest,
    normalized: dict,
    messages: list[dict],
) -> bool:
    latest_inbound = _latest_inbound_message(request)
    if latest_inbound is None or request.previousToolResults:
        return False

    selection = _capture_selection(request, latest_inbound)
    if selection is None:
        return False
    offering, completed_field = selection
    fields = _ordered_guest_capture_fields(offering)
    if not fields:
        return False

    if understanding_enabled() and offering.offeringCode == "ROOM_SERVICE" and completed_field == "deliveryLocation":
        state = _latest_capture_state(request)
        captured = dict(state.get("capturedFields", {})) if state.get("pendingOffering") == "ROOM_SERVICE" else {}
        items = _coerce_order_items(captured.get("items"), allow_partial=True)
        location = _structured_capture_value(latest_inbound.interactionReplyId)
        if items and location:
            # Preserve guest wording; strict catalog drafts also recheck their live selections.
            captured["deliveryLocation"] = location
            if catalog_for(offering, "items") or any(item.get("quantity") is None for item in items):
                output = _room_service_capture_output(request, offering, captured)
                messages[:] = output["messages"]
                normalized.update(disposition="RESPONSE_READY", toolCalls=[], updatedConversationSummary=output["updated_summary"])
                return True
            confirmation = _room_service_confirmation_message(request, offering, captured)
            catalog = offering.inputSchema.get("properties", {}).get("items", {}).get("x-chatbotinn-capture", {}).get("catalog", {})
            url = catalog.get("externalUrl")
            if isinstance(url, str) and url:
                confirmation["text"] += "\n" + url
                if len(confirmation["text"]) > 1024:
                    confirmation["interaction"] = None
                elif confirmation["interaction"]:
                    confirmation["interaction"]["body"] = confirmation["text"]
            messages[:] = [confirmation]
            normalized.update(disposition="RESPONSE_READY", toolCalls=[], updatedConversationSummary=
                              _room_service_summary(captured, True, "AWAITING_CONFIRMATION"))
            return True

    field_index = 0
    if completed_field is not None:
        matching_index = next(
            (index for index, (code, _) in enumerate(fields) if code == completed_field),
            None,
        )
        if matching_index is None or matching_index + 1 >= len(fields):
            return False
        field_index = matching_index + 1

    field_code, field_schema = fields[field_index]
    capture = field_schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return False
    message = _capture_message(request, offering.offeringCode, field_code, field_schema, capture)
    if message is None:
        return False

    messages[:] = [message]
    normalized["disposition"] = "RESPONSE_READY"
    normalized["toolCalls"] = []
    state = _latest_capture_state(request)
    captured = state.get("capturedFields")
    completed_values = (dict(captured) if state.get("pendingOffering") == offering.offeringCode
                        and isinstance(captured, dict) else {})
    if completed_field is not None:
        completed_value = _structured_capture_value(latest_inbound.interactionReplyId)
        if completed_value is not None:
            completed_values[completed_field] = completed_value
    normalized["updatedConversationSummary"] = _capture_summary(
        request,
        offering.offeringCode,
        completed_values,
        False,
    )
    return True


def _capture_selection(request: AgentTurnRequest, latest_inbound):
    reply_id = (latest_inbound.interactionReplyId or "").strip()
    offerings = {offering.offeringCode: offering for offering in request.availableOfferings}
    if reply_id.startswith("offering:"):
        offering_code = reply_id.split(":", 1)[1]
        offering = offerings.get(offering_code)
        return (offering, None) if offering is not None else None

    if reply_id.startswith("field:"):
        parts = reply_id.split(":", 3)
        if len(parts) != 4:
            return None
        _, offering_code, field_code, option_code = parts
        offering = offerings.get(offering_code)
        if offering is None:
            return None
        properties = offering.inputSchema.get("properties")
        field_schema = properties.get(field_code) if isinstance(properties, dict) else None
        if not isinstance(field_schema, dict):
            return None
        capture = field_schema.get("x-chatbotinn-capture")
        if not isinstance(capture, dict) or option_code not in _capture_option_codes(capture):
            return None
        return offering, field_code

    normalized_text = _normalized_phrase(latest_inbound.text)
    for offering in request.availableOfferings:
        if normalized_text in {
            _normalized_phrase(offering.offeringCode.replace("_", " ")),
            _normalized_phrase(offering.name),
        }:
            return offering, None
    return None


def _capture_message(
    request: AgentTurnRequest,
    offering_code: str,
    field_code: str,
    field_schema: dict,
    capture: dict,
) -> dict | None:
    input_mode = str(capture.get("inputMode") or "AUTO").upper()
    if input_mode == "AUTO":
        return None
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    title = str(field_schema.get("title") or field_code).strip()
    text = str(
        capture.get("introMessage")
        or field_schema.get("description")
        or (
            f"Por favor, indica {title.lower()}."
            if spanish
            else f"Please provide {title.lower()}."
        )
    ).strip()
    interaction = None

    if language_enabled(request) and not capture.get("introMessage") and not field_schema.get("description"):
        text = template("capture.required", request.guest.preferredLanguage, field=title.lower())

    if input_mode == "SINGLE_SELECT":
        options = _capture_options(capture)
        if not options:
            return None
        interaction_options = [
            {
                "id": f"field:{offering_code}:{field_code}:{option['code']}",
                "label": str(option["label"])[:24],
            }
            for option in options[:10]
        ]
        interaction = {
            "type": "BUTTONS" if len(interaction_options) <= 3 else "LIST",
            "title": title[:60],
            "body": text[:1024],
            "buttonText": template("options.open", request.guest.preferredLanguage) if language_enabled(request) else "Ver opciones" if spanish else "View options",
            "options": interaction_options,
        }
    elif input_mode == "MULTI_SELECT":
        labels = [str(option["label"]) for option in _capture_options(capture)]
        if labels:
            choices = ", ".join(labels)
            suffix = (
                f" Opciones disponibles: {choices}. Indica todas las que deseas."
                if spanish
                else f" Available options: {choices}. Tell us every option you want."
            )
            text = (text + suffix)[:20000]
    elif input_mode == "CATALOG_ITEMS":
        catalog = capture.get("catalog")
        external_url = catalog.get("externalUrl") if isinstance(catalog, dict) else None
        if isinstance(external_url, str) and external_url.strip() and external_url not in text:
            text = f"{text}\n{external_url.strip()}"

    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CLARIFICATION",
        "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [],
        "conversationTaskIds": [],
        "interaction": interaction,
    }


def _capture_options(capture: dict) -> list[dict]:
    catalog = capture.get("catalog")
    raw_options = catalog.get("options") if isinstance(catalog, dict) else None
    if not isinstance(raw_options, list):
        return []
    return [
        option for option in raw_options
        if isinstance(option, dict)
        and isinstance(option.get("code"), str)
        and option["code"].strip()
        and isinstance(option.get("label"), str)
        and option["label"].strip()
    ]


def _capture_option_codes(capture: dict) -> set[str]:
    return {str(option["code"]) for option in _capture_options(capture)}


def _latest_inbound_message(request: AgentTurnRequest):
    if request.trigger.messageId is not None:
        message = next((m for m in request.conversation.recentMessages
                        if m.messageId == request.trigger.messageId and m.direction == "INBOUND"), None)
        if message is not None:
            return message
    return next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )


def _normalized_phrase(value: str) -> str:
    return " ".join(value.casefold().strip().replace("_", " ").split()).strip("!?., ")


def _remove_maintenance_issue_interactions(
    request: AgentTurnRequest,
    messages: list[dict],
) -> None:
    latest_inbound = next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )
    if latest_inbound is None:
        return

    reply_id = (latest_inbound.interactionReplyId or "").strip().casefold()
    normalized_text = " ".join(latest_inbound.text.casefold().strip().split()).strip("!?., ")
    selected_maintenance = reply_id == "offering:maintenance" or normalized_text in {
        "maintenance",
        "mantenimiento",
        "servicio de mantenimiento",
    }
    if not selected_maintenance:
        return

    # The issue is an unconstrained guest description. Model-generated categories
    # would discard useful details and can route the request incorrectly.
    for message in messages:
        message["interaction"] = None


def _ensure_personalized_service_menu(request: AgentTurnRequest, messages: list[dict],
                                      force: bool = False) -> None:
    if not request.availableOfferings or (not force and not _is_greeting_turn(request)):
        return
    if not messages:
        messages.append({
            "messageDraftId": str(uuid4()),
            "purpose": "ANSWER",
            "text": "",
            "language": request.guest.preferredLanguage,
            "operationIds": [],
            "conversationTaskIds": [],
            "interaction": None,
        })
    message = messages[0]
    display_name = request.guest.displayName.strip()
    first_name = display_name.split()[0] if display_name else ""
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    if spanish:
        salutation = f"Hola, {first_name}. " if first_name else "Hola. "
        text = (
            salutation
            + "\u00bfC\u00f3mo podemos ayudarte hoy? Por favor, elige una opci\u00f3n del men\u00fa."
        )
    else:
        salutation = f"Hello, {first_name}. " if first_name else "Hello. "
        text = salutation + "How can we help you today? Please choose an option from the menu."

    options = [
        {"id": f"offering:{offering.offeringCode}", "label": offering.name[:24]}
        for offering in request.availableOfferings[:10]
    ]
    if language_enabled(request):
        text = (template("greeting.named", request.guest.preferredLanguage, name=first_name)
                if first_name else template("greeting", request.guest.preferredLanguage))
    message["text"] = text
    message["interaction"] = {
        "type": "BUTTONS" if len(options) <= 3 else "LIST",
        "title": template("menu.title", request.guest.preferredLanguage) if language_enabled(request) else "Servicios del hotel" if spanish else "Hotel services",
        "body": text,
        "buttonText": template("menu.open", request.guest.preferredLanguage) if language_enabled(request) else "Ver servicios" if spanish else "View services",
        "options": options,
    }


def _ensure_service_start_acknowledgements(request: AgentTurnRequest, messages: list[dict]) -> None:
    started = _acknowledgeable_service_starts(request)
    if not started:
        return
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    offering_names = {
        offering.offeringCode: offering.name for offering in request.availableOfferings
    }
    acknowledgements: list[dict] = []
    for operation in started:
        reference = str(operation["referenceCode"])
        operation_id = str(operation.get("operationId") or "")
        offering_code = str(operation.get("offeringCode") or "")
        if offering_code == "FAQ":
            continue
        offering_name = offering_names.get(offering_code, offering_code.replace("_", " ").title())
        text = (
            f"La solicitud de {offering_name.lower()} ha sido iniciada con el folio {reference}. "
            "Recibirás actualizaciones por este medio; por favor, mantente atento."
            if spanish
            else f"The {offering_name.lower()} request has been started with reference {reference}. "
            "We will send updates through this channel; please keep an eye on your messages."
        )
        if offering_code == "FRONT_DESK":
            text = (
                f"Hemos avisado a recepción. Tu solicitud quedó registrada con el folio {reference}. "
                "El equipo se pondrá en contacto contigo directamente para ayudarte."
                if spanish else
                f"We have notified the front desk. Your request reference is {reference}. "
                "The team will contact you directly to help."
            )
        if language_enabled(request):
            text = (template("frontdesk.started", request.guest.preferredLanguage, reference=reference)
                    if offering_code == "FRONT_DESK" else
                    template("service.started", request.guest.preferredLanguage,
                             service=offering_name.lower(), reference=reference))
        linked_operations = [operation_id] if operation_id else []
        acknowledgements.append({
            "messageDraftId": str(uuid4()),
            "purpose": "STATUS_UPDATE",
            "text": text,
            "language": request.guest.preferredLanguage,
            "operationIds": linked_operations,
            "conversationTaskIds": [],
            "interaction": None,
        })
    if acknowledgements:
        messages[:] = acknowledgements


def _acknowledgeable_service_starts(request: AgentTurnRequest) -> list[dict]:
    # Do not hide failures, FAQ responses, or other mutations behind a start-only reply.
    supporting_tools = {DomainToolName.LIST_AVAILABLE_OFFERINGS.value,
                        DomainToolName.GET_OFFERING_DEFINITION.value,
                        DomainToolName.SEARCH_CATALOG.value}
    started = []
    seen_operations: set[UUID] = set()
    for result in request.previousToolResults:
        if result.status != "SUCCEEDED" or result.error is not None:
            return []
        if result.toolName in supporting_tools:
            continue
        if result.toolName != DomainToolName.START_SERVICE.value:
            return []
        operation = result.result
        if not isinstance(operation, dict) or any(
            not isinstance(operation.get(key), str) or not operation[key].strip()
            for key in ("offeringCode", "referenceCode", "operationId")
        ):
            return []
        if operation["offeringCode"].strip().upper() == "FAQ":
            return []
        try:
            operation_id = UUID(operation["operationId"])
        except ValueError:
            return []
        if operation_id in seen_operations:
            continue
        seen_operations.add(operation_id)
        started.append(operation)
    return started if len(started) <= 10 else []


def _started_service_summary(request: AgentTurnRequest, started: list[dict]) -> str:
    offering_codes = {operation["offeringCode"] for operation in started}
    if "SPA" in offering_codes:
        request = request.model_copy(deep=True)
        request.conversation.summary = started_spa_summary(request)
    summary = request.conversation.summary
    try:
        state = json.loads(summary)
        structured_summary = True
    except ValueError:
        state = _latest_capture_state(request)
        structured_summary = False
    if not isinstance(state, dict):
        return summary
    pending_offering = state.get("pendingOffering")
    submitted = (state.get("phase") == "STARTING" or state.get("readyToStart") is True
                 or pending_offering == "FRONT_DESK")
    if not isinstance(pending_offering, str) or pending_offering not in offering_codes or not submitted:
        return summary
    for key in ("pendingOffering", "capturedFields", "awaitingExplicitConfirmation",
                "readyToStart", "phase"):
        state.pop(key, None)
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if structured_summary:
        return encoded
    # Replace only the latest capture snapshot, keeping prose and independent state.
    lines = summary.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        try:
            if isinstance(json.loads(lines[index]), dict):
                lines[index] = encoded
                break
        except ValueError:
            continue
    return "\n".join(lines)


def _is_greeting_turn(request: AgentTurnRequest) -> bool:
    inbound = _latest_inbound_message(request)
    if inbound is None:
        return False
    text = " ".join(inbound.text.casefold().strip().split()).strip("!?., ")
    if language_enabled(request) and greeting_language(text):
        return True
    return text in {
        "hola", "hello", "hi", "hey", "buen dia", "buen día", "buenos dias",
        "buenos días", "buenas tardes", "buenas noches", "que tal", "qué tal",
    }


def _validate_conversation_task_call(call, tasks_by_id) -> None:
    if call.toolName not in {
        DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS,
        DomainToolName.COMPLETE_CONVERSATION_TASK,
    }:
        return
    if call.targetConversationTaskId is None:
        raise AgentModelError(f"{call.toolName.value} requires targetConversationTaskId")

    task = tasks_by_id.get(call.targetConversationTaskId)
    if task is None:
        raise AgentModelError("Conversation-task tool targets an unavailable task")
    argument_task_id = _argument_uuid(
        call.arguments.get("conversationTaskId"),
        f"{call.toolName.value} conversationTaskId",
    )
    if argument_task_id != call.targetConversationTaskId:
        raise AgentModelError("Conversation-task IDs do not match")
    if call.targetOperationId is not None and call.targetOperationId != task.operationId:
        raise AgentModelError("Conversation task belongs to another operation")
    if call.arguments.get("expectedVersion") != task.version:
        raise AgentModelError("Conversation-task tool uses a stale task version")
    if not call.evidenceMessageIds:
        raise AgentModelError("Conversation-task tool requires guest evidence")

    payload_name = (
        "partialResult"
        if call.toolName == DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS
        else "result"
    )
    payload = call.arguments.get(payload_name)
    if not isinstance(payload, dict):
        raise AgentModelError(f"{call.toolName.value} {payload_name} must be an object")
    if call.toolName == DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS:
        accumulated = dict(task.partialResult)
        accumulated.update(payload)
        if _satisfies_required_output_schema(accumulated, task.requiredOutputSchema):
            raise AgentModelError(
                "Conversation-task progress already satisfies the required output schema; "
                "use COMPLETE_CONVERSATION_TASK"
            )
    elif not _satisfies_required_output_schema(payload, task.requiredOutputSchema):
        raise AgentModelError("Conversation-task result does not satisfy requiredOutputSchema")


def _satisfies_required_output_schema(value, schema: dict) -> bool:
    return satisfies_schema(value, schema)


def _argument_uuid(value, field: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise AgentModelError(f"{field} must be a UUID") from exc
