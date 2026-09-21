"""Resolve historical maintenance replies before they can become new issue descriptions."""
from copy import deepcopy
import json
import re
from uuid import uuid4

from app.schemas.v2_turns import DomainToolName

RESOLUTION = re.compile(r"^maintenance-resolution:([a-fA-F0-9-]{36}):(RESOLVED|NOT_RESOLVED)$")
RECURRENCE = re.compile(r"^maintenance-recurrence:([a-fA-F0-9-]{36}):([a-f0-9]{32}):(CONFIRM|CANCEL)$")
YES = {"si", "sí", "yes", "confirmar", "confirm", "sí, abrir folio", "open new request"}
NO = {"no", "cancel", "cancelar", "ahora no", "not now"}
REFERENCE_QUESTION = (
    '¿A qué folio de mantenimiento te refieres? Así conservaré los datos del problema correcto.',
    'Which maintenance request do you mean? That will let me preserve the correct issue details.',
)
UNKNOWN_REFERENCE = (
    'No encuentro ese folio entre tus reportes de mantenimiento disponibles. Revisa el folio del mensaje de mantenimiento y envíamelo completo.',
    'I cannot find that reference among your available maintenance requests. Please check the maintenance message and send its complete reference.',
)


def plan_maintenance_recurrence(request, state, latest, scope=None):
    if request.trigger.type != 'INBOUND_MESSAGE' or request.previousToolResults or latest is None:
        return None
    reply = latest.interactionReplyId or ''
    resolution, recurrence = RESOLUTION.fullmatch(reply), RECURRENCE.fullmatch(reply)
    pending = state.get('maintenanceRecurrence')
    pending = pending if isinstance(pending, dict) else {}
    spanish = request.guest.preferredLanguage.lower().startswith('es')
    text = latest.text.strip().lower()
    context = request.trigger.eventPayload.get('maintenanceButtonContext') or {}
    source = context.get('source')
    written = bool(scope and scope.kind in {'SERVICE_REQUEST', 'CONTEXT_REPLY', 'STATUS_REQUEST'}
                   and scope.maintenanceFollowUp == 'RECURRENCE'
                   and scope.maintenanceFollowUpConfidence >= .9
                   and scope.maintenanceFollowUpEvidence == latest.text)
    operations = {str(o.operationId): o for o in [*request.recentOperations, *request.activeOperations]
                  if o.offeringCode == 'MAINTENANCE'}
    last_outbound = next((m for m in reversed(request.conversation.recentMessages)
                          if m.direction == 'OUTBOUND'), None)
    # Also recover conversations that asked this question before the state was persisted.
    reference_prompt = bool(last_outbound and last_outbound.text in (*REFERENCE_QUESTION, *UNKNOWN_REFERENCE))
    awaiting_reference = pending.get('phase') == 'AWAITING_REFERENCE' or (not pending and reference_prompt)

    def referenced_operations():
        return [o for o in operations.values() if o.referenceCode and re.search(
            r'(?<![\w-])' + re.escape(o.referenceCode) + r'(?![\w-])', text, re.IGNORECASE)]

    def response(es, en, operation=None, summary=None, interaction=None):
        message = {'purpose': 'ANSWER', 'text': es if spanish else en,
                   'language': request.guest.preferredLanguage,
                   'operationIds': [operation['operationId']] if operation else [],
                   'conversationTaskIds': [], 'interaction': interaction}
        return dict(disposition='RESPONSE_READY', messages=[message],
                    updated_summary=request.conversation.summary if summary is None else summary)

    def summary(value):
        updated = deepcopy(state)
        if value is None:
            updated.pop('maintenanceRecurrence', None)
        else:
            updated['maintenanceRecurrence'] = value
        encoded = json.dumps(updated, ensure_ascii=False)
        original = request.conversation.summary
        try:
            if isinstance(json.loads(original), dict):
                return encoded
        except ValueError:
            pass
        lines = original.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            try:
                if isinstance(json.loads(lines[index]), dict):
                    lines[index] = encoded
                    return '\n'.join(lines)
            except ValueError:
                pass
        return original.rstrip() + ('\n' if original.strip() else '') + encoded

    if resolution:
        if any(str(t.conversationTaskId) == resolution[1]
               for op in request.activeOperations for t in op.pendingConversationTasks):
            return None  # The ordinary open-task handler preserves the existing folio.
        if not source:
            return response('No pude identificar el reporte de ese botón. Indícame el folio o el problema que necesita atención.',
                            'I could not identify the report for that button. Please share the reference or the issue needing attention.')
        ref = source.get('referenceCode') or source['operationId']
        if resolution[2] == 'RESOLVED':
            if source['detailedStatus'] == 'RESOLUTION_CONFIRMED':
                return response(f'La confirmación de solución del folio {ref} ya estaba registrada.',
                                f'Resolution of request {ref} was already confirmed.', source)
            return response(f'Ese botón corresponde a una revisión anterior del folio {ref}. No registré una nueva confirmación. Si el problema continúa, indícamelo.',
                            f'That button belongs to an earlier review of request {ref}. I did not record a new confirmation. Please tell me if the issue continues.', source)
    elif recurrence:
        if (not source or pending.get('sourceOperationId') != recurrence[1]
                or pending.get('token') != recurrence[2]) and not context.get('recurrence'):
            return response('Esa confirmación ya no corresponde a una solicitud pendiente. Indícame si el problema volvió a presentarse.',
                            'That confirmation no longer belongs to a pending request. Please tell me if the issue has returned.')
    elif not reply and awaiting_reference and reference_prompt and (
            text in YES | NO or referenced_operations() or re.search(r'\b(?:req|test)-[\w-]+', text)):
        if text in NO:
            return response('De acuerdo. No abriré otro folio de mantenimiento.',
                            'Understood. I will not open another maintenance request.', summary=summary(None))
        candidates = referenced_operations()
        allowed = pending.get('candidateOperationIds', list(operations))
        if (len(candidates) != 1 or str(candidates[0].operationId) not in allowed
                or text != (candidates[0].referenceCode or '').lower()):
            return response(*(REFERENCE_QUESTION if text in YES else UNKNOWN_REFERENCE),
                            summary=summary({'phase': 'AWAITING_REFERENCE', 'candidateOperationIds': allowed}))
        source = candidates[0].model_dump(mode='json')
    elif not reply and pending.get('source') and text in YES | NO:
        source = pending.get('source')
    elif written:
        # An open maintenance confirmation must remain with its existing task.
        if any(o.pendingConversationTasks for o in operations.values()):
            return None
        refs = referenced_operations()
        candidates = refs if refs else list(operations.values())
        if len(candidates) != 1:
            return response(*REFERENCE_QUESTION, summary=summary({
                'phase': 'AWAITING_REFERENCE', 'candidateOperationIds': list(operations)}))
        source = candidates[0].model_dump(mode='json')
    elif reply.startswith(('maintenance-resolution:', 'maintenance-recurrence:')):
        return response('No pude identificar ese botón de mantenimiento. Indícame el folio para revisar su estado.',
                        'I could not identify that maintenance button. Please share the request reference.')
    else:
        return None

    if not source:
        return response('Indícame qué problema volvió a presentarse.', 'Please tell me which issue has returned.')
    ref = source.get('referenceCode') or source['operationId']
    if context.get('recurrence'):
        existing = context['recurrence']
        existing_ref = existing.get('referenceCode') or existing['operationId']
        active = existing['lifecycle'] in {'ACTIVE', 'WAITING_FOR_GUEST', 'WAITING_FOR_STAFF'}
        return response(f'La reincidencia del folio {ref} ya tiene el folio {existing_ref}. '
                        + ('Sigue abierto para seguimiento.' if active else 'Ese seguimiento ya cerró. Si volvió a fallar, indícame ese último folio.'),
                        f'The recurrence of request {ref} already has reference {existing_ref}. '
                        + ('It remains open for follow-up.' if active else 'That follow-up is closed. If the issue returned again, please refer to that latest request.'),
                        existing, summary(None))
    failures = state.get('serviceStartFailures', {})
    if any(f.get('offeringCode', 'UNKNOWN') in {'MAINTENANCE', 'UNKNOWN'}
           for f in failures.values() if isinstance(f, dict)):
        return response('El inicio anterior de mantenimiento quedó sin confirmar. Es necesario revisar su resultado antes de intentar abrir otro folio.',
                        'The previous maintenance start is unconfirmed. Its result must be checked before attempting another request.', source)
    if source['lifecycle'] != 'COMPLETED':
        return response(f'El folio {ref} sigue en seguimiento. No abrí otro folio.',
                        f'Request {ref} is still being followed up. I did not open another request.', source, summary(None))
    cancelled = bool(recurrence and recurrence[3] == 'CANCEL' or not reply and pending and text in NO)
    if cancelled:
        return response('De acuerdo. No abriré otro folio de mantenimiento.',
                        'Understood. I will not open another maintenance request.', source, summary(None))
    confirmed = bool(recurrence and recurrence[3] == 'CONFIRM' or not reply and pending and text in YES)
    issue = source.get('input', {}).get('issue')
    if not isinstance(issue, str) or not issue.strip() or issue.strip().lower() in {'not resolved', 'no resuelto', 'sigue pendiente'}:
        return response(f'El folio {ref} no conserva una descripción suficiente. ¿Qué problema necesita atención?',
                        f'Request {ref} does not contain enough issue details. What needs attention?', source)
    offering = next((o for o in request.availableOfferings if o.offeringCode == 'MAINTENANCE'), None)
    if not offering or DomainToolName.START_SERVICE not in request.toolPolicy.allowedTools or request.toolPolicy.maxToolCalls < 1:
        return response('No puedo abrir otro folio de mantenimiento en este momento. Contacta a recepción para solicitar seguimiento.',
                        'I cannot open another maintenance request right now. Please contact the front desk for follow-up.', source)
    if confirmed:
        return dict(disposition='TOOL_CALLS_REQUIRED', messages=[], updated_summary=summary(None), tool_calls=[{
            'toolCallId': str(uuid4()), 'toolName': 'START_SERVICE',
            'targetOperationId': None, 'targetConversationTaskId': None, 'confidence': 1,
            'evidenceMessageIds': [str(latest.messageId)],
            'arguments': {'offeringCode': 'MAINTENANCE', 'input': deepcopy(source['input']),
                          'recurrenceSourceOperationId': source['operationId'],
                          'recurrenceConfirmationToken': pending['token'],
                          'guestConfirmationEvidenceMessageId': str(latest.messageId)},
        }])
    token = uuid4().hex
    prefix = f'maintenance-recurrence:{source["operationId"]}:{token}'
    explanation = ('se cerró después de tu confirmación' if source['detailedStatus'] == 'RESOLUTION_CONFIRMED'
                   else 'se cerró sin tu confirmación de solución' if source['detailedStatus'] == 'CLOSED_NO_GUEST_CONFIRMATION'
                   else 'ya está cerrado')
    explanation_en = ('was closed after your confirmation' if source['detailedStatus'] == 'RESOLUTION_CONFIRMED'
                      else 'was closed without your confirmation of resolution' if source['detailedStatus'] == 'CLOSED_NO_GUEST_CONFIRMATION'
                      else 'is already closed')
    es = f'El folio {ref} {explanation}. Problema reportado: {issue}. ¿Quieres abrir un nuevo folio por este mismo problema? Conservaré los datos y la referencia al anterior.'
    en = f'Request {ref} {explanation_en}. Reported issue: {issue}. Would you like to open a new request for this same issue? I will retain its details and link the previous reference.'
    interaction = {'type': 'BUTTONS', 'title': 'Seguimiento' if spanish else 'Follow-up',
                   'body': (es if spanish else en)[:1024], 'options': [
                       {'id': prefix + ':CONFIRM', 'label': 'Sí, abrir folio' if spanish else 'Open new request'},
                       {'id': prefix + ':CANCEL', 'label': 'Ahora no' if spanish else 'Not now'}]}
    return response(es, en, source, summary({'sourceOperationId': source['operationId'], 'token': token,
                                           'source': source}), interaction)
