"""Fresh confirmations for advertised reservation actions; old menus only select a source."""
import re
from uuid import uuid4
from app.agents.spa_turns import summary_state, _reply, _fields_text
from app.schemas.v2_turns import DomainToolName

INTENT = re.compile(r'^reservation:([a-fA-F0-9-]{36}):(CHANGE|CANCEL)$')
CONFIRM = re.compile(r'^reservation-confirm:([a-fA-F0-9-]{36}):(\d+):([a-f0-9]{32}):(CHANGE|CANCEL|DECLINE)$')


def plan_reservation_action(request, latest, scope=None):
    state = summary_state(request.conversation.summary)
    for result in request.previousToolResults:
        if result.toolName == 'EXECUTE_SERVICE_ACTION' and (result.result or {}).get('executedActionCode', '').startswith('RESERVATION_'):
            state.pop('reservationAction', None)
            return _reply(request, state)  # BPM sends the durable outcome, after the transition.
    if request.previousToolResults or request.trigger.type != 'INBOUND_MESSAGE' or latest is None:
        return None
    es = request.guest.preferredLanguage.lower().startswith('es')
    def say(spanish, english, options=None): return _reply(request, state, spanish if es else english, options=options)
    reply = latest.interactionReplyId or ''
    intent, confirm = INTENT.fullmatch(reply), CONFIRM.fullmatch(reply)
    operations = {str(o.operationId): o for o in [*request.recentOperations, *request.activeOperations]
                  if o.offeringCode == 'SPA' or any(a.actionCode.startswith('RESERVATION_') for a in o.availableActions)}
    source_id = (request.trigger.eventPayload.get('reservationButtonContext') or {}).get('operationId')
    action = intent[2] if intent else confirm[4] if confirm else None
    if reply.startswith(('spa:', 'spa-draft:')) and source_id:
        # Current BPM tasks retain their dedicated alternative/change-details handler.
        if any(str(t.conversationTaskId) == reply.split(':')[1] for o in request.activeOperations for t in o.pendingConversationTasks):
            return None
        action = reply.split(':')[2]
    elif not reply and scope:
        if scope.offeringCode not in {'SPA', *(o.offeringCode for o in operations.values())}:
            return None
        if (scope.separateRequest or scope.kind not in {'SERVICE_REQUEST','CONTEXT_REPLY','STATUS_REQUEST'}
                or scope.replyAction not in {'CHANGE','CANCEL'} or scope.replyActionConfidence < .9
                or scope.replyActionEvidence != latest.text): return None
        if scope.kind != 'STATUS_REQUEST' and (state.get('spaDraft') or state.get('pendingOffering')
                or any(o.pendingConversationTasks for o in operations.values())): return None
        candidates = [o for o in operations.values() if o.referenceCode and re.search(
            r'(?<![\w-])'+re.escape(o.referenceCode)+r'(?![\w-])',latest.text,re.I)]
        if not candidates and not re.search(r'\bREQ-[\w-]+',latest.text,re.I):
            candidates = [o for o in operations.values() if scope.offeringCode == o.offeringCode]
        action = scope.replyAction
        if len(candidates) == 1: source_id = str(candidates[0].operationId)
        elif candidates:
            options=[{'id':f'reservation:{o.operationId}:{action}','label':o.referenceCode or o.offeringCode} for o in candidates[:10]]
            return say('¿Qué reservación quieres gestionar? Selecciona su folio.', 'Which reservation would you like to manage? Select its reference.',options)
    elif not intent and not confirm:
        if reply.startswith(('reservation:', 'reservation-confirm:')):
            return say('No pude identificar ese botón de reservación. Indícame el folio.', 'I could not identify that reservation button. Please provide its reference.')
        return None
    operation = operations.get(source_id)
    if (intent or confirm) and source_id != (intent or confirm)[1]:
        operation = None
    if not operation:
        return say('No pude identificar esa reservación. Indícame su folio para revisar el estado.', 'I could not identify that reservation. Please provide its reference so I can check its status.')
    ref = operation.referenceCode or str(operation.operationId)
    if operation.lifecycle == 'CANCELLED':
        return say(f'La reservación {ref} ya está cancelada. Ese botón no la reactiva; puedes solicitar una reservación nueva.',
                   f'Reservation {ref} is already cancelled. That button does not reactivate it; you can request a new booking.')
    if operation.detailedStatus == 'SERVICE_COMPLETED':
        return say(f'El servicio de la reservación {ref} ya se realizó. Para otra fecha o tratamiento, puedes solicitar una reservación nueva.',
                   f'The service for reservation {ref} has already been completed. You can request a new booking for another date or treatment.')
    if operation.detailedStatus == 'ACTION_RESERVATION_CANCEL_REQUESTED':
        return say(f'La solicitud de cancelación de la reservación {ref} ya se está procesando. Recibirás la confirmación; no envié otra solicitud.',
                   f'The cancellation request for reservation {ref} is already being processed. You will receive confirmation; I did not submit another request.')
    if operation.detailedStatus == 'ACTION_RESERVATION_CHANGE_REQUESTED':
        return say(f'Ya solicitaste cambiar la reservación {ref}. Se está preparando el paso para recoger los nuevos datos; no envié otra solicitud.',
                   f'You already requested a change to reservation {ref}. The next step is being prepared to collect your new details; I did not submit another request.')
    if confirm and confirm[4] == 'DECLINE':
        pending=state.get('reservationAction') or {}
        if pending.get('operationId') == source_id and pending.get('token') == confirm[3]:
            state.pop('reservationAction', None)
        return say('De acuerdo. No envié cambios ni cancelé la reservación.', 'Understood. I did not request changes or cancel the reservation.')
    available = next((a for a in operation.availableActions if a.actionCode == 'RESERVATION_'+str(action)),None)
    if not available or operation.lifecycle in {'COMPLETED','CANCELLED','FAILED'}:
        if operation.pendingConversationTasks:
            return say(f'La reservación {ref} espera tu respuesta en el menú de seguimiento actual del SPA. Ese botón pertenece a una revisión anterior; no modifiqué la reservación.',
                       f'Reservation {ref} is waiting for your response in the current SPA follow-up menu. That button belongs to an earlier review; I did not modify the reservation.')
        detail = 'está confirmada' if operation.detailedStatus == 'RESERVATION_CONFIRMED' else 'está en revisión' if operation.lifecycle == 'WAITING_FOR_STAFF' else 'tiene otro estado'
        detail_en = 'is confirmed' if operation.detailedStatus == 'RESERVATION_CONFIRMED' else 'is under review' if operation.lifecycle == 'WAITING_FOR_STAFF' else 'has a different status'
        return say(f'La reservación {ref} {detail}. Ese botón corresponde a un paso anterior. No hice cambios; consulta las opciones actuales de la reservación.',
                   f'Reservation {ref} {detail_en}. That button belongs to an earlier step. I made no changes; please use the current reservation options.')
    pending = state.get('reservationAction') or {}
    if confirm:
        if (pending != {'operationId':source_id,'version':int(confirm[2]),'token':confirm[3],'action':action}
                or operation.version != int(confirm[2])):
            return say(f'El estado de la reservación {ref} cambió o esa confirmación ya fue utilizada. Solicita de nuevo la acción para revisar los datos actuales.',
                       f'Reservation {ref} changed or that confirmation was already used. Please request the action again to review its current details.')
        if DomainToolName.EXECUTE_SERVICE_ACTION not in request.toolPolicy.allowedTools or request.toolPolicy.maxToolCalls<1:
            return say('No puedo enviar esta acción en este momento.', 'I cannot submit this action right now.')
        return _reply(request,state,calls=[{'toolName':'EXECUTE_SERVICE_ACTION','targetOperationId':source_id,
            'targetConversationTaskId':None,'confidence':1,'evidenceMessageIds':[str(latest.messageId)],
            'arguments':{'operationId':source_id,'expectedVersion':operation.version,'actionCode':available.actionCode,
                         'input':{},'evidenceMessageIds':[str(latest.messageId)]}}])
    token=uuid4().hex
    state['reservationAction']={'operationId':source_id,'version':operation.version,'token':token,'action':action}
    prefix=f'reservation-confirm:{source_id}:{operation.version}:{token}'
    text = (f'Reservación {ref}:\n' if es else f'Reservation {ref}:\n') + _fields_text(request,operation.input)
    text += ('\n¿Confirmas que deseas cancelar esta reservación?' if es else '\nDo you confirm that you want to cancel this reservation?') if action=='CANCEL' else (
        '\n¿Quieres solicitar un cambio? Recogeré los nuevos datos para aprobación del equipo.' if es else
        '\nWould you like to request a change? I will collect the new details for staff approval.')
    if action == 'CHANGE' and (operation.detailedStatus == 'RESERVATION_CONFIRMED' or 'pendingReservationChange' in operation.input):
        text += (' La reserva confirmada se conserva mientras se revisa.' if es else
                 ' Your confirmed reservation is retained during review.')
    return _reply(request,state,text,options=[{'id':prefix+':'+action,'label':('Sí, cancelar' if es else 'Yes, cancel') if action=='CANCEL' else ('Solicitar cambio' if es else 'Request change')},
                                            {'id':prefix+':DECLINE','label':'Ahora no' if es else 'Not now'}])
