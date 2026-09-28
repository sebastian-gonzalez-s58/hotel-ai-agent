"""Explain rejected room-service interactions using authoritative operation snapshots.

This module never authorizes a mutation or selects the newest order by recency.
"""
import re


def room_operations(request):
    operations = {str(o.operationId): o for o in request.recentOperations}
    operations.update({str(o.operationId): o for o in request.activeOperations})
    return {key: o for key, o in operations.items() if o.offeringCode == 'ROOM_SERVICE'}


def resolve_order(request, latest, *, button):
    operations = room_operations(request)
    if button:
        # Only the backend's persisted successful start may associate an old menu.
        context = request.trigger.eventPayload.get('roomServiceButtonContext', {})
        operation_id = context.get('operationId') if isinstance(context, dict) else None
        return operations.get(operation_id)
    references = [o for o in operations.values() if o.referenceCode and re.search(
        r'(?<![\w-])' + re.escape(o.referenceCode) + r'(?![\w-])', latest.text, re.IGNORECASE)]
    if len(references) == 1:
        return references[0]
    if references:
        return None
    # An unmatched explicit folio must never fall back to a different order.
    if re.search(r'\bREQ-[\w-]+', latest.text, re.IGNORECASE):
        return None
    if len(latest.operationIds) == 1:
        return operations.get(str(latest.operationIds[0]))
    return next(iter(operations.values())) if len(operations) == 1 else None


def status_message(request, operation, action, *, button=False):
    es = request.guest.preferredLanguage.lower().startswith('es')
    verb = {'CHANGE': ('modificarlo', 'change it'), 'CANCEL': ('cancelarlo', 'cancel it'),
            'CONFIRM': ('confirmarlo de nuevo', 'confirm it again')}.get(action, ('cambiarlo', 'change it'))[0 if es else 1]
    if operation is None:
        text = ('No puedo identificar a qué pedido corresponde ese botón. No hice cambios. Indícame el folio del pedido que quieres consultar.'
                if button and es else
                'I cannot identify which order that button belongs to. I made no changes. Please provide the order reference.'
                if button else
                'No puedo identificar con certeza qué pedido quieres modificar o cancelar. No hice cambios. Indícame su folio.'
                if es else 'I cannot identify which order you want to change or cancel. I made no changes. Please provide its reference.')
        ids = []
    else:
        label = (f'Tu pedido de room service ({operation.referenceCode})' if es else
                 f'Your room-service order ({operation.referenceCode})') if operation.referenceCode else (
                     'Tu pedido de room service' if es else 'Your room-service order')
        terminal = operation.lifecycle in {'COMPLETED', 'CANCELLED', 'FAILED'}
        if operation.lifecycle == 'COMPLETED' and operation.detailedStatus == 'DELIVERED':
            text = (f'{label} ya fue entregado, por lo que no es posible {verb}.' if es else
                    f'{label} has already been delivered, so it is no longer possible to {verb}.')
        elif operation.lifecycle == 'CANCELLED':
            reason = {'CANCELLED_BY_KITCHEN': (' por cocina', ' by the kitchen'),
                      'CANCELLED_BY_GUEST': (' a tu solicitud', ' at your request'),
                      'CANCELLED_GUEST_TIMEOUT': (' al vencer el plazo de respuesta', ' when the response period expired')}
            suffix = reason.get(operation.detailedStatus, ('', ''))[0 if es else 1]
            text = (f'{label} ya estaba cancelado{suffix}. No realicé otra cancelación ni abrí un pedido nuevo.' if es else
                    f'{label} was already cancelled{suffix}. I did not cancel it again or open a new order.')
        elif operation.lifecycle == 'FAILED':
            text = (f'{label} no pudo completarse y está cerrado. No hice cambios ni creé otro pedido.' if es else
                    f'{label} could not be completed and is closed. I made no changes and did not create another order.')
        elif terminal:
            text = (f'{label} está cerrado y no permite {verb}. No hice cambios.' if es else
                    f'{label} is closed and cannot be changed. I made no changes.')
        else:
            state = {
                'KITCHEN_REVIEW': ('está en revisión de cocina', 'is awaiting kitchen review'),
                'AWAITING_DELIVERY': ('ya fue aceptado por cocina y está pendiente de entrega', 'has been accepted by the kitchen and is awaiting delivery'),
                'CREATED': ('ya fue registrado', 'has already been registered'),
            }.get(operation.detailedStatus, ('sigue en curso', 'is still in progress'))[0 if es else 1]
            if operation.pendingConversationTasks:
                explanation = ('Usa la solicitud de respuesta actual de cocina; este botón pertenece a la confirmación anterior.' if es else
                               'Use the current kitchen response request; this button belongs to the earlier confirmation.')
            elif operation.availableActions:
                explanation = ('Este botón pertenece a la confirmación anterior. Indica por escrito qué quieres hacer para revisar las acciones disponibles.' if es else
                               'This button belongs to the earlier confirmation. Tell me what you want to do so I can check the available actions.')
            else:
                explanation = (f'En esta etapa no puedo {verb} desde este chat. No hice cambios.' if es else
                               f'At this stage I cannot {verb} through this chat. I made no changes.')
            text = f'{label} {state}. {explanation}'
        ids = [str(operation.operationId)]
    return {'purpose': 'CLARIFICATION', 'text': text, 'language': request.guest.preferredLanguage,
            'operationIds': ids, 'conversationTaskIds': []}


def existing_order_message(request, latest, scope, capture_state):
    action = scope.existingOrderAction
    if (action not in {'CHANGE', 'CANCEL'} or scope.existingOrderEvidence != latest.text
            or scope.existingOrderConfidence < .9 or scope.separateRequest or scope.containsUnrelatedTopic):
        return None
    operation = resolve_order(request, latest, button=False)
    explicit_reference = operation and operation.referenceCode and operation.referenceCode.casefold() in latest.text.casefold()
    # Preserve draft editing and kitchen-requested replacement/cancellation workflows.
    if capture_state.get('pendingOffering') and not explicit_reference:
        return None
    if operation and operation.lifecycle not in {'COMPLETED', 'CANCELLED', 'FAILED'}:
        if operation.pendingConversationTasks or operation.availableActions:
            return None
    return status_message(request, operation, action)
