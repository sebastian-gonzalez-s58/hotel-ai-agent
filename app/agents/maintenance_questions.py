"""Staff questions collect evidence on an existing folio; they never resolve it."""
import re
from uuid import uuid4

from app.core.errors import AgentModelError
from app.schemas.v2_turns import DomainToolName

TASK_TYPE = 'MAINTENANCE_GUEST_QUESTION'


def questions(request):
    return [task for operation in request.activeOperations for task in operation.pendingConversationTasks
            if operation.offeringCode == 'MAINTENANCE' and task.taskType == TASK_TYPE]


def select_question(request, message):
    candidates = questions(request)
    refs = re.findall(r'\bREQ-[A-Z0-9-]+', message.text.upper())
    if request.trigger.conversationTaskId:
        # An explicit trigger is authoritative. Do not let a stale task id
        # complete another guest's question.
        candidates = [task for task in candidates if task.conversationTaskId == request.trigger.conversationTaskId]
    elif request.trigger.operationId:
        candidates = [task for task in candidates if task.operationId == request.trigger.operationId]
    elif message.conversationTaskIds:
        selected = [task for task in candidates if task.conversationTaskId in message.conversationTaskIds]
        # A stale correlation from another service must not make the only
        # pending maintenance question ambiguous.
        if selected:
            candidates = selected
        elif len(candidates) != 1:
            return None
    elif message.operationIds:
        selected = [task for task in candidates if task.operationId in message.operationIds]
        if selected:
            candidates = selected
        elif len(candidates) != 1:
            return None
    elif refs:
        ids = {op.operationId for op in request.activeOperations if (op.referenceCode or '').upper() in refs}
        candidates = [task for task in candidates if task.operationId in ids]
    elif len(candidates) != 1:
        # A folio is only necessary when several maintenance questions are open.
        # Other service tasks, such as a pending spa alternative, are unrelated.
        return None
    return candidates[0] if len(candidates) == 1 else None


def plan_question(request, latest, scope=None):
    spanish = request.guest.preferredLanguage.lower().startswith('es')
    common = {'updated_summary': request.conversation.summary}
    for result in request.previousToolResults:
        value = result.result
        if (result.status == 'SUCCEEDED' and result.toolName == 'COMPLETE_CONVERSATION_TASK'
                and isinstance(value, dict) and value.get('taskType') == TASK_TYPE):
            return dict(common, disposition='RESPONSE_READY', messages=[{
                'purpose': 'ANSWER', 'language': request.guest.preferredLanguage,
                'text': ('Compartí tu respuesta con el equipo de mantenimiento para que continúe la atención.'
                         if spanish else 'I shared your reply with the maintenance team so they can continue assisting you.'),
                'operationIds': [value['operationId']], 'conversationTaskIds': [value['conversationTaskId']],
            }])
    if (request.previousToolResults or request.trigger.type != 'INBOUND_MESSAGE' or latest is None
            or latest.interactionReplyId or not questions(request) or scope is None
            or scope.kind != 'CONTEXT_REPLY' or scope.separateRequest or scope.languageChangeOnly
            or scope.containsUnrelatedTopic or scope.offeringCode not in {None, 'MAINTENANCE'}
            or scope.confidence < .8 or not latest.text.strip()):
        return None
    if DomainToolName.COMPLETE_CONVERSATION_TASK not in request.toolPolicy.allowedTools:
        return None
    task = select_question(request, latest)
    if task is None:
        refs = [op.referenceCode for op in request.activeOperations
                if any(t.taskType == TASK_TYPE for t in op.pendingConversationTasks) and op.referenceCode]
        text = ('¿A qué pregunta de mantenimiento respondes? Indica el folio junto con tu respuesta: '
                if spanish else 'Which maintenance question are you answering? Include its reference with your reply: ')
        return dict(common, disposition='RESPONSE_READY', messages=[{
            'purpose': 'CLARIFICATION', 'text': text + ', '.join(refs),
            'language': request.guest.preferredLanguage, 'operationIds': [], 'conversationTaskIds': [],
        }])
    if latest.createdAt < task.createdAt:
        return dict(common, disposition='RESPONSE_READY', messages=[{
            'purpose': 'CLARIFICATION', 'text': task.context.get('question', ''),
            'language': request.guest.preferredLanguage, 'operationIds': [str(task.operationId)], 'conversationTaskIds': [],
        }])
    return dict(common, disposition='TOOL_CALLS_REQUIRED', messages=[], tool_calls=[{
        'toolCallId': str(uuid4()), 'toolName': 'COMPLETE_CONVERSATION_TASK',
        'targetOperationId': str(task.operationId), 'targetConversationTaskId': str(task.conversationTaskId),
        'arguments': {'conversationTaskId': str(task.conversationTaskId), 'expectedVersion': task.version,
                      'result': {'maintenanceGuestAnswer': latest.text}},
        'confidence': 1.0, 'evidenceMessageIds': [str(latest.messageId)],
    }])


def validate_question_call(request, call, task, latest):
    if task is None or task.taskType != TASK_TYPE: return
    if call.toolName != DomainToolName.COMPLETE_CONVERSATION_TASK:
        raise AgentModelError('Maintenance questions must preserve a complete verbatim guest reply')
    if (latest is None or latest.interactionReplyId or latest.actor != 'GUEST'
            or latest.createdAt < task.createdAt or select_question(request, latest) != task
            or call.evidenceMessageIds != [latest.messageId]
            or call.arguments.get('result') != {'maintenanceGuestAnswer': latest.text}):
        raise AgentModelError('Maintenance answer must match the current guest reply and exactly one question')
