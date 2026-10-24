"""Resolve kitchen-selected option changes without recapturing the order."""
from copy import deepcopy
from decimal import Decimal
from uuid import uuid4

from app.core.errors import AgentModelError
from app.services.catalog_orders import catalog_for, pending_choice, apply_choice


def kitchen_context(task):
    context = task.context.get('kitchenChange')
    return context if isinstance(context, dict) and isinstance(context.get('baseInput', {}).get('items'), list) else None


def initial_pending(context):
    groups = context.get('groups', []) if context else []
    return deepcopy(groups[0]) if context and context.get('optionsOnly') and groups else None


def refresh_pending(offering, pending):
    """Discard removed/changed choices before accepting even an old interactive reply."""
    pending = deepcopy(pending)
    item = next((i for i in (catalog_for(offering, 'items') or {}).get('options', []) if i['id'] == pending['itemId']), None)
    group = next((g for g in (item or {}).get('optionGroups', []) if g['id'] == pending['groupId']), None)
    prices = [p for p in (item or {}).get('prices', []) if p.get('priceType') == 'BASE' and p.get('active', True)]
    choices = []
    allowed = set(pending['groupOptionIds']) - set(pending['blockedOptionIds'])
    if group and len(prices) == 1:
        for option in group.get('options', []):
            if option['id'] in allowed and option.get('available', True):
                choices.append({'id': option['id'], 'label': option['name'], 'priceAdjustment': option.get('priceAdjustment') or 0, 'currency': prices[0]['currency']})
        if pending.get('optionalOffer'):
            choices.append({'id': '__skip__', 'label': 'Sin este complemento', 'action': 'SKIP'})
    def signature(rows):
        return [(c['id'], c['label'], Decimal(str(c.get('priceAdjustment') or 0)), c.get('currency', '')) for c in rows]
    if signature(choices) != signature(pending['choices']):
        pending.update(token=str(uuid4()), page=0)
    pending['choices'] = choices
    return pending


def advance_options(offering, context, draft, message):
    items = deepcopy(draft['items'])
    step = draft.get('kitchenStep', 0)
    groups = context['groups']
    if step >= len(groups):
        return items, None
    pending = refresh_pending(offering, draft.get('pending') or groups[step])
    choice = pending_choice(pending, message)
    if choice:
        if choice.get('action') == 'PAGE':
            pending['page'] = choice['page']
        else:
            index = pending['itemIndex']
            before_notes = deepcopy(items[index].get('modifications', []))
            items[index] = apply_choice(items[index], pending, choice)
            # Kitchen option edits cannot drop dietary/preparation instructions.
            items[index]['modifications'] = before_notes
            items[index]['catalogSelection']['sourceModifications'] = before_notes
            step += 1
            draft['kitchenStep'] = step
            pending = refresh_pending(offering, groups[step]) if step < len(groups) else None
    return items, pending


def validate_scope(context, items):
    if not context or not context.get('optionsOnly'):
        return
    before = context['baseInput']['items']
    if len(items) != len(before):
        raise AgentModelError('A kitchen option change cannot replace the entire order')
    for index, (old, new) in enumerate(zip(before, items)):
        if any(old.get(k) != new.get(k) for k in ('name', 'quantity', 'modifications')) or old['catalogSelection']['itemId'] != new['catalogSelection']['itemId']:
            raise AgentModelError('Unrelated order fields changed')
        groups = [g for g in context['groups'] if g['itemIndex'] == index]
        allowed = {i for g in groups for i in g['groupOptionIds']}
        blocked = {i for g in groups for i in g['blockedOptionIds']}
        old_ids, new_ids = set(old['catalogSelection']['optionIds']), set(new['catalogSelection']['optionIds'])
        if new_ids & blocked or not (old_ids - new_ids) <= blocked or not (new_ids - old_ids) <= allowed:
            raise AgentModelError('Kitchen option change is unresolved or changed an unrelated option')
