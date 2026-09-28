import json
from typing import Any


INTERACTION_JSON_GUIDANCE = """
You may include an optional "interaction" object when predefined WhatsApp actions would make the next reply easier.
Use this shape:
{
  "interaction": {
    "type": "BUTTONS | LIST",
    "title": "short title or null",
    "body": "short body text or null",
    "buttonText": "short list button text or null",
    "actions": [
      {"id": "STABLE_ACTION_ID", "label": "guest-facing label"}
    ]
  }
}
Interaction rules:
- Use BUTTONS only for 2 or 3 choices.
- Use LIST for 4 to 10 choices.
- Action ids must be stable uppercase identifiers in English.
- Labels must match the guest's language.
- The message must still make sense without the interaction.
- Omit interaction or use null when free text is better.
"""


def extraction_prompt(history_text: str, known_context: dict[str, Any]) -> str:
    return f"""
You are a hotel WhatsApp assistant extraction agent.

Your job is to read the conversation and extract the guest's current request as structured JSON.

If the latest guest message is a short answer to the assistant's previous clarification question, combine it with the earlier request. For example, if the assistant asks for a room number and the guest replies "100", set roomNumber to "100" and keep the original intent and issue or order details from the conversation.

Return ONLY valid JSON with this exact structure:

{{
  "intent": "FAQ | ROOM_SERVICE | MAINTENANCE | HOUSEKEEPING | SPA | RESTAURANT_RESERVATION | COMPLAINT | HUMAN_REVIEW | UNKNOWN",
  "confidence": 0.0,
  "containsEmergency": false,
  "roomNumber": null,
  "language": "en",
  "missingFields": [],
  "extractedEntities": {{}},
  "requestComplete": false
}}

Intent rules:
- FAQ: hotel information, check-in/out time, wifi, amenities, pool, gym, breakfast, parking, nearby places
- ROOM_SERVICE: food, drinks, minibar, ordering items from kitchen/bar
- MAINTENANCE: broken AC, water leak, electricity, TV not working, plumbing, lock problem
- HOUSEKEEPING: towels, cleaning, toiletries, extra pillows, blankets
- SPA: spa services, massages, wellness treatments, spa menu, spa reservations
- RESTAURANT_RESERVATION: booking or reserving a table at the hotel restaurant, including date or time requests for restaurant reservations
- COMPLAINT: dissatisfaction, noise, bad service, dirty room, serious negative experience
- HUMAN_REVIEW: guest explicitly asks for staff/human/manager or the request is sensitive
- UNKNOWN: unclear request

Emergency rules:
Set containsEmergency=true for:
- medical emergencies
- fire/smoke
- gas smell
- violence/threats
- serious security incident
- flooding/water leak causing immediate danger

Completeness rules:
For ROOM_SERVICE, required fields are:
- roomNumber
- items
- deliveryLocation

For ROOM_SERVICE, extractedEntities must use this shape when possible:
{{
  "items": [
    {{
      "quantity": 1,
      "name": "burger",
      "modifiers": null
    }}
  ],
  "roomNumber": "402",
  "deliveryLocation": "ROOM | DOCK_1 | DOCK_2 | POOL_1 | POOL_2",
  "specialInstructions": null
}}
Use integer quantities. If the guest gives an item without a quantity, infer quantity 1.
If the guest says delivery is for their room, use deliveryLocation="ROOM".

For MAINTENANCE, required fields are:
- roomNumber
- issueDescription

For HOUSEKEEPING, required fields are:
- roomNumber
- requestedItemsOrService

For RESTAURANT_RESERVATION, required fields are:
- roomNumber
- reservationDate
- reservationTime

For RESTAURANT_RESERVATION, extractedEntities must use this shape when possible:
{{
  "reservationDate": "2026-06-01",
  "reservationTime": "19:00",
  "partySize": null,
  "restaurantName": null,
  "specialRequests": null
}}
If the guest gives a natural-language date or time, preserve it if you cannot confidently normalize it.

For COMPLAINT, required fields are:
- roomNumber if the complaint is room-specific
- complaintDescription

For FAQ, usually no required fields unless the question is ambiguous.

If required information is missing, add field names to missingFields and set requestComplete=false.
If enough information is present, missingFields=[] and requestComplete=true.

Interactive reply id rules:
- ROOM_SERVICE means the guest selected room service
- SPA means the guest selected SPA
- MAINTENANCE means the guest selected maintenance
- FRONT_DESK means the guest selected front desk or human help
- FAQ means the guest selected general questions or hotel information
- ROOM, DOCK_1, DOCK_2, POOL_1, POOL_2 are room service delivery locations

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""


def clarification_prompt(extraction: dict[str, Any], history_text: str) -> str:
    return f"""
You are a polite hotel WhatsApp assistant.

The guest's request is not complete yet.
Write ONE short clarification question to get the missing information.

Rules:
- Be natural and concise
- Do not mention JSON, intent, confidence, or internal fields
- Ask only for the most important missing information
- If roomNumber is missing, ask for the room number
- If room service items are missing, ask what they would like to order
- If restaurant reservation date is missing, ask what date they would like
- If restaurant reservation time is missing, ask what time they would like
- If maintenance issue is unclear, ask what is not working
- If the intent is UNKNOWN or the guest appears to be greeting the assistant, offer a main service menu
- If room service deliveryLocation is missing, ask where the order should be delivered
- Return ONLY valid JSON:

{{
  "message": "your clarification question",
  "interaction": null
}}

Recommended interactions:
- Main menu: LIST with ROOM_SERVICE, SPA, MAINTENANCE, FRONT_DESK, FAQ
- Room service delivery location: LIST with ROOM, DOCK_1, DOCK_2, POOL_1, POOL_2

{INTERACTION_JSON_GUIDANCE}

Extraction:
{json.dumps(extraction, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""


def room_service_confirmation_prompt(
    extraction: dict[str, Any],
    history_text: str,
    menu_knowledge: dict[str, Any] | None = None,
) -> str:
    return f"""
You are a hotel room service assistant.

Normalize the room service request into a pending order and write a short confirmation message.

Return ONLY valid JSON with this exact structure:

{{
  "message": "I have your room service order as: 1 burger and 2 cokes. Is that correct, or would you like to change anything?",
  "pendingOrder": {{
    "roomNumber": null,
    "items": [
      {{
        "quantity": 1,
        "name": "burger",
        "modifiers": null
      }}
    ],
    "specialInstructions": null,
    "deliveryLocation": null
  }},
  "missingFields": [],
  "requestComplete": true,
  "interaction": {{
    "type": "BUTTONS",
    "title": null,
    "body": null,
    "buttonText": null,
    "actions": [
      {{"id": "CONFIRM_ORDER", "label": "Confirm"}},
      {{"id": "CHANGE_ORDER", "label": "Change"}},
      {{"id": "CANCEL_ORDER", "label": "Cancel"}}
    ]
  }}
}}

Rules:
- List every food or drink item in the message
- Use the menu knowledge to canonicalize item names when possible
- Keep item names natural and concise
- Use integer quantities; if no quantity is stated, use 1
- Include roomNumber if present in the extraction
- Include deliveryLocation if present in the extraction
- Ask whether the order is correct or if the guest wants to change anything
- Include confirmation actions with labels in the guest's language
- Do not say the order has been placed yet
- If the guest asks for an item that is clearly unavailable or not present in the menu, politely mention that it may need staff confirmation and keep it in pendingOrder with the closest natural name
- The only required fields are items and deliveryLocation
- requestComplete is true only when pendingOrder has at least one valid item and deliveryLocation is present
- missingFields contains exactly "items" and/or "deliveryLocation" when either is missing
- If requestComplete is false, ask only for the missing information and set interaction to null
- If requestComplete is true, list the complete order once and include the confirmation actions

{INTERACTION_JSON_GUIDANCE}

Extraction:
{json.dumps(extraction, ensure_ascii=False, indent=2)}

Menu knowledge:
{json.dumps(menu_knowledge or {}, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""


def room_service_delivery_location_prompt(
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return f"""
You are opening the room service flow for a hotel guest.

Ask only where the order should be delivered. Do not ask what they want to order
yet and do not ask for their room number.

Return ONLY valid JSON with this exact structure:

{{
  "message": "short guest-facing question",
  "interaction": {{
    "type": "LIST",
    "title": "short title",
    "body": "short body",
    "buttonText": "short button label",
    "actions": [
      {{"id": "ROOM", "label": "guest-language label for their room"}},
      {{"id": "DOCK_1", "label": "guest-language label for dock 1"}},
      {{"id": "DOCK_2", "label": "guest-language label for dock 2"}},
      {{"id": "POOL_1", "label": "guest-language label for pool 1"}},
      {{"id": "POOL_2", "label": "guest-language label for pool 2"}}
    ]
  }}
}}

Rules:
- Match the guest's language when clear; otherwise use the preferred language in knownContext
- Include all five actions exactly once and preserve their stable ids
- Keep the message concise

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""


def room_service_menu_prompt(
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "Continue the room service flow. The message MUST include this exact menu URL: "
            "https://hotelcristalino.menudigitalonline.com/ . "
            "Tell the guest that the link contains the digital food and drink menu, then ask "
            "which products and quantities they want to order. Do not ask for their delivery "
            "location or room number and do not ask for confirmation yet. Do not include an interaction."
        ),
        history_text=history_text,
        known_context=known_context,
    )


def room_service_order_accepted_prompt(
    guest_message: str | None,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "Confirm that hotel staff has accepted the room service order and that it will be "
            "delivered as soon as possible. This is the final message: do not ask another question "
            "and do not include buttons or a list."
        ),
        history_text=history_text,
        known_context=known_context,
        guest_message=guest_message,
    )


def room_service_kitchen_rejected_prompt(
    guest_message: str | None,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "Tell the guest that kitchen could not accept the current room service order. "
            "Use roomServiceKitchenRejectionReason from known context when present, but phrase "
            "it politely and do not expose internal notes. Ask the guest to either change the "
            "order or cancel it. Include exactly two BUTTONS actions: "
            "CHANGE_ORDER for changing the order and CANCEL_ORDER for cancelling it. "
            "Translate the message and button labels to the guest's language. Do not offer an "
            "option to reconfirm the rejected order."
        ),
        history_text=history_text,
        known_context=known_context,
        guest_message=guest_message,
    )


def room_service_confirmation_evaluation_prompt(
    guest_message: str,
    pending_order: dict[str, Any],
    history_text: str,
) -> str:
    return f"""
You are a hotel room service assistant evaluating a guest reply to an order confirmation.

Classify the reply and update the pending room service order when needed.

Return ONLY valid JSON with this exact structure:

{{
  "confirmationAction": "CONFIRMED | CHANGE_REQUESTED | CANCELLED | UNCLEAR",
  "updatedOrder": {{
    "roomNumber": null,
    "items": [
      {{
        "quantity": 1,
        "name": "burger",
        "modifiers": null
      }}
    ],
    "specialInstructions": null,
    "deliveryLocation": null
  }},
  "message": "short WhatsApp message",
  "interaction": null
}}

Decision rules:
- CONFIRMED: the guest clearly accepts the pending order, such as yes, correct, looks good, go ahead, or CONFIRM_ORDER
- CHANGE_REQUESTED: the guest changes quantities, adds/removes items, changes instructions, changes the room, or selects CHANGE_ORDER
- CANCELLED: the guest clearly cancels the order or selects CANCEL_ORDER
- UNCLEAR: the reply does not clearly confirm, cancel, or change the order

Message rules:
- For CHANGE_REQUESTED with concrete changes, restate the full updated order once and ask if it is correct
- If the guest only selects CHANGE_ORDER without describing a change, ask what they want to add, remove, or modify and do not restate the unchanged order
- For UNCLEAR, ask the guest to confirm, change, or cancel the order
- For CANCELLED, confirm that the order was cancelled
- For CONFIRMED, write a brief acknowledgement; the BPMN process will place the order next
- Include confirmation buttons after applying concrete changes or for UNCLEAR
- When merely asking what should be changed, set interaction to null

{INTERACTION_JSON_GUIDANCE}

Pending order:
{json.dumps(pending_order, ensure_ascii=False, indent=2)}

Guest reply:
{guest_message}

Conversation:
{history_text}
"""


def faq_prompt(guest_message: str, history_text: str, known_context: dict[str, Any]) -> str:
    return f"""
You are a hotel WhatsApp assistant answering FAQ questions.

Answer the guest in a helpful, concise, hotel-friendly tone.

Rules:
- If the answer is available in knownContext, use it
- If not available in knownContext or faqKnowledge, do not invent an answer
- Do not invent exact prices, opening hours, or policies unless provided
- Prefer faqKnowledge entries when they answer the question
- Return ONLY valid JSON:

{{
  "message": "answer to guest, or a short internal fallback message if human help is needed",
  "answered": true,
  "needsHumanAnswer": false,
  "category": "short category such as horarios, politicas, servicios, general",
  "interaction": null
}}

Set answered=true and needsHumanAnswer=false only when the answer is supported by knownContext or faqKnowledge.
Set answered=false and needsHumanAnswer=true when the hotel knowledge does not contain a reliable answer.

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Current guest message:
{guest_message}

Conversation:
{history_text}
"""


def generic_message_prompt(
    purpose: str,
    history_text: str,
    known_context: dict[str, Any] | None = None,
    guest_message: str | None = None,
) -> str:
    return f"""
You are a polite hotel WhatsApp assistant.

Write ONE short guest-facing WhatsApp message for this purpose:
{purpose}

Rules:
- Match the guest's language when clear; otherwise use the hotel's preferred language from knownContext
- Be concise, warm, and operational
- Do not mention internal systems, JSON, tools, tickets IDs, BPMN, or process variables
- Do not promise exact times unless provided
- Return ONLY valid JSON:

{{
  "message": "guest-facing message",
  "interaction": null
}}

Recommended interactions:
- If asking whether the guest wants to continue after inactivity, use CONTINUE_CONVERSATION and CANCEL_REQUEST.
- If asking the guest to confirm maintenance resolution, use MAINTENANCE_RESOLVED and MAINTENANCE_NOT_RESOLVED.
- If presenting a main navigation choice, use ROOM_SERVICE, SPA, MAINTENANCE, FRONT_DESK, FAQ.

{INTERACTION_JSON_GUIDANCE}

Known context:
{json.dumps(known_context or {}, ensure_ascii=False, indent=2)}

Guest message:
{guest_message or ""}

Conversation:
{history_text}
"""


def spa_menu_prompt(history_text: str, known_context: dict[str, Any]) -> str:
    return generic_message_prompt(
        purpose=(
            "Open the SPA reservation flow. The message MUST include this exact catalog URL: "
            "https://spa-hotelcristalino.catalogodigitalonline.com/ . "
            "Tell the guest that the link contains the SPA services, then ask them to provide "
            "the service they want, the desired date, and the desired time. "
            "Ask for exactly those three details, do not ask for party size, and do not ask "
            "for confirmation. Write in the guest's language and keep the message concise."
        ),
        history_text=history_text,
        known_context=known_context,
    )


def spa_reservation_details_prompt(
    guest_message: str,
    pending_reservation: dict[str, Any],
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return f"""
You collect the minimum information required for a hotel SPA availability request.

Merge the latest guest message into the pending reservation and return ONLY valid JSON:

{{
  "message": "short guest-facing message",
  "pendingReservation": {{
    "serviceName": null,
    "reservationDate": null,
    "reservationTime": null
  }},
  "missingFields": ["serviceName", "reservationDate", "reservationTime"],
  "requestComplete": false,
  "interaction": null
}}

Rules:
- The only required fields are serviceName, reservationDate, and reservationTime
- Preserve every non-empty value already present in pendingReservation unless the guest changes it
- Extract values from natural language without inventing information
- Keep natural-language dates and times when exact normalization is uncertain
- requestComplete is true only when all three required fields are non-empty
- missingFields contains exactly the required fields that are still empty
- If fields are missing, ask only for those missing fields in the guest's language
- If complete, briefly acknowledge that all request details were received
- Do not ask the guest to confirm the reservation
- Do not claim that SPA availability or the reservation has been confirmed
- Do not add buttons or lists

Pending reservation:
{json.dumps(pending_reservation or {}, ensure_ascii=False, indent=2)}

Known context:
{json.dumps(known_context or {}, ensure_ascii=False, indent=2)}

Latest guest message:
{guest_message}

Conversation:
{history_text}
"""


def spa_request_received_prompt(
    guest_message: str | None,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "Tell the guest that the SPA request was received and that the hotel team will "
            "review availability and contact them shortly to confirm the reservation. "
            "Do not claim the reservation is already confirmed. Do not ask another question "
            "and do not add buttons or lists. Write in the guest's language."
        ),
        history_text=history_text,
        known_context=known_context,
        guest_message=guest_message,
    )


def spa_reservation_confirmation_prompt(
    guest_message: str | None,
    extraction: dict[str, Any],
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return f"""
You are a hotel SPA reservation assistant.

Normalize the guest's SPA reservation request and write a short confirmation message.

Return ONLY valid JSON with this exact structure:

{{
  "message": "I have your SPA reservation as: relaxing massage for 2 people on Friday at 5 pm. Is that correct, or would you like to change anything?",
  "pendingReservation": {{
    "serviceName": null,
    "reservationDate": null,
    "reservationTime": null,
    "partySize": null,
    "roomNumber": null,
    "specialRequests": null
  }},
  "interaction": {{
    "type": "BUTTONS",
    "title": null,
    "body": null,
    "buttonText": null,
    "actions": [
      {{"id": "CONFIRM_SPA", "label": "Confirm"}},
      {{"id": "CHANGE_SPA", "label": "Change"}},
      {{"id": "CANCEL_SPA", "label": "Cancel"}}
    ]
  }}
}}

Rules:
- Use knownContext for available services and hours when provided
- Preserve natural-language dates or times if you cannot confidently normalize them
- Ask whether the reservation details are correct or if the guest wants to change anything
- Include confirmation actions with labels in the guest's language
- Do not say the reservation is confirmed with SPA staff yet

{INTERACTION_JSON_GUIDANCE}

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Extraction:
{json.dumps(extraction, ensure_ascii=False, indent=2)}

Guest message:
{guest_message or ""}

Conversation:
{history_text}
"""


def spa_confirmation_evaluation_prompt(
    guest_message: str,
    pending_reservation: dict[str, Any],
    history_text: str,
) -> str:
    return f"""
You are a hotel SPA assistant evaluating a guest reply to a SPA reservation confirmation.

Return ONLY valid JSON with this exact structure:

{{
  "confirmationAction": "CONFIRMED | CHANGE_REQUESTED | CANCELLED | UNCLEAR",
  "updatedReservation": {{
    "serviceName": null,
    "reservationDate": null,
    "reservationTime": null,
    "partySize": null,
    "roomNumber": null,
    "specialRequests": null
  }},
  "message": "short WhatsApp message",
  "interaction": null
}}

Decision rules:
- CONFIRMED: guest clearly accepts the pending reservation or selects CONFIRM_SPA
- CHANGE_REQUESTED: guest changes service, date, time, people, room, instructions, or selects CHANGE_SPA
- CANCELLED: guest clearly cancels or selects CANCEL_SPA
- UNCLEAR: reply is ambiguous

Message rules:
- For CHANGE_REQUESTED, restate the full updated reservation and ask if it is correct
- For UNCLEAR, ask the guest to confirm, change, or cancel
- For CANCELLED, confirm cancellation
- For CONFIRMED, write a brief acknowledgement; the BPMN process will notify SPA staff next
- For CHANGE_REQUESTED or UNCLEAR, include confirmation buttons: CONFIRM_SPA, CHANGE_SPA, CANCEL_SPA

{INTERACTION_JSON_GUIDANCE}

Pending reservation:
{json.dumps(pending_reservation, ensure_ascii=False, indent=2)}

Guest reply:
{guest_message}

Conversation:
{history_text}
"""


def spa_staff_response_prompt(
    staff_decision: str,
    staff_message: str,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return f"""
You are the copy editor and translator for a hotel's SPA team.

Rewrite the staff message as a concise, warm, professional WhatsApp response in the
language currently used by the guest. Return ONLY valid JSON:

{{
  "message": "guest-facing response",
  "interaction": null
}}

Strict rules:
- The staff message is the only source of truth for availability
- Preserve every concrete service, date, time, restriction, and alternative exactly
- Never invent or infer availability, services, dates, or times
- Do not omit alternatives supplied by staff
- Do not mention internal systems, spreadsheets, dashboards, or staff decisions
- If staffDecision is CONFIRMED, clearly confirm the reservation and do not ask a question
- If staffDecision is ALTERNATIVES_PROPOSED, explain the result and present the exact
  alternatives supplied by staff, then ask the guest to choose one, propose another
  date/time, or cancel
- Keep interaction null because alternatives may be free-form and change per response

Staff decision:
{staff_decision}

Staff message:
{staff_message}

Known context:
{json.dumps(known_context or {}, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""


def maintenance_initial_response_prompt(
    extraction: dict[str, Any],
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "Acknowledge that maintenance has been notified about the guest's issue. "
            "Mention the room if available and reassure the guest that the team will follow up."
        ),
        history_text=history_text,
        known_context={**known_context, "extraction": extraction},
    )


def maintenance_staff_update_prompt(
    staff_status: str,
    staff_message: str | None,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    if staff_status == "SOLVED":
        purpose = (
            "Tell the guest that maintenance marked the issue as resolved and ask them to confirm "
            "whether everything is now working correctly."
        )
    else:
        purpose = (
            "Tell the guest that maintenance reviewed the issue and further steps are required. "
            "Set expectations without inventing exact timing."
        )

    return generic_message_prompt(
        purpose=purpose,
        history_text=history_text,
        known_context={**known_context, "staffStatus": staff_status, "staffMessage": staff_message},
    )


def maintenance_guest_resolution_evaluation_prompt(
    guest_message: str,
    history_text: str,
    known_context: dict[str, Any],
) -> str:
    return f"""
You are a hotel maintenance assistant.

Evaluate whether the guest confirms that the maintenance issue was resolved.

Return ONLY valid JSON with this exact structure:

{{
  "guestConfirmedResolved": true,
  "message": "short WhatsApp message",
  "interaction": null
}}

Rules:
- guestConfirmedResolved=true if the guest clearly says the issue is solved, fixed, working, okay, thanks in a confirming way, or selects MAINTENANCE_RESOLVED
- guestConfirmedResolved=false if the guest says it is not fixed, still broken, worse, unclear, asks for more help, or selects MAINTENANCE_NOT_RESOLVED
- If true, thank the guest and say the request will be closed
- If false, apologize briefly and say the team will continue following up

{INTERACTION_JSON_GUIDANCE}

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Guest reply:
{guest_message}

Conversation:
{history_text}
"""


def unmatched_guest_response_prompt(
    guest_message: str,
    from_phone_number: str | None,
    known_context: dict[str, Any],
) -> str:
    return generic_message_prompt(
        purpose=(
            "The sender's phone number is not registered with an active hotel stay. "
            "Apologize and ask them to contact front desk if they are currently a guest."
        ),
        history_text="",
        known_context={**known_context, "fromPhoneNumber": from_phone_number},
        guest_message=guest_message,
    )
