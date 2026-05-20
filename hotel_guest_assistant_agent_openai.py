from typing import TypedDict, List, Dict, Any, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os
import json


app = FastAPI()


# ---------- OpenAI client ----------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def call_openai_json(prompt: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0,
        text={
            "format": {
                "type": "json_object"
            }
        }
    )

    content = response.output_text
    return json.loads(content)


# ---------- Request / Response models ----------

class ConversationMessage(BaseModel):
    role: Literal["guest", "assistant", "staff", "system"]
    content: str


class HotelConversationRequest(BaseModel):
    conversationId: str | None = None
    guestMessage: str
    conversationHistory: List[ConversationMessage] = []
    knownContext: Dict[str, Any] = {}


class HotelExtractionResponse(BaseModel):
    intent: Literal[
        "FAQ",
        "ROOM_SERVICE",
        "MAINTENANCE",
        "HOUSEKEEPING",
        "COMPLAINT",
        "HUMAN_REVIEW",
        "UNKNOWN",
    ]
    confidence: float
    containsEmergency: bool
    roomNumber: str | None = None
    language: str | None = None
    missingFields: List[str]
    extractedEntities: Dict[str, Any]
    requestComplete: bool


class ClarificationRequest(BaseModel):
    extraction: Dict[str, Any]
    conversationHistory: List[ConversationMessage] = []


class ClarificationResponse(BaseModel):
    message: str


class FaqResponseRequest(BaseModel):
    guestMessage: str
    conversationHistory: List[ConversationMessage] = []
    knownContext: Dict[str, Any] = {}


class FaqResponse(BaseModel):
    message: str


# ---------- LangGraph state ----------

class HotelConversationState(TypedDict):
    guest_message: str
    conversation_history: List[Dict[str, Any]]
    known_context: Dict[str, Any]
    history_text: str
    extraction_json: Dict[str, Any]
    clarification_message: str
    faq_message: str


# ---------- Helpers ----------

def build_history_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------- Nodes ----------

def collect_conversation_context(state: HotelConversationState) -> HotelConversationState:
    conversation_history = state["conversation_history"]
    guest_message = state["guest_message"]

    full_history = conversation_history + [
        {
            "role": "guest",
            "content": guest_message,
        }
    ]

    return {
        **state,
        "history_text": build_history_text(full_history),
    }


def extract_intent_and_entities(state: HotelConversationState) -> HotelConversationState:
    history_text = state["history_text"]
    known_context = state["known_context"]

    prompt = f"""
You are a hotel WhatsApp assistant extraction agent.

Your job is to read the conversation and extract the guest's current request as structured JSON.

Return ONLY valid JSON with this exact structure:

{{
  "intent": "FAQ | ROOM_SERVICE | MAINTENANCE | HOUSEKEEPING | COMPLAINT | HUMAN_REVIEW | UNKNOWN",
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

For MAINTENANCE, required fields are:
- roomNumber
- issueDescription

For HOUSEKEEPING, required fields are:
- roomNumber
- requestedItemsOrService

For COMPLAINT, required fields are:
- roomNumber if the complaint is room-specific
- complaintDescription

For FAQ, usually no required fields unless the question is ambiguous.

If required information is missing, add field names to missingFields and set requestComplete=false.
If enough information is present, missingFields=[] and requestComplete=true.

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""

    extraction_json = call_openai_json(prompt)

    return {
        **state,
        "extraction_json": extraction_json,
    }


def generate_clarification_question(state: HotelConversationState) -> HotelConversationState:
    extraction = state["extraction_json"]
    history_text = state["history_text"]

    prompt = f"""
You are a polite hotel WhatsApp assistant.

The guest's request is not complete yet.
Write ONE short clarification question to get the missing information.

Rules:
- Be natural and concise
- Do not mention JSON, intent, confidence, or internal fields
- Ask only for the most important missing information
- If roomNumber is missing, ask for the room number
- If room service items are missing, ask what they would like to order
- If maintenance issue is unclear, ask what is not working
- Return ONLY valid JSON:

{{
  "message": "your clarification question"
}}

Extraction:
{json.dumps(extraction, ensure_ascii=False, indent=2)}

Conversation:
{history_text}
"""

    clarification_json = call_openai_json(prompt)

    return {
        **state,
        "clarification_message": clarification_json["message"],
    }


def generate_faq_response(state: HotelConversationState) -> HotelConversationState:
    guest_message = state["guest_message"]
    history_text = state["history_text"]
    known_context = state["known_context"]

    prompt = f"""
You are a hotel WhatsApp assistant answering FAQ questions.

Answer the guest in a helpful, concise, hotel-friendly tone.

Rules:
- If the answer is available in knownContext, use it
- If not available, say you can check with the front desk
- Do not invent exact prices, opening hours, or policies unless provided
- Return ONLY valid JSON:

{{
  "message": "answer to guest"
}}

Known context:
{json.dumps(known_context, ensure_ascii=False, indent=2)}

Current guest message:
{guest_message}

Conversation:
{history_text}
"""

    faq_json = call_openai_json(prompt)

    return {
        **state,
        "faq_message": faq_json["message"],
    }


# ---------- Build extraction graph ----------

extract_graph_builder = StateGraph(HotelConversationState)
extract_graph_builder.add_node("collect_context", collect_conversation_context)
extract_graph_builder.add_node("extract_intent", extract_intent_and_entities)

extract_graph_builder.set_entry_point("collect_context")
extract_graph_builder.add_edge("collect_context", "extract_intent")
extract_graph_builder.add_edge("extract_intent", END)

extract_graph = extract_graph_builder.compile()


# ---------- Build clarification graph ----------

clarification_graph_builder = StateGraph(HotelConversationState)
clarification_graph_builder.add_node("collect_context", collect_conversation_context)
clarification_graph_builder.add_node("clarify", generate_clarification_question)

clarification_graph_builder.set_entry_point("collect_context")
clarification_graph_builder.add_edge("collect_context", "clarify")
clarification_graph_builder.add_edge("clarify", END)

clarification_graph = clarification_graph_builder.compile()


# ---------- Build FAQ graph ----------

faq_graph_builder = StateGraph(HotelConversationState)
faq_graph_builder.add_node("collect_context", collect_conversation_context)
faq_graph_builder.add_node("faq", generate_faq_response)

faq_graph_builder.set_entry_point("collect_context")
faq_graph_builder.add_edge("collect_context", "faq")
faq_graph_builder.add_edge("faq", END)

faq_graph = faq_graph_builder.compile()


# ---------- API endpoints for BPMN delegates ----------

@app.post("/hotel/extract-intent", response_model=HotelExtractionResponse)
def hotel_extract_intent(request: HotelConversationRequest):
    initial_state: HotelConversationState = {
        "guest_message": request.guestMessage,
        "conversation_history": [item.model_dump() for item in request.conversationHistory],
        "known_context": request.knownContext,
        "history_text": "",
        "extraction_json": {},
        "clarification_message": "",
        "faq_message": "",
    }

    result = extract_graph.invoke(initial_state)
    return result["extraction_json"]


@app.post("/hotel/generate-clarification", response_model=ClarificationResponse)
def hotel_generate_clarification(request: ClarificationRequest):
    history = [item.model_dump() for item in request.conversationHistory]

    initial_state: HotelConversationState = {
        "guest_message": "",
        "conversation_history": history,
        "known_context": {},
        "history_text": build_history_text(history),
        "extraction_json": request.extraction,
        "clarification_message": "",
        "faq_message": "",
    }

    result = generate_clarification_question(initial_state)

    return {
        "message": result["clarification_message"],
    }


@app.post("/hotel/faq-response", response_model=FaqResponse)
def hotel_faq_response(request: FaqResponseRequest):
    initial_state: HotelConversationState = {
        "guest_message": request.guestMessage,
        "conversation_history": [item.model_dump() for item in request.conversationHistory],
        "known_context": request.knownContext,
        "history_text": "",
        "extraction_json": {},
        "clarification_message": "",
        "faq_message": "",
    }

    result = faq_graph.invoke(initial_state)

    return {
        "message": result["faq_message"],
    }


@app.post("/hotel/debug")
def hotel_debug(request: HotelConversationRequest):
    initial_state: HotelConversationState = {
        "guest_message": request.guestMessage,
        "conversation_history": [item.model_dump() for item in request.conversationHistory],
        "known_context": request.knownContext,
        "history_text": "",
        "extraction_json": {},
        "clarification_message": "",
        "faq_message": "",
    }

    result = extract_graph.invoke(initial_state)

    return {
        "history_text": result["history_text"],
        "extraction": result["extraction_json"],
    }