from typing import TypedDict, List, Dict, Any, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import json


app = FastAPI()


# ---------- Request / Response models ----------

class EvaluationRecord(BaseModel):
    ratingJson: Dict[str, Any] | None = None
    educationalComment: str | None = None


class EmployeeReviewRequest(BaseModel):
    evaluations: List[EvaluationRecord]


class EmployeeRecommendation(BaseModel):
    summary: str
    recommendation: Literal[
        "keep_as_is",
        "coaching_plan",
        "closer_monitoring",
        "promotion_consideration",
    ]
    priority_areas: List[str]
    manager_note: str


# ---------- LangGraph state ----------

class ReviewState(TypedDict):
    evaluations: List[Dict[str, Any]]
    history_text: str
    trends: Dict[str, Any]
    recommendation_json: Dict[str, Any]


# ---------- Local model ----------

llm = ChatOllama(
    model="gemma3",
    temperature=0,
)


# ---------- Nodes ----------

def collect_history(state: ReviewState) -> ReviewState:
    evaluations = state["evaluations"]

    parts = []
    for i, item in enumerate(evaluations, start=1):
        rating = item.get("ratingJson")
        comment = item.get("educationalComment")

        parts.append(f"Evaluation #{i}")
        parts.append(f"ratingJson: {json.dumps(rating, ensure_ascii=False)}")
        parts.append(f"educationalComment: {comment}")
        parts.append("")

    history_text = "\n".join(parts)

    return {
        **state,
        "history_text": history_text,
    }


def compute_trends(state: ReviewState) -> ReviewState:
    evaluations = state["evaluations"]

    metric_names = [
        "clarity_of_communication",
        "professionalism_and_tone",
        "needs_discovery",
        "objection_handling",
        "closing_effectiveness",
    ]

    grade_counts = {
        metric: {"A": 0, "B": 0, "C": 0, "D": 0}
        for metric in metric_names
    }

    strength_counts: Dict[str, int] = {}
    weakness_counts: Dict[str, int] = {}

    for item in evaluations:
        rating = item.get("ratingJson") or {}
        metrics = rating.get("metrics") or {}

        for metric in metric_names:
            grade = ((metrics.get(metric) or {}).get("grade"))
            if grade in grade_counts[metric]:
                grade_counts[metric][grade] += 1

        for text in rating.get("did_right", []) or []:
            strength_counts[text] = strength_counts.get(text, 0) + 1

        for text in rating.get("did_wrong", []) or []:
            weakness_counts[text] = weakness_counts.get(text, 0) + 1

    weakest_metrics = sorted(
        metric_names,
        key=lambda m: (
            grade_counts[m]["D"] * 10 +
            grade_counts[m]["C"] * 5 -
            grade_counts[m]["A"] * 2
        ),
        reverse=True
    )

    strongest_metrics = sorted(
        metric_names,
        key=lambda m: (
            grade_counts[m]["A"] * 10 +
            grade_counts[m]["B"] * 3 -
            grade_counts[m]["D"] * 5
        ),
        reverse=True
    )

    top_strengths = sorted(
        strength_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top_weaknesses = sorted(
        weakness_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    trends = {
        "total_evaluations": len(evaluations),
        "grade_counts_by_metric": grade_counts,
        "strongest_metrics": strongest_metrics[:3],
        "weakest_metrics": weakest_metrics[:3],
        "top_strengths": top_strengths,
        "top_weaknesses": top_weaknesses,
    }

    return {
        **state,
        "trends": trends,
    }


def recommend_action(state: ReviewState) -> ReviewState:
    history_text = state["history_text"]
    trends = state["trends"]

    prompt = f"""
You are an employee performance review agent.

You are reviewing the historical sales evaluation results for ONE employee.
All evaluations below belong to the same employee.

You have:
1. the raw evaluation history
2. computed trend data summarizing repeated patterns

Use BOTH.

Return ONLY valid JSON with this exact structure:

{{
  "summary": "short summary",
  "recommendation": "keep_as_is | coaching_plan | closer_monitoring | promotion_consideration",
  "priority_areas": ["area 1", "area 2"],
  "manager_note": "short manager note"
}}

Rules:
- Be realistic and conservative
- If repeated weaknesses appear, prioritize coaching_plan or closer_monitoring
- Only recommend promotion_consideration if the pattern is consistently strong
- priority_areas must contain exactly 2 items
- Return JSON only, no markdown fences

Computed trends:
{json.dumps(trends, ensure_ascii=False, indent=2)}

Employee evaluation history:
{history_text}
"""

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    cleaned = (
        content.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    recommendation_json = json.loads(cleaned)

    return {
        **state,
        "recommendation_json": recommendation_json,
    }


# ---------- Build graph ----------

graph = StateGraph(ReviewState)
graph.add_node("collect_history", collect_history)
graph.add_node("compute_trends", compute_trends)
graph.add_node("recommend_action", recommend_action)

graph.set_entry_point("collect_history")
graph.add_edge("collect_history", "compute_trends")
graph.add_edge("compute_trends", "recommend_action")
graph.add_edge("recommend_action", END)

review_graph = graph.compile()


# ---------- API ----------

@app.post("/employee-review")
def employee_review(request: EmployeeReviewRequest):
    initial_state: ReviewState = {
        "evaluations": [item.model_dump() for item in request.evaluations],
        "history_text": "",
        "trends": {},
        "recommendation_json": {},
    }

    result = review_graph.invoke(initial_state)
    return result["recommendation_json"]


@app.post("/employee-review/debug")
def employee_review_debug(request: EmployeeReviewRequest):
    initial_state: ReviewState = {
        "evaluations": [item.model_dump() for item in request.evaluations],
        "history_text": "",
        "trends": {},
        "recommendation_json": {},
    }

    result = review_graph.invoke(initial_state)
    return {
        "trends": result["trends"],
        "recommendation": result["recommendation_json"],
    }