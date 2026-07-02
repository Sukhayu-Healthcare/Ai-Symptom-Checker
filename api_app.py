
"""
api_app.py
==========

FastAPI backend for the AI Symptom Checker.

This file ONLY wires up the web layer (FastAPI routes, request/response
shapes, conversation-history -> prompt-text reconstruction, and the
zone -> max_followups rule). It does NOT touch:

  - SentenceTransformer embedding
  - Qdrant retrieval
  - Weighted k-NN
  - Groq prompt / temperature / disease list

All of that lives untouched in rag_with_llm_step6.py and is called through
analyze_patient(text).

Conversation flow ("Option A")
-------------------------------
The frontend sends the ORIGINAL complaint plus the full list of
question/answer pairs so far. This backend reconstructs the exact same
text format that conversation_flow_step7.py builds turn-by-turn, so Groq
sees identical context to the terminal version and does not lose track
of what was already asked / answered.

Reconstructed format (must match conversation_flow_step7.py exactly):

    <complaint>

            FOLLOW-UP 1

            Question:
            <question 1>

            Patient Answer:
            <answer 1>

            FOLLOW-UP 2

            Question:
            <question 2>

            Patient Answer:
            <answer 2>
"""

from fastapi.responses import HTMLResponse
from fastapi import Request

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from rag_with_llm_step6 import analyze_patient


app = FastAPI(title="AI Symptom Checker")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =========================
# Zone metadata (display only — does not touch AI logic)
# =========================

ZONE_LABELS = {
    "Red": "उच्च धोक्याची पातळी",
    "Orange": "मध्यम धोक्याची पातळी",
    "Yellow": "कमी धोक्याची पातळी",
}

# Mirrors the exact rule used in conversation_flow_step7.run_conversation()
def max_followups_for_zone(zone: str) -> int:
    if zone == "Red":
        return 3
    elif zone == "Orange":
        return 3
    return 4


# =========================
# Request / response schemas
# =========================

class ConversationTurn(BaseModel):
    question: str
    answer: str


class AnalyzeRequest(BaseModel):
    complaint: str
    conversation_history: Optional[List[ConversationTurn]] = []


class AnalyzeResponse(BaseModel):
    zone: str
    zone_label: str
    internal_disease: str
    patient_symptoms_line: str
    patient_action_line: str
    confidence: float
    followup_question: str
    followups_used: int
    max_followups: int


# =========================
# Helpers
# =========================

def reconstruct_text(complaint: str, history: List[ConversationTurn]) -> str:
    """
    Rebuild the exact prompt text conversation_flow_step7.py would have
    built turn by turn, from a complaint + full conversation history.
    """
    text = complaint.strip()

    for i, turn in enumerate(history, start=1):
        text += f"""

        FOLLOW-UP {i}

        Question:
        {turn.question}

        Patient Answer:
        {turn.answer}
        """

    return text


def normalize_answer(question: str, answer: str) -> str:
    """Same lightweight normalization conversation_flow_step7.py applies."""
    a = answer.strip()
    low = a.lower()

    if low in ("ho", "haa", "yes"):
        a = "हो"
    elif low in ("nhi", "no"):
        a = "नाही"

    if question.startswith("ताप किती दिवस") and a.isdigit():
        a += " दिवस"

    return a


# =========================
# Routes
# =========================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):

    return templates.TemplateResponse(
        "performance.html",
        {
            "request": request
        }
    )

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    complaint = (payload.complaint or "").strip()
    if not complaint:
        raise HTTPException(status_code=400, detail="Complaint text is required.")

    history = payload.conversation_history or []

    # Normalize the most recent answer the same way the terminal script does
    if history:
        last = history[-1]
        last.answer = normalize_answer(last.question, last.answer)

    full_text = reconstruct_text(complaint, history)

    try:
        result = analyze_patient(full_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    llm = result["llm_result"]
    # Don't show retrieval score for out-of-domain queries
    confidence = result["baseline_prediction"]["confidence"]

    if llm["internal_disease"] == "":
        confidence = 0.0

    zone = llm["final_zone"]
    max_followups = max_followups_for_zone(zone)
    followups_used = len(history)

    followup_question = (llm["followup_question"] or "").strip()

    # Respect the same stopping rule as the terminal version
    if followups_used >= max_followups:
        followup_question = ""

    return AnalyzeResponse(
        zone=zone,
        zone_label=ZONE_LABELS.get(zone, zone),
        internal_disease=llm["internal_disease"],
        patient_symptoms_line=llm["patient_symptoms_line"],
        patient_action_line=llm["patient_action_line"],
        confidence=confidence,
        followup_question=followup_question,
        followups_used=followups_used,
        max_followups=max_followups,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True)