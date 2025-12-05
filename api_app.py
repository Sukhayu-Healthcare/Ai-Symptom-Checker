from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rag_with_llm_step6 import analyze_patient


# -----------------------------
# FastAPI app initialization
# -----------------------------

app = FastAPI(
    title="AI Symptom Checker API",
    description="RAG + Gemini-based triage (Red/Orange/Yellow) with Marathi output",
    version="1.0.0",
)


# -----------------------------
# Request & Response models
# -----------------------------

class AnalyzeRequest(BaseModel):
    complaint: str
    followup_answers: List[str] = []  # previous answers in order


class AnalyzeResponse(BaseModel):
    zone: str
    zone_label: str
    internal_disease: str
    patient_symptoms_line: str
    patient_action_line: str
    followup_question: Optional[str]  # null if no more questions
    followups_used: int
    max_followups: int
    # optional debug info (can hide in frontend)
    baseline_disease: str
    baseline_zone: str


def get_zone_label(zone: str) -> str:
    labels = {
        "Red": "Zone: 🔴 Red – उच्च धोक्याची पातळी",
        "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
        "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
    }
    return labels.get(zone, f"Zone: {zone}")


def get_max_followups_for_zone(zone: str) -> int:
    if zone == "Red":
        return 3
    else:
        return 5  # Yellow or Orange


# -----------------------------
# Core helper: run analysis with history
# -----------------------------

def build_full_text(complaint: str, followup_answers: List[str]) -> str:
    """
    Rebuild text exactly like conversation_flow_step7.py:
    base complaint + appended "अतिरिक्त माहिती (फॉलो-अप उत्तर i): ..."
    """
    text = complaint.strip()
    for i, ans in enumerate(followup_answers, start=1):
        text += (
            "\n\n"
            + f"अतिरिक्त माहिती (फॉलो-अप उत्तर {i}): "
            + ans.strip()
        )
    return text


def run_triage(complaint: str, followup_answers: List[str]) -> AnalyzeResponse:
    """
    Stateless triage:
    - Rebuild full text from complaint + followup_answers
    - Call analyze_patient()
    - Enforce max followups per zone (Red:3, others:5)
    - If followups_used >= max, then do not send any more followup_question
    """

    full_text = build_full_text(complaint, followup_answers)
    result = analyze_patient(full_text)

    llm = result["llm_result"]
    baseline = result["baseline_prediction"]

    zone = llm["final_zone"]
    internal_disease = llm["internal_disease"]
    patient_symptoms_line = llm["patient_symptoms_line"]
    patient_action_line = llm["patient_action_line"]
    model_followup_q = (llm["followup_question"] or "").strip()

    baseline_disease = baseline["predicted_disease"]
    baseline_zone = baseline["predicted_zone"]

    zone_label = get_zone_label(zone)
    max_followups = get_max_followups_for_zone(zone)
    followups_used = len(followup_answers)

    # Enforce max followups rule: if we've already used enough, no more questions
    if followups_used >= max_followups:
        final_followup_q: Optional[str] = None
    else:
        final_followup_q = model_followup_q if model_followup_q else None

    return AnalyzeResponse(
        zone=zone,
        zone_label=zone_label,
        internal_disease=internal_disease,
        patient_symptoms_line=patient_symptoms_line,
        patient_action_line=patient_action_line,
        followup_question=final_followup_q,
        followups_used=followups_used,
        max_followups=max_followups,
        baseline_disease=baseline_disease,
        baseline_zone=baseline_zone,
    )


# -----------------------------
# API endpoints
# -----------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Analyze a patient's Marathi complaint + any previous follow-up answers.

    Frontend flow:
    - First call:  send complaint, followup_answers = []
    - If response.followup_question != null:
          show question to user, collect answer,
          call again with SAME complaint and followup_answers + [new_answer]
    - Stop when followup_question is null OR followups_used == max_followups.
    """
    return run_triage(request.complaint, request.followup_answers)


@app.get("/")
def root():
    return {"message": "AI Symptom Checker API is running. Use POST /analyze."}
