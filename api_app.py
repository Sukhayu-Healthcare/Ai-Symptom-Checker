# from typing import List, Optional

# from fastapi import FastAPI
# from pydantic import BaseModel

# from rag_with_llm_step6 import analyze_patient


# # -----------------------------
# # FastAPI app initialization
# # -----------------------------

# app = FastAPI(
#     title="AI Symptom Checker API",
#     description="RAG + Gemini-based triage (Red/Orange/Yellow) with Marathi output",
#     version="1.0.0",
# )


# # -----------------------------
# # Request & Response models
# # -----------------------------

# class AnalyzeRequest(BaseModel):
#     complaint: str
#     followup_answers: List[str] = []  # previous answers in order


# class AnalyzeResponse(BaseModel):
#     zone: str
#     zone_label: str
#     internal_disease: str
#     patient_symptoms_line: str
#     patient_action_line: str
#     followup_question: Optional[str]  # null if no more questions
#     followups_used: int
#     max_followups: int
#     # optional debug info (can hide in frontend)
#     baseline_disease: str
#     baseline_zone: str


# def get_zone_label(zone: str) -> str:
#     labels = {
#         "Red": "Zone: 🔴 Red – उच्च धोक्याची पातळी",
#         "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
#         "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
#     }
#     return labels.get(zone, f"Zone: {zone}")


# def get_max_followups_for_zone(zone: str) -> int:
#     if zone == "Red":
#         return 3
#     else:
#         return 5  # Yellow or Orange


# # -----------------------------
# # Core helper: run analysis with history
# # -----------------------------

# def build_full_text(complaint: str, followup_answers: List[str]) -> str:
#     """
#     Rebuild text exactly like conversation_flow_step7.py:
#     base complaint + appended "अतिरिक्त माहिती (फॉलो-अप उत्तर i): ..."
#     """
#     text = complaint.strip()
#     for i, ans in enumerate(followup_answers, start=1):
#         text += (
#             "\n\n"
#             + f"अतिरिक्त माहिती (फॉलो-अप उत्तर {i}): "
#             + ans.strip()
#         )
#     return text


# def run_triage(complaint: str, followup_answers: List[str]) -> AnalyzeResponse:
#     """
#     Stateless triage:
#     - Rebuild full text from complaint + followup_answers
#     - Call analyze_patient()
#     - Enforce max followups per zone (Red:3, others:5)
#     - If followups_used >= max, then do not send any more followup_question
#     """

#     full_text = build_full_text(complaint, followup_answers)
#     result = analyze_patient(full_text)

#     llm = result["llm_result"]
#     baseline = result["baseline_prediction"]

#     zone = llm["final_zone"]
#     internal_disease = llm["internal_disease"]
#     patient_symptoms_line = llm["patient_symptoms_line"]
#     patient_action_line = llm["patient_action_line"]
#     model_followup_q = (llm["followup_question"] or "").strip()

#     baseline_disease = baseline["predicted_disease"]
#     baseline_zone = baseline["predicted_zone"]

#     zone_label = get_zone_label(zone)
#     max_followups = get_max_followups_for_zone(zone)
#     followups_used = len(followup_answers)

#     # Enforce max followups rule: if we've already used enough, no more questions
#     if followups_used >= max_followups:
#         final_followup_q: Optional[str] = None
#     else:
#         final_followup_q = model_followup_q if model_followup_q else None

#     return AnalyzeResponse(
#         zone=zone,
#         zone_label=zone_label,
#         internal_disease=internal_disease,
#         patient_symptoms_line=patient_symptoms_line,
#         patient_action_line=patient_action_line,
#         followup_question=final_followup_q,
#         followups_used=followups_used,
#         max_followups=max_followups,
#         baseline_disease=baseline_disease,
#         baseline_zone=baseline_zone,
#     )


# # -----------------------------
# # API endpoints
# # -----------------------------

# @app.post("/analyze", response_model=AnalyzeResponse)
# def analyze(request: AnalyzeRequest):
#     """
#     Analyze a patient's Marathi complaint + any previous follow-up answers.

#     Frontend flow:
#     - First call:  send complaint, followup_answers = []
#     - If response.followup_question != null:
#           show question to user, collect answer,
#           call again with SAME complaint and followup_answers + [new_answer]
#     - Stop when followup_question is null OR followups_used == max_followups.
#     """
#     return run_triage(request.complaint, request.followup_answers)


# @app.get("/")
# def root():
#     return {"message": "AI Symptom Checker API is running. Use POST /analyze."}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

from rag_with_llm_step6 import analyze_patient


app = FastAPI()

# Optional CORS (safe for local and Android app later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request & Response Models ----------

class AnalyzeRequest(BaseModel):
    complaint: str
    followup_answers: List[str] = []


class AnalyzeResponse(BaseModel):
    zone: str
    zone_label: str
    internal_disease: str
    patient_symptoms_line: str
    patient_action_line: str
    followup_question: str
    followups_used: int
    max_followups: int
    baseline_disease: str
    baseline_zone: str


# ---------- API endpoint for Android / tools ----------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Core API: takes complaint + optional followup answers,
    calls RAG + Gemini pipeline and returns structured triage info.
    """
    # Build combined text (same as conversation_flow_step7 logic)
    text = req.complaint.strip()
    for i, ans in enumerate(req.followup_answers, start=1):
        text += f"\n\nअतिरिक्त माहिती (फॉलो-अप उत्तर {i}): {ans}"

    # Run full pipeline
    result = analyze_patient(text)
    llm = result["llm_result"]

    zone = llm["final_zone"]
    zone_labels = {
        "Red": "Zone: 🔴 Red – उच्च धोक्याची पातळी",
        "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
        "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
    }
    zone_label = zone_labels.get(zone, f"Zone: {zone}")

    # Decide max followups based on zone (same logic as conversation_flow_step7)
    if zone == "Red":
        max_followups = 3
    else:
        max_followups = 5

    return AnalyzeResponse(
        zone=zone,
        zone_label=zone_label,
        internal_disease=llm["internal_disease"],
        patient_symptoms_line=llm["patient_symptoms_line"],
        patient_action_line=llm["patient_action_line"],
        followup_question=llm["followup_question"],
        followups_used=len(req.followup_answers),
        max_followups=max_followups,
        baseline_disease=result["baseline_prediction"]["predicted_disease"],
        baseline_zone=result["baseline_prediction"]["predicted_zone"],
    )


# ---------- SIMPLE WEB UI (BASIC WEBSITE) ----------

@app.get("/", response_class=HTMLResponse)
def web_ui():
    """
    Very simple HTML UI so you can test the symptom checker in browser
    instead of terminal or Hoppscotch.
    """
    html = """
    <!DOCTYPE html>
    <html lang="mr">
    <head>
        <meta charset="UTF-8" />
        <title>AI Symptom Checker</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: flex-start;
            }
            .container {
                max-width: 800px;
                width: 100%;
                background: #ffffff;
                margin-top: 30px;
                padding: 20px 24px 32px;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            }
            h1 {
                font-size: 1.6rem;
                margin-bottom: 4px;
            }
            h2 {
                font-size: 1.2rem;
                margin-top: 24px;
            }
            p.subtitle {
                margin-top: 0;
                color: #555;
                font-size: 0.95rem;
            }
            label {
                font-weight: 600;
                display: block;
                margin-bottom: 6px;
            }
            textarea, input[type="text"] {
                width: 100%;
                box-sizing: border-box;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid #ccc;
                font-size: 0.95rem;
                resize: vertical;
            }
            textarea {
                min-height: 100px;
            }
            button {
                margin-top: 12px;
                padding: 10px 16px;
                border-radius: 999px;
                border: none;
                font-size: 0.95rem;
                cursor: pointer;
                background: #2563eb;
                color: white;
                font-weight: 600;
            }
            button.secondary {
                background: #6b7280;
                margin-left: 8px;
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .output {
                margin-top: 20px;
                padding: 14px 16px;
                background: #f9fafb;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                font-size: 0.95rem;
            }
            .row {
                margin-bottom: 8px;
            }
            .label {
                font-weight: 600;
            }
            .zone-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.9rem;
                font-weight: 600;
                margin-top: 4px;
            }
            .zone-red {
                background: #fee2e2;
                color: #b91c1c;
            }
            .zone-orange {
                background: #ffedd5;
                color: #c05621;
            }
            .zone-yellow {
                background: #fef9c3;
                color: #854d0e;
            }
            .note {
                margin-top: 10px;
                font-size: 0.85rem;
                color: #555;
            }
            .followup-block {
                margin-top: 16px;
                padding-top: 10px;
                border-top: 1px dashed #e5e7eb;
            }
            .small {
                font-size: 0.85rem;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Symptom Checker (Marathi)</h1>
            <p class="subtitle">तुमची तक्रार मराठीत लिहा आणि प्रणाली तुम्हाला कोणत्या जोखीम झोनमध्ये येता ते दाखवेल.</p>

            <label for="complaint">आपली मुख्य तक्रार (लक्षणे):</label>
            <textarea id="complaint" placeholder="उदा. मला छातीत खूप दुखतंय आणि श्वास घ्यायला त्रास होतोय"></textarea>

            <button id="analyzeBtn">Analyze / विश्लेषण करा</button>
            <button id="resetBtn" class="secondary">Reset</button>

            <div class="output" id="output" style="display:none;">
                <div class="row">
                    <span class="label">Zone:</span>
                    <div id="zoneLabel"></div>
                </div>
                <div class="row">
                    <span class="label">लक्षणे:</span>
                    <div id="symptomsLine"></div>
                </div>
                <div class="row">
                    <span class="label">काय करावे:</span>
                    <div id="actionLine"></div>
                </div>
                <div class="note">
                    नोट: हा केवळ प्राथमिक अंदाज आहे, कृपया डॉक्टरांचा सल्ला नक्की घ्या.
                </div>

                <div class="followup-block" id="followupBlock" style="display:none;">
                    <div class="row">
                        <span class="label">Follow-up प्रश्न:</span>
                        <div id="followupQuestion"></div>
                    </div>
                    <label for="followupAnswer">आपले उत्तर:</label>
                    <input type="text" id="followupAnswer" placeholder="मराठीत उत्तर लिहा" />
                    <button id="sendFollowupBtn">Send Follow-up / पुन्हा विचारणा</button>
                    <div class="small" id="followupInfo"></div>
                </div>
            </div>
        </div>

        <script>
            const analyzeBtn = document.getElementById("analyzeBtn");
            const resetBtn = document.getElementById("resetBtn");
            const outputDiv = document.getElementById("output");
            const complaintInput = document.getElementById("complaint");
            const zoneLabelDiv = document.getElementById("zoneLabel");
            const symptomsLineDiv = document.getElementById("symptomsLine");
            const actionLineDiv = document.getElementById("actionLine");
            const followupBlock = document.getElementById("followupBlock");
            const followupQuestionDiv = document.getElementById("followupQuestion");
            const followupAnswerInput = document.getElementById("followupAnswer");
            const sendFollowupBtn = document.getElementById("sendFollowupBtn");
            const followupInfo = document.getElementById("followupInfo");

            let followupAnswers = [];

            function zoneBadge(zoneLabelText) {
                if (!zoneLabelText) return "";
                let cls = "zone-yellow";
                if (zoneLabelText.includes("Red")) cls = "zone-red";
                else if (zoneLabelText.includes("Orange")) cls = "zone-orange";
                return `<span class="zone-badge ${cls}">${zoneLabelText}</span>`;
            }

            async function callAnalyze() {
                const complaint = complaintInput.value.trim();
                if (!complaint) {
                    alert("कृपया आपली तक्रार लिहा.");
                    return;
                }

                analyzeBtn.disabled = true;
                sendFollowupBtn.disabled = true;

                try {
                    const resp = await fetch("/analyze", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            complaint: complaint,
                            followup_answers: followupAnswers
                        })
                    });

                    if (!resp.ok) {
                        const txt = await resp.text();
                        alert("Error from server: " + txt);
                        analyzeBtn.disabled = false;
                        sendFollowupBtn.disabled = false;
                        return;
                    }

                    const data = await resp.json();

                    outputDiv.style.display = "block";
                    zoneLabelDiv.innerHTML = zoneBadge(data.zone_label);
                    symptomsLineDiv.textContent = data.patient_symptoms_line;
                    actionLineDiv.textContent = data.patient_action_line;

                    if (data.followup_question && data.followup_question.trim() !== "") {
                        followupBlock.style.display = "block";
                        followupQuestionDiv.textContent = data.followup_question;
                        followupInfo.textContent = `Follow-ups used: ${data.followups_used} / ${data.max_followups}`;
                        sendFollowupBtn.disabled = false;
                    } else {
                        followupBlock.style.display = "none";
                    }

                } catch (err) {
                    alert("Network or server error: " + err);
                } finally {
                    analyzeBtn.disabled = false;
                }
            }

            analyzeBtn.addEventListener("click", () => {
                followupAnswers = [];  // fresh start each time for complaint
                followupAnswerInput.value = "";
                callAnalyze();
            });

            sendFollowupBtn.addEventListener("click", () => {
                const ans = followupAnswerInput.value.trim();
                if (!ans) {
                    alert("कृपया follow-up प्रश्नाचे उत्तर लिहा.");
                    return;
                }
                followupAnswers.push(ans);
                followupAnswerInput.value = "";
                callAnalyze();
            });

            resetBtn.addEventListener("click", () => {
                complaintInput.value = "";
                followupAnswers = [];
                outputDiv.style.display = "none";
                followupBlock.style.display = "none";
                followupAnswerInput.value = "";
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
