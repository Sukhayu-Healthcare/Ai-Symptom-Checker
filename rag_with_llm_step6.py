from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import Dict, Any
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from google import genai  # Gemini SDK


# =========================
# STEP 0: Configure Gemini client
# =========================

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is not set in environment.")

client = genai.Client(api_key=gemini_key)


# =========================
# STEP 1: Configure Qdrant + embedding model
# =========================

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = "symptom_cases"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment.")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,
)

print("\nLoading embedding model (for query encoding)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")


# =========================
# STEP 2: Retrieval + k-NN prediction (via Qdrant)
# =========================

def retrieve_similar_cases(query_text: str, k: int = 10):
    """
    Use Qdrant to retrieve top-k similar past cases.
    """

    # Encode query to vector
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0].astype("float32")

    # Qdrant search (cosine similarity -> higher score = more similar)
    search_result = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding.tolist(),
        limit=k,
        with_payload=True,
    )

    results = []
    for point in search_result:
        payload = point.payload or {}
        score = float(point.score)  # similarity

        # convert similarity to distance-style so old weighting logic still works
        distance = 1.0 - score

        case = {
            "utterance": payload.get("utterance", ""),
            "disease": payload.get("disease", ""),
            "zone": payload.get("zone", ""),
            "present_symptoms": payload.get("present_symptoms", ""),
            "advice": payload.get("advice", ""),
            "distance": distance,
        }
        results.append(case)

    return results


def predict_disease_zone(user_text: str, k: int = 10) -> Dict[str, Any]:
    """
    k-NN-style prediction over retrieved cases:
      - Aggregate scores per disease using distance-based weights.
      - Choose majority zone among top cases of that disease.
    """

    retrieved = retrieve_similar_cases(user_text, k=k)

    disease_scores = defaultdict(float)
    disease_examples = defaultdict(list)
    eps = 1e-6

    for case in retrieved:
        d = case["disease"]
        dist = case["distance"]
        # smaller distance -> higher weight
        weight = 1.0 / (dist + eps)

        disease_scores[d] += weight
        disease_examples[d].append(case)

    if not disease_scores:
        return {}

    best_disease = max(disease_scores.items(), key=lambda x: x[1])[0]
    best_disease_score = disease_scores[best_disease]

    zones = [c["zone"] for c in disease_examples[best_disease]]
    zone_counts = Counter(zones)
    best_zone = zone_counts.most_common(1)[0][0]

    best_example = sorted(
        disease_examples[best_disease],
        key=lambda c: c["distance"]
    )[0]

    representative_advice = best_example["advice"]
    representative_symptoms = best_example["present_symptoms"]

    result = {
        "user_text": user_text,
        "predicted_disease": best_disease,
        "predicted_zone": best_zone,
        "representative_advice": representative_advice,
        "representative_symptoms": representative_symptoms,
        "supporting_cases": retrieved,
        "disease_scores": dict(disease_scores),
        "raw_zone_counts": dict(zone_counts),
        "best_disease_score": best_disease_score,
    }

    return result


# =========================
# STEP 3: Build LLM prompt (same logic as before)
# =========================

ALLOWED_DISEASES = [
    "Viral Fever (without warning signs)",
    "Gastritis / Acid Reflux",
    "Migraine / Tension Headache",
    "Skin Infection / Cellulitis (mild)",
    "Moderate Hypertension (BP 140–160/90–100)",
    "Pregnancy Complications (Bleeding / Pain)",
    "Severe Dehydration",
    "Snake Bite (Suspected)",
    "Tuberculosis (with cough >2 weeks)",
    "Seizure / Fits (resolved)",
    "Heart Attack (Myocardial Infarction)",
    "Unconscious / Coma",
    "Stroke (CVA)",
    "Severe Head Injury",
    "Severe Trauma with Bleeding / Fracture",
]

ALLOWED_ZONES = ["Red", "Orange", "Yellow"]


def build_llm_prompt(pred: Dict[str, Any]) -> str:
    user_text = pred["user_text"]
    predicted_disease = pred["predicted_disease"]
    predicted_zone = pred["predicted_zone"]
    representative_advice = pred["representative_advice"]
    representative_symptoms = pred["representative_symptoms"]
    supporting_cases = pred["supporting_cases"]

    similar_blocks = []
    for i, case in enumerate(supporting_cases[:5], start=1):
        block = (
            f"Case {i}:\n"
            f"  Utterance: {case['utterance']}\n"
            f"  Disease: {case['disease']}\n"
            f"  Zone: {case['zone']}\n"
            f"  Symptoms: {case['present_symptoms']}\n"
            f"  Advice: {case['advice']}\n"
        )
        similar_blocks.append(block)

    similar_text = "\n".join(similar_blocks)

    system_instructions = f"""
You are an AI symptom checker assistant helping with triage (emergency vs non-emergency) for patients.
The patient speaks Marathi. Respond in SIMPLE Marathi.

You are given:
- A free-text complaint from the patient.
- A list of similar past cases from a structured triage dataset, with disease, zone and advice.
- A baseline predicted disease and zone computed from these cases.

Your behavior:

1. Decide the FINAL disease (for internal use only) from:
   {ALLOWED_DISEASES}

   Return it as "internal_disease". This is NOT shown directly to the patient.

2. Decide the FINAL triage zone (patient-facing severity) as exactly one of:
   {ALLOWED_ZONES}

   - "Red"    = emergency / life threatening → needs immediate ER / 108 call.
   - "Orange" = urgent but not instant death risk → needs doctor today / within few hours.
   - "Yellow" = mild / stable → can manage at home + OPD visit if needed.

3. For the patient-facing output, DO NOT scare them with disease names, especially for Red/Orange.
   Instead:
   - Summarize important warning symptoms in Marathi.
   - Focus on zone and what action to take (ER / call 108 / see doctor / home care).

4. Generate the following fields:

   - "internal_disease": string from the allowed disease list (for backend/internal use).
   - "final_zone": "Red" or "Orange" or "Yellow".
   - "patient_symptoms_line": ONE short Marathi sentence describing key symptoms.
   - "patient_action_line": ONE or two short Marathi sentences focusing on what to do now.
   - "followup_question": EITHER one important follow-up question in Marathi
       OR empty string "" if you think no follow-up is needed.

6. IMPORTANT: Output format

   You MUST output ONLY valid JSON with these EXACT keys:
       internal_disease, final_zone, patient_symptoms_line, patient_action_line, followup_question

   Do NOT add any explanation, markdown, or text outside JSON.
"""

    user_context = f"""
PATIENT COMPLAINT (Marathi):
{user_text}

BASELINE PREDICTION (from k-NN over dataset):
- predicted_disease: {predicted_disease}
- predicted_zone: {predicted_zone}
- representative_symptoms: {representative_symptoms}
- representative_advice: {representative_advice}

SIMILAR PAST CASES:
{similar_text}
"""

    full_prompt = system_instructions + "\n" + user_context
    return full_prompt


# =========================
# STEP 4: LLM call using Gemini
# =========================

def call_llm(prompt: str) -> Dict[str, Any]:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw_text = response.text.strip()
    except Exception as e:
        print("Gemini API Error:\n", e)
        raise RuntimeError("Gemini call failed")

    print("\n[DEBUG] Raw Gemini Response:\n", raw_text[:1000])

    try:
        data = json.loads(raw_text)
    except Exception:
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

    expected = [
        "internal_disease",
        "final_zone",
        "patient_symptoms_line",
        "patient_action_line",
        "followup_question",
    ]
    for key in expected:
        if key not in data:
            raise RuntimeError(f"Missing key in Gemini response: {key}")

    return {
        "internal_disease": data["internal_disease"],
        "final_zone": data["final_zone"],
        "patient_symptoms_line": data["patient_symptoms_line"],
        "patient_action_line": data["patient_action_line"],
        "followup_question": data["followup_question"],
    }


# =========================
# STEP 5: End-to-end function
# =========================

def analyze_patient(user_text: str) -> Dict[str, Any]:
    base_pred = predict_disease_zone(user_text, k=10)
    if not base_pred:
        raise RuntimeError("Could not compute base prediction.")

    prompt = build_llm_prompt(base_pred)
    llm_result = call_llm(prompt)

    return {
        "input_text": user_text,
        "baseline_prediction": base_pred,
        "llm_result": llm_result,
    }


if __name__ == "__main__":
    print("\n=== AI Symptom Checker: RAG + Gemini + Qdrant (Step 6) ===")
    print("Enter patient complaint in Marathi.\n")

    user_text = input("Patient complaint: ").strip()
    if not user_text:
        print("No input given. Exiting.")
        raise SystemExit

    full_result = analyze_patient(user_text)

    zone = full_result["llm_result"]["final_zone"]
    symptoms_line = full_result["llm_result"]["patient_symptoms_line"]
    action_line = full_result["llm_result"]["patient_action_line"]

    zone_labels = {
        "Red":   "Zone: 🔴 Red – उच्च धोक्याची पातळी",
        "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
        "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
    }

    print("\n========== FINAL OUTPUT (GEMINI, PATIENT-FACING) ==========")
    print("Patient complaint:", full_result["input_text"], "\n")
    print(zone_labels.get(zone, f"Zone: {zone}"))
    print(symptoms_line)
    print("काय करावे:")
    print(action_line)
    print("\nनोट: हा केवळ प्राथमिक अंदाज आहे, कृपया डॉक्टरांचा सल्ला नक्की घ्या.")
    print("===========================================================\n")
