from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import Dict, Any
from collections import defaultdict, Counter
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
import requests  # <-- use raw HTTP for Qdrant

from groq import Groq


# =========================
# Lazy singletons (Groq + embedding model)
# =========================

@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    print("Loading SentenceTransformer model (first time)...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("Model loaded!")
    return model


# =========================
# Qdrant config (HTTP)
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. https://xxxx.qdrant.tech
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "symptom_cases"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment.")


def qdrant_search(vector, k: int = 10):
    """
    Low-level HTTP search against Qdrant.
    Uses /collections/{collection}/points/search
    """
    base_url = QDRANT_URL.rstrip("/")
    url = f"{base_url}/collections/{QDRANT_COLLECTION}/points/search"

    headers = {
        "Content-Type": "application/json",
        # Qdrant Cloud uses "api-key" header for auth
        "api-key": QDRANT_API_KEY,
    }

    payload = {
        "vector": vector,
        "limit": k,
        "with_payload": True,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", [])


# =========================
# STEP 2: Retrieval + k-NN prediction (via Qdrant HTTP)
# =========================

def retrieve_similar_cases(query_text: str, k: int = 10):
    """
    Use Qdrant HTTP API to retrieve top-k similar past cases.
    """

    model = get_embedding_model()

    # Encode query to vector
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0].astype("float32")
    query_vector = query_embedding.tolist()

    # Qdrant search (cosine similarity -> higher score = more similar)
    search_result = qdrant_search(query_vector, k=k)

    results = []
    for point in search_result:
        payload = point.get("payload") or {}
        score = float(point.get("score", 0.0))  # similarity

        # convert similarity to distance-style so old weighting logic still works
        distance = 1.0 - score

        case = {
            "utterance": payload.get("utterance", ""),
            "disease": payload.get("disease", ""),
            "zone": payload.get("zone", ""),
            "present_symptoms": payload.get("present_symptoms", ""),
            "advice": payload.get("advice", ""),
            "distance": distance,
            "similarity": score,
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

    # Confidence based on top retrieved similarity
    top_similarity = max(case["similarity"] for case in retrieved)
    confidence = round(top_similarity * 100, 1)
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
        "confidence": confidence,
    }

    return result


# =========================
# STEP 3: Build LLM prompt
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
    "Seizure / Fits",
    "Heart Attack (Myocardial Infarction)",
    "Unconscious / Coma",
    "Stroke (CVA)",
    "Severe Head Injury",
    "Severe Trauma with Bleeding / Fracture",
]

ALLOWED_ZONES = ["Red", "Orange", "Yellow"]

def build_llm_prompt(pred: Dict[str, Any], evaluation_mode: bool = False) -> str:
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
            f"  Similarity: {case['similarity']*100:.2f}%\n"
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

FIRST DECISION (MANDATORY):

Before doing ANY retrieval reasoning or diagnosis,
you MUST first determine whether the user's complaint is medical.

If it is NOT medical, immediately return the JSON below and STOP.
Do NOT continue with any disease reasoning.

{{
  "internal_disease": "",
  "final_zone": "Yellow",
  "patient_symptoms_line": "तुमची तक्रार आरोग्याशी संबंधित वाटत नाही.",
  "patient_action_line": "मी फक्त आरोग्याशी संबंधित लक्षणांचे विश्लेषण करू शकतो. कृपया तुमची आरोग्याशी संबंधित लक्षणे सांगा.",
  "followup_question": ""
}}

Do not retrieve or infer any medical disease for non-medical questions.


You are given:

- The complete conversation history including previous follow-up questions and patient answers.
- A free-text complaint from the patient.
- A retrieval summary generated from vector search.
- A confidence score computed from vector retrieval.
- A list of similar past cases from the structured medical dataset.

Your behavior:

The retrieved cases are strong supporting evidence, but they are NOT always correct.

Your task is to perform a final medical reasoning step before making the diagnosis.

You MUST consider ALL of the following:

1. The patient's current complaint.
2. The complete conversation history including follow-up answers.
3. The retrieved similar cases.
4. The similarity scores.
5. The representative symptoms.
6. The representative advice.
7. The confidence score.

Reasoning Rules:

• First analyse the patient's symptoms independently.

• Then compare them with the retrieved cases.

• If both agree, keep the retrieved disease.

• If the retrieved disease does NOT fully explain the patient's symptoms, but another disease from the allowed list clearly matches better, you MAY override the retrieved disease.

• The retrieved disease is supportive evidence, NOT the final truth.

• Always choose the disease that has the strongest overall medical evidence after combining patient symptoms and retrieved cases.

• Never ignore highly relevant retrieved evidence, but never blindly trust it either.

• PATIENT SAFETY HAS THE HIGHEST PRIORITY.

• If the patient's symptoms clearly indicate a life-threatening emergency, ALWAYS classify the case as "Red" even if the retrieved disease or retrieval summary suggests a lower-risk diagnosis.

• Emergency symptoms include (but are not limited to):
  - Vomiting blood (hematemesis)
  - Black tarry stools
  - Severe uncontrolled bleeding
  - Chest pain with sweating, breathlessness, or pain radiating to the left arm or jaw
  - Sudden one-sided weakness or paralysis
  - Slurred speech or facial drooping
  - Loss of consciousness
  - Seizures
  - Pregnancy with heavy bleeding or severe abdominal pain
  - Severe trauma or severe head injury
  - Severe breathing difficulty

• In these situations, patient safety overrides retrieval similarity and the final_zone MUST be "Red".

• If the emergency is already obvious, DO NOT ask further follow-up questions. Return:
"followup_question": ""

• When neurological symptoms (one-sided weakness, facial drooping, slurred speech, inability to move one side) strongly suggest Stroke, you should prioritize those findings even if retrieval suggests another disease.

• When symptoms strongly match Gastritis / Acid Reflux (burning after meals, acidity, sour belching), prioritize those findings over unrelated retrieved diseases.

Only after this reasoning should you produce the FINAL disease and FINAL triage zone.

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

   - "patient_symptoms_line":
     ONE short Marathi sentence describing the patient's important symptoms.

   - "patient_action_line":
     ONE or two short Marathi sentences explaining what the patient should do next.

     Use natural spoken Marathi.

     Prefer phrases like:
     - "डॉक्टरांचा सल्ला घ्या"
     - "पुरेसे पाणी प्या"
     - "विश्रांती घ्या"
     - "१०८ ला कॉल करा"
     - "जवळच्या रुग्णालयात तातडीने जा"

     Avoid unnatural phrases such as:
     - "डॉक्टरची सल्ला"
     - "हायड्रेशन करा"
     - "ER ला जा"

     Use simple words that ordinary Marathi-speaking patients naturally use.

   - "followup_question":
     EITHER one important follow-up question in Marathi
     OR empty string "" if no follow-up is required.


5. Follow-up Rules

Use the suggested disease and retrieved evidence.

Never ask the same question twice.

Never ask about medicines unless absolutely necessary.

Ask ONLY symptom-related questions.

If enough information is already available,
return:

"followup_question": ""
If the patient's complaint, retrieved evidence, similarity scores, and confidence score already provide sufficient information to confidently classify the disease and triage zone, do NOT ask another follow-up question.

Instead, return:

"followup_question": ""

Avoid unnecessary follow-up questions once the diagnosis is sufficiently supported by the retrieved evidence.

Choose follow-up questions ONLY from the list below.
Never ask a follow-up question that has already been answered in the conversation history.

--------------------------------------------------

Viral Fever
- ताप किती दिवसांपासून आहे?
- खोकला आहे का?
- अंगदुखी आहे का?
- ताप किती आहे?

--------------------------------------------------

Heart Attack
- वेदना डाव्या हातात पसरते का?
- घाम येत आहे का?
- श्वास घेण्यास त्रास होतो का?

--------------------------------------------------

Stroke
- चेहरा वाकडा झाला आहे का?
- बोलण्यात अडचण येते का?
- दोन्ही हात हलवू शकता का?

--------------------------------------------------

Pregnancy Complications
- किती महिन्यांची गरोदर आहात?
- रक्तस्त्राव होत आहे का?
- बाळाच्या हालचाली जाणवत आहेत का?

--------------------------------------------------

Severe Dehydration
- उलट्या होत आहेत का?
- लघवी कमी होत आहे का?
- चक्कर येत आहे का?

--------------------------------------------------

Tuberculosis
- खोकला २ आठवड्यांपेक्षा जास्त आहे का?
- वजन कमी झाले आहे का?
- रात्री घाम येतो का?

--------------------------------------------------

Snake Bite
- साप चावल्याचे दिसले का?
- सूज वाढत आहे का?
- श्वास घेण्यास त्रास होतो का?

--------------------------------------------------

Gastritis / Acid Reflux

- तिखट किंवा तेलकट अन्न खाल्ल्यानंतर जळजळ वाढते का?
- आंबट ढेकर येतात का?
- उलट्या होत आहेत का?

--------------------------------------------------

Head Injury
- शुद्ध गेली होती का?
- उलटी झाली का?
- चक्कर येते का?

--------------------------------------------------
If the patient's latest answer resolves the uncertainty, immediately finalize the triage instead of asking another question.
If ALL important questions are already answered,

return

"followup_question": ""

Do NOT invent any other questions.

--------------------------------------------------

SPECIAL EVALUATION MODE

If evaluation_mode is True:

- Do NOT ask any follow-up question.
- Always return:

"followup_question": ""

- Do not spend tokens deciding follow-up questions.
- Focus only on predicting the most accurate disease and triage zone.

--------------------------------------------------

6. IMPORTANT: Output format

   You MUST output ONLY valid JSON with these EXACT keys:
       internal_disease, final_zone, patient_symptoms_line, patient_action_line, followup_question

   Do NOT add any explanation, markdown, or text outside JSON.
"""

    user_context = f"""
    Evaluation Mode:
    {evaluation_mode}

    Confidence Score:
    {pred["best_disease_score"]:.2f}

    PATIENT COMPLAINT (Marathi):
    {user_text}

    RETRIEVAL SUMMARY (generated from vector search)
    Suggested Disease:
    {predicted_disease}

    Suggested Zone:
    {predicted_zone}

    Representative Symptoms:
    {representative_symptoms}

    Representative Advice:
    {representative_advice}

    The retrieved disease and zone are only suggestions based on semantic similarity.

    You MUST independently analyse:

    - the patient's complaint,
    - the complete conversation history,
    - the retrieved similar cases,
    - the representative symptoms,
    - the representative advice,
    - and the confidence score.

    If the retrieved disease strongly matches the patient's symptoms, keep it.

    However, if you find a medically more appropriate diagnosis from the allowed disease list that is better supported by the patient's symptoms and conversation, you may override the retrieved disease.
    Your final diagnosis should maximize overall medical consistency between the patient's symptoms, conversation history, retrieved evidence, and similarity scores. Do not override retrieval unless there is clear clinical evidence supporting another diagnosis.
    Always explain the patient symptoms and action according to your FINAL diagnosis, not the retrieved suggestion.
    """

    full_prompt = system_instructions + "\n" + user_context
    return full_prompt


# =========================
# STEP 4: LLM call using Groq
# =========================

def call_llm(prompt: str) -> Dict[str, Any]:
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq API Error:\n", e)
        raise RuntimeError("Groq call failed")

    print("\n[DEBUG] Raw Groq Response:\n", raw_text[:1000])

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
            raise RuntimeError(f"Missing key in Groq response: {key}")

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

def analyze_patient(user_text: str,evaluation_mode=False) -> Dict[str, Any]:
    base_pred = predict_disease_zone(user_text, k=10)
    if not base_pred:
        raise RuntimeError("Could not compute base prediction.")

    prompt = build_llm_prompt(
        base_pred,
        evaluation_mode=evaluation_mode
    )
    llm_result = call_llm(prompt)

    return {
        "input_text": user_text,
        "baseline_prediction": base_pred,
        "llm_result": llm_result,
    }


if __name__ == "__main__":
    print("\n=== AI Symptom Checker: RAG + Groq + Qdrant (Step 6) ===")
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

    print("\n========== FINAL OUTPUT (GROQ, PATIENT-FACING) ==========")
    print("Patient complaint:", full_result["input_text"], "\n")
    print(zone_labels.get(zone, f"Zone: {zone}"))
    print(symptoms_line)
    print("काय करावे:")
    print(action_line)
    print("\nनोट: हा केवळ प्राथमिक अंदाज आहे, कृपया डॉक्टरांचा सल्ला नक्की घ्या.")
    print("===========================================================\n")

