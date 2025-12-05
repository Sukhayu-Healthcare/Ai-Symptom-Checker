import json
import os
from collections import defaultdict, Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==============================
# STEP 1: Load index, metadata, model
# ==============================

ARTIFACTS_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "symptom_index.faiss")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "symptom_metadata.json")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

print(f"Loading FAISS index from: {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)

print(f"Loading metadata from: {METADATA_PATH}")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Total cases in metadata:", len(metadata))

print("\nLoading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")


# ==============================
# STEP 2: Retrieval function
# (same idea as test_retrieval.py)
# ==============================

def retrieve_similar_cases(query_text: str, k: int = 10):
    """Return top-k similar past cases with distances."""
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for idx, dist in zip(indices, distances):
        case = metadata[idx].copy()
        case["distance"] = float(dist)
        results.append(case)

    return results


# ==============================
# STEP 3: Simple k-NN style predictor
# ==============================

def predict_disease_zone(user_text: str, k: int = 10):
    """
    Use retrieved cases to predict:
      - most likely disease
      - most likely zone
      - representative advice
    """

    retrieved = retrieve_similar_cases(user_text, k=k)

    # --- 3A. Aggregate scores per disease using distance ---
    disease_scores = defaultdict(float)
    disease_examples = defaultdict(list)

    # Small epsilon to avoid division by zero
    eps = 1e-6

    for case in retrieved:
        d = case["disease"]
        dist = case["distance"]

        # Smaller distance = more similar -> higher weight
        weight = 1.0 / (dist + eps)

        disease_scores[d] += weight
        disease_examples[d].append(case)

    if not disease_scores:
        return None

    # --- 3B. Pick best disease by total score ---
    best_disease = max(disease_scores.items(), key=lambda x: x[1])[0]
    best_disease_score = disease_scores[best_disease]

    # --- 3C. Among examples of that disease, choose zone by majority vote ---
    zones = [c["zone"] for c in disease_examples[best_disease]]
    zone_counter = Counter(zones)
    best_zone = zone_counter.most_common(1)[0][0]

    # --- 3D. Choose a representative advice (closest example of that disease) ---
    best_example = sorted(
        disease_examples[best_disease],
        key=lambda c: c["distance"]
    )[0]

    representative_advice = best_example["advice"]
    representative_symptoms = best_example["present_symptoms"]

    # --- 3E. Build result object ---
    result = {
        "user_text": user_text,
        "predicted_disease": best_disease,
        "predicted_zone": best_zone,
        "representative_advice": representative_advice,
        "representative_symptoms": representative_symptoms,
        "supporting_cases": retrieved,        # all k retrieved
        "disease_scores": dict(disease_scores),
        "raw_zone_counts": dict(zone_counter),
        "best_disease_score": best_disease_score,
    }

    return result


# ==============================
# STEP 4: Manual test
# ==============================

if __name__ == "__main__":
    print("\n=== AI Symptom Checker: k-NN RAG Inference Test ===")
    print("Enter patient complaint in Marathi (free text).")
    print("Example: मला कालपासून खूप ताप आणि अंगदुखी आहे.\n")

    user_text = input("Patient complaint: ").strip()
    if not user_text:
        print("No input provided. Exiting.")
        exit()

    k = 10
    print(f"\nAnalyzing with top {k} similar cases...\n")
    prediction = predict_disease_zone(user_text, k=k)

    if prediction is None:
        print("Could not make a prediction. (No retrieved cases?)")
        exit()

    print("============== PREDICTION ==============")
    print("Patient text: ", prediction["user_text"])
    print("Predicted disease:", prediction["predicted_disease"])
    print("Predicted zone:   ", prediction["predicted_zone"])
    print("Representative symptoms from similar case:", prediction["representative_symptoms"])
    print("Suggested advice:", prediction["representative_advice"])
    print("========================================\n")

    print("TOP SUPPORTING CASES (from retrieval):\n")
    for i, case in enumerate(prediction["supporting_cases"], start=1):
        print(f"---- Similar Case {i} ----")
        print("Distance: ", f"{case['distance']:.4f}")
        print("Disease:  ", case["disease"])
        print("Zone:     ", case["zone"])
        print("Utterance:", case["utterance"])
        print("Advice:   ", case["advice"])
        print()
