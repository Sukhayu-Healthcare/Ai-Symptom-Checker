import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==============================
# STEP 1: Load FAISS index + metadata + model
# ==============================

ARTIFACTS_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "symptom_index.faiss")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "symptom_metadata.json")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Did you run build_vector_index_step3.py?")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}.")

print(f"Loading FAISS index from: {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)

print(f"Loading metadata from: {METADATA_PATH}")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Total metadata records:", len(metadata))

print("\nLoading embedding model (same as Step 3)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")


# ==============================
# STEP 2: Define retrieval function
# ==============================

def retrieve_similar_cases(query_text: str, k: int = 5):
    """
    Given a patient's complaint (Marathi text),
    return top-k most similar past cases from the dataset.
    """
    # 1. Encode query
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype("float32")  # shape: (1, dim)

    # 2. Search in FAISS index
    distances, indices = index.search(query_embedding, k)  # distances: (1, k), indices: (1, k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for idx, dist in zip(indices, distances):
        case = metadata[idx].copy()
        case["distance"] = float(dist)
        results.append(case)

    return results


# ==============================
# STEP 3: Simple manual test
# ==============================

if __name__ == "__main__":
    print("\n=== AI Symptom Checker: Retrieval Test ===")
    print("Enter a patient's complaint in Marathi.")
    print("Example: मला कालपासून खूप ताप आणि अंगदुखी आहे.\n")

    # You can use input() for interactive testing:
    user_query = input("Patient complaint (Marathi): ").strip()

    if not user_query:
        print("No input given. Exiting.")
        exit()

    top_k = 5
    print(f"\nSearching for top {top_k} similar past cases...\n")

    results = retrieve_similar_cases(user_query, k=top_k)

    for i, r in enumerate(results, start=1):
        print(f"----------- Result {i} -----------")
        print("Similarity (distance):", f"{r['distance']:.4f}")
        print("Utterance:", r["utterance"])
        print("Disease:", r["disease"])
        print("Zone:", r["zone"])
        print("Present symptoms:", r["present_symptoms"])
        print("Advice:", r["advice"])
        print("---------------------------------\n")
