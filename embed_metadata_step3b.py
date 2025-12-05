import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import math

ARTIFACTS_DIR = "artifacts"
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "symptom_metadata.json")
EMB_PATH = os.path.join(ARTIFACTS_DIR, "symptom_embeddings.npy")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

print(f"Loading metadata from: {METADATA_PATH}")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

utterances = [m["utterance"] for m in metadata]
n = len(utterances)
print(f"Total utterances to embed: {n}")

print("\nLoading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")

batch_size = 64
num_batches = math.ceil(n / batch_size)

all_embeddings = []

print(f"\nStarting batched encoding with batch_size={batch_size}, total_batches={num_batches}...\n")

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, n)
    batch_texts = utterances[start:end]

    print(f"Encoding batch {batch_idx + 1}/{num_batches} (rows {start}–{end})...")
    batch_emb = model.encode(batch_texts, convert_to_numpy=True)
    all_embeddings.append(batch_emb)

print("\nConcatenating all batch embeddings...")
embeddings = np.vstack(all_embeddings)
print("Final embeddings shape:", embeddings.shape)

print(f"\nSaving embeddings to: {EMB_PATH}")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
np.save(EMB_PATH, embeddings)

print("\n✅ Done! Embeddings saved successfully.")

