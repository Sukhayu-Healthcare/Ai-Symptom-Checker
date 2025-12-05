import os
import json

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ==============================
# STEP 1: Load merged dataset
# ==============================

DATA_PATH = os.path.join("artifacts", "merged_symptom_dataset.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find merged dataset at {DATA_PATH}. "
        "Make sure you ran build_knowledge_base_step2.py successfully."
    )

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded!")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# Ensure required columns exist
required_cols = ["utterance", "disease", "zone", "present_symptoms", "advice"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' is missing in the dataset!")

# Drop rows with empty utterance
df["utterance"] = df["utterance"].astype(str)
df = df[df["utterance"].str.strip().ne("")]
print("Rows after dropping empty utterances:", len(df))


# ==============================
# STEP 2: Load embedding model
# ==============================

print("\nLoading sentence-transformer model (this may take some time the first time)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")


# ==============================
# STEP 3: Create text list to embed
# ==============================
# For now we only use 'utterance'. Later we can experiment with combining
# 'utterance' + 'present_symptoms'.

texts = df["utterance"].tolist()
print(f"Total texts to embed: {len(texts)}")


# ==============================
# STEP 4: Compute embeddings
# ==============================

# encode() will automatically batch internally
print("\nEncoding texts into embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")
print("Embeddings shape:", embeddings.shape)  # (n_rows, dim)


# ==============================
# STEP 5: Build FAISS index
# ==============================

dim = embeddings.shape[1]
print("\nCreating FAISS index with dimension:", dim)

index = faiss.IndexFlatL2(dim)  # Simple L2 index (no ANN for now)
index.add(embeddings)

print("FAISS index built!")
print("Index size (number of vectors):", index.ntotal)


# ==============================
# STEP 6: Save index and metadata
# ==============================

os.makedirs("artifacts", exist_ok=True)

index_path = os.path.join("artifacts", "symptom_index.faiss")
faiss.write_index(index, index_path)
print(f"\n✅ Saved FAISS index to: {index_path}")

# Build metadata list (to map from index -> row info)
metadata = []
for _, row in df.iterrows():
    metadata.append({
        "utterance": str(row["utterance"]),
        "disease": str(row["disease"]),
        "zone": str(row["zone"]),
        "present_symptoms": str(row["present_symptoms"]),
        "advice": str(row["advice"]),
    })

metadata_path = os.path.join("artifacts", "symptom_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Saved metadata to: {metadata_path}")

print("\n🎉 STEP 3 DONE: Embeddings + FAISS index created successfully!")
