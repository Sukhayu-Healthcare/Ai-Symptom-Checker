from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt

import time
import json

import pandas as pd

from rag_with_llm_step6 import predict_disease_zone

# ============================================
# Configuration
# ============================================

DATASET_PATH = "artifacts/merged_symptom_dataset.csv"

SAMPLE_SIZE = 5000

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Total rows: {len(df)}")

sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)

print(f"Evaluating {len(sample_df)} random samples...\n")

# ============================================
# Result storage
# ============================================

actual_disease = []
predicted_disease = []

actual_zone = []
predicted_zone = []

case_match_scores = []

response_times = []

# ============================================
# Evaluate Retrieval
# ============================================

print("Evaluating Retrieval Model...\n")

for i, row in sample_df.iterrows():

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(sample_df)} samples...")

    user_text = str(row["utterance"])

    true_disease = row["disease"]
    true_zone = row["zone"]

    start = time.time()

    try:

        result = predict_disease_zone(user_text, k=10)

        elapsed = time.time() - start

    except Exception as e:

        print("ERROR:", e)
        continue

    actual_disease.append(true_disease)
    predicted_disease.append(result["predicted_disease"])

    actual_zone.append(true_zone)
    predicted_zone.append(result["predicted_zone"])

    case_match_scores.append(result["confidence"])

    response_times.append(elapsed)

print("\nFinished Retrieval Evaluation!\n")

# ============================================
# Metrics
# ============================================

disease_accuracy = accuracy_score(
    actual_disease,
    predicted_disease
)

zone_accuracy = accuracy_score(
    actual_zone,
    predicted_zone
)

precision, recall, f1, _ = precision_recall_fscore_support(
    actual_disease,
    predicted_disease,
    average="weighted",
    zero_division=0
)

avg_case_match = (
    sum(case_match_scores) / len(case_match_scores)
    if case_match_scores else 0
)

avg_response_time = (
    sum(response_times) / len(response_times)
    if response_times else 0
)

print("=" * 50)
print("RETRIEVAL MODEL PERFORMANCE")
print("=" * 50)

print(f"Disease Accuracy : {disease_accuracy*100:.2f}%")
print(f"Zone Accuracy    : {zone_accuracy*100:.2f}%")
print(f"Precision        : {precision*100:.2f}%")
print(f"Recall           : {recall*100:.2f}%")
print(f"F1 Score         : {f1*100:.2f}%")
print(f"Average Case Match : {avg_case_match:.2f}%")
print(f"Average Response Time : {avg_response_time:.3f} sec")

#Save results
results = {
    "evaluation_type": "Retrieval Only",
    "dataset_size": len(df),
    "samples_evaluated": len(sample_df),
    "disease_accuracy": round(disease_accuracy*100,2),
    "zone_accuracy": round(zone_accuracy*100,2),
    "precision": round(precision*100,2),
    "recall": round(recall*100,2),
    "f1_score": round(f1*100,2),
    "average_case_match": round(avg_case_match,2),
    "average_response_time": round(avg_response_time,3)
}

with open("retrieval_evaluation_results.json","w") as f:
    json.dump(results,f,indent=4)

print("\nSaved retrieval_evaluation_results.json")

labels = sorted(list(set(actual_disease + predicted_disease)))

cm = confusion_matrix(
    actual_disease,
    predicted_disease,
    labels=labels
)

plt.figure(figsize=(12,10))

plt.imshow(cm, interpolation="nearest")

plt.title("Retrieval Confusion Matrix")

plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)

plt.yticks(range(len(labels)), labels, fontsize=8)

plt.xlabel("Predicted Disease")

plt.ylabel("Actual Disease")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "retrieval_confusion_matrix.png",
    dpi=300
)

print("Saved retrieval_confusion_matrix.png")
print("\nEvaluation Complete!")
print(f"Dataset Size      : {len(df)}")
print(f"Samples Evaluated : {len(sample_df)}")
print("Results saved successfully.")

