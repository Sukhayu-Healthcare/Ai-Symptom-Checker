import time
import json

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from rag_with_llm_step6 import analyze_patient

DATASET_PATH = "artifacts/merged_symptom_dataset.csv"

SAMPLE_SIZE = 50

overall_start = time.time()

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)
print(f"Total rows : {len(df)}")

sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)

print(f"Evaluating {len(sample_df)} random samples...\n")

actual_disease = []
predicted_disease = []

actual_zone = []
predicted_zone = []

response_times = []

#evaluation loop
print("Evaluating Full AI Pipeline...\n")

for i, row in sample_df.iterrows():

    if (i + 1) % 20 == 0:
        print(f"Processed {i+1}/{len(sample_df)} samples...")

    user_text = str(row["utterance"])

    true_disease = row["disease"]
    true_zone = row["zone"]

    start = time.time()

    try:

        result = analyze_patient(
            user_text,
            evaluation_mode=True
        )

        elapsed = time.time() - start

        llm = result["llm_result"]

        predicted = llm["internal_disease"]
        predicted_zone_value = llm["final_zone"]

    except Exception as e:

        print("ERROR:", e)
        continue

    actual_disease.append(true_disease)
    predicted_disease.append(predicted)

    actual_zone.append(true_zone)
    predicted_zone.append(predicted_zone_value)

    response_times.append(elapsed)

#evaluation loop
print("\nFinished LLM Evaluation!\n")

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

avg_response_time = (
    sum(response_times) / len(response_times)
    if response_times else 0
)

print("=" * 50)
print("FULL AI MODEL PERFORMANCE")
print("=" * 50)

print(f"Disease Accuracy : {disease_accuracy*100:.2f}%")
print(f"Zone Accuracy    : {zone_accuracy*100:.2f}%")
print(f"Precision        : {precision*100:.2f}%")
print(f"Recall           : {recall*100:.2f}%")
print(f"F1 Score         : {f1*100:.2f}%")
print(f"Average Response Time : {avg_response_time:.3f} sec")

#evaluation loop:
results = {

    "evaluation_type": "Retrieval + Groq",

    "dataset_size": len(df),

    "samples_evaluated": len(sample_df),

    "disease_accuracy": round(disease_accuracy*100,2),

    "zone_accuracy": round(zone_accuracy*100,2),

    "precision": round(precision*100,2),

    "recall": round(recall*100,2),

    "f1_score": round(f1*100,2),

    "average_response_time": round(avg_response_time,3)
}

with open("llm_evaluation_results.json","w") as f:
    json.dump(results,f,indent=4)

print("\nSaved llm_evaluation_results.json")

#Confusion Matrix 
labels = sorted(list(set(actual_disease + predicted_disease)))

cm = confusion_matrix(
    actual_disease,
    predicted_disease,
    labels=labels
)

plt.figure(figsize=(12,10))

plt.imshow(cm, interpolation="nearest")

plt.title("LLM Confusion Matrix")

plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)

plt.yticks(range(len(labels)), labels, fontsize=8)

plt.xlabel("Predicted Disease")

plt.ylabel("Actual Disease")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "llm_confusion_matrix.png",
    dpi=300
)
plt.close()

print("Saved llm_confusion_matrix.png")

overall_end = time.time()

print("\nEvaluation Complete!")

print(f"Dataset Size      : {len(df)}")

print(f"Samples Evaluated : {len(sample_df)}")

print(f"Total Time : {(overall_end-overall_start)/60:.2f} minutes")