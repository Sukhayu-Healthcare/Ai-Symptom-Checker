import random
import time
import json

import pandas as pd

from rag_with_llm_step6 import analyze_patient

#Load dataset
DATASET_PATH = "artifacts/merged_symptom_dataset.csv"

SAMPLE_SIZE = 500

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Total rows : {len(df)}")

# Random sample for evaluation
sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)

print(f"Evaluating {len(sample_df)} samples...\n")

#Create Result List 
actual_disease = []
predicted_disease = []

actual_zone = []
predicted_zone = []

case_match_scores = []

response_times = []

#Prediction Loop
print("Running model evaluation...\n")

for i, row in sample_df.iterrows():

    print(f"[{i+1}/{len(sample_df)}] Processing...")

    user_text = str(row["utterance"])

    true_disease = row["disease"]
    true_zone = row["zone"]

    start = time.time()

    try:
        result = analyze_patient(user_text)

        elapsed = time.time() - start

        llm = result["llm_result"]

        predicted = llm["internal_disease"]
        predicted_triage = llm["final_zone"]

        confidence = result["baseline_prediction"]["confidence"]

    except Exception as e:

        print("ERROR:", e)
        continue

    actual_disease.append(true_disease)
    predicted_disease.append(predicted)

    actual_zone.append(true_zone)
    predicted_zone.append(predicted_triage)

    case_match_scores.append(confidence)
    response_times.append(elapsed)

