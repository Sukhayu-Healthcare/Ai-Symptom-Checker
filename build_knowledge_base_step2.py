import os
from glob import glob

import pandas as pd


# ==============================
# STEP 1: Find all Excel files
# ==============================

# Folder where you kept your 15 files
DATA_DIR = "data"

# This will automatically pick ALL .xlsx files in the data/ folder
excel_files = glob(os.path.join(DATA_DIR, "*.xlsx"))

print("Found Excel files:")
for f in excel_files:
    print(" -", f)

if not excel_files:
    raise FileNotFoundError(
        "No .xlsx files found in the 'data' folder. "
        "Make sure your 15 Excel files are inside a folder named 'data' next to this script."
    )


# ==============================
# STEP 2: Load and combine all files
# ==============================

dfs = []  # list of DataFrames

for file_path in excel_files:
    print(f"\nReading file: {file_path}")
    df_part = pd.read_excel(file_path)

    print("  Columns in this file:", list(df_part.columns))
    print("  Rows in this file:", len(df_part))

    dfs.append(df_part)

# Combine all 15 files into a single DataFrame
df = pd.concat(dfs, ignore_index=True)
print("\n======================================")
print("ALL FILES COMBINED")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
print("======================================\n")


# ==============================
# STEP 3: Normalize column names
# (so all files work together)
# ==============================

# Lowercase + strip spaces for safety
df.columns = [c.strip().lower() for c in df.columns]

# Try to map to standard names
column_mapping = {
    "utterance": "utterance",
    "sentence": "utterance",
    "query": "utterance",
    "disease": "disease",
    "diagnosis": "disease",
    "zone": "zone",
    "risk_zone": "zone",
    "present_symptoms": "present_symptoms",
    "symptoms": "present_symptoms",
    "advice": "advice",
    "recommendation": "advice",
}

standard_cols = {
    "utterance": None,
    "disease": None,
    "zone": None,
    "present_symptoms": None,
    "advice": None,
}

for col in df.columns:
    if col in column_mapping:
        std_name = column_mapping[col]
        standard_cols[std_name] = col

print("Mapped columns:")
for std, original in standard_cols.items():
    print(f"  {std:17} <- {original}")

missing = [k for k, v in standard_cols.items() if v is None]
if missing:
    raise ValueError(
        f"\nERROR: Could not find these required columns in any file: {missing}\n"
        "Please check your Excel column names and adjust 'column_mapping' in the script."
    )

# Now build a clean DataFrame with consistent names
df_clean = pd.DataFrame({
    "utterance": df[standard_cols["utterance"]].astype(str),
    "disease": df[standard_cols["disease"]].astype(str),
    "zone": df[standard_cols["zone"]].astype(str),
    "present_symptoms": df[standard_cols["present_symptoms"]].astype(str),
    "advice": df[standard_cols["advice"]].astype(str),
})

print("\nAfter cleaning:")
print(df_clean.head())
print("Total cleaned rows:", len(df_clean))
print("Cleaned columns:", list(df_clean.columns))


# ==============================
# STEP 4: Basic sanity checks
# ==============================

print("\nUnique diseases:", df_clean["disease"].unique())
print("\nUnique zones:", df_clean["zone"].unique())

# Drop rows where utterance is empty/NaN
df_clean = df_clean[df_clean["utterance"].str.strip().ne("")]
print("\nAfter dropping empty utterances, rows:", len(df_clean))


# ==============================
# STEP 5: Save merged dataset
# ==============================

os.makedirs("artifacts", exist_ok=True)
output_path = os.path.join("artifacts", "merged_symptom_dataset.csv")
df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ DONE! Saved merged & cleaned dataset to: {output_path}")

