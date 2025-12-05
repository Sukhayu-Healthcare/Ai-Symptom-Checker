import os
import json
import time
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException


# ---------- CONFIG ----------

ARTIFACTS_DIR = "artifacts"
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "symptom_metadata.json")
EMB_PATH = os.path.join(ARTIFACTS_DIR, "symptom_embeddings.npy")

COLLECTION_NAME = "symptom_cases"

# small batch size to be safe
BATCH_SIZE = 50

# retry settings
MAX_RETRIES = 5
RETRY_SLEEP_SEC = 3


def main():
    # 1) Load local metadata + embeddings
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMB_PATH}")

    print(f"Loading metadata from: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loading embeddings from: {EMB_PATH}")
    embeddings = np.load(EMB_PATH).astype("float32")

    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Metadata length and embeddings count do not match!")

    dim = embeddings.shape[1]
    total = embeddings.shape[0]
    print(f"Embeddings shape: {embeddings.shape} (dim = {dim})")

    # OPTIONAL: for testing, limit to first N points
    # N = 10000
    # metadata = metadata[:N]
    # embeddings = embeddings[:N]
    # total = embeddings.shape[0]
    # print(f"Limiting upload to first {total} points for testing.")

    # 2) Connect to Qdrant (cloud)
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "Please set QDRANT_URL and QDRANT_API_KEY in your environment."
        )

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60.0,  # longer timeout
    )

    print(f"Assuming collection '{COLLECTION_NAME}' already exists in Qdrant.")
    print(f"Uploading {total} points in batches of {BATCH_SIZE}...\n")

    # 3) Upload in batches with retry
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_vectors = embeddings[start:end]
        batch_meta = metadata[start:end]

        points = []
        for local_idx, (vec, meta) in enumerate(zip(batch_vectors, batch_meta)):
            idx = start + local_idx  # global ID
            payload = {
                "utterance": meta["utterance"],
                "disease": meta["disease"],
                "zone": meta["zone"],
                "present_symptoms": meta.get("present_symptoms", ""),
                "advice": meta.get("advice", ""),
            }
            points.append(
                PointStruct(
                    id=idx,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        attempt = 1
        while True:
            try:
                print(
                    f"  -> Uploading points {start} to {end - 1} "
                    f"(attempt {attempt})..."
                )
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
                # success → break out of retry loop
                break
            except ResponseHandlingException as e:
                print(f"     !! Qdrant ResponseHandlingException: {e}")
            except Exception as e:
                print(f"     !! Other error: {e}")

            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(
                    f"Giving up on batch {start}-{end-1} after {MAX_RETRIES} retries."
                )

            print(f"     Waiting {RETRY_SLEEP_SEC} seconds before retry...")
            time.sleep(RETRY_SLEEP_SEC)

    print("\n✅ Done! Uploaded all points to Qdrant.")


if __name__ == "__main__":
    main()
