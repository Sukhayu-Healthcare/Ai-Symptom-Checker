import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "symptom_cases"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Delete old collection if it exists (optional)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,          # MiniLM embedding dimension
        distance=Distance.COSINE
    ),
)

print(f"✅ Collection '{COLLECTION_NAME}' created successfully!")