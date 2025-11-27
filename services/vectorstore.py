from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from configs.settings import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
import json

client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)

def create_collection():
    print("Create Collection")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def upsert_embeddings(embeddings, texts):
    print("Writing JSON file...")

    points = []
    print(len(embeddings))
    
    for i, (emb, text) in enumerate(zip(embeddings, texts)):
        points.append({
            "id": i,
            "vector": emb,
            "payload": {"text": text}
        })

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def search(query_embedding, top_k=2):
    print("Search Collection")
    
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding[0],
        limit=top_k
    )
    
    return res
