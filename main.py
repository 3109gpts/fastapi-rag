import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ▶ 環境変数からQdrantの接続情報を取得
QDRANT_URL = "https://4045ecb7-a2b4-4770-ab63-2f073a8bbab3.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bwPIRAB_SZ6_Y_7x1bugApmqxJyWRlfbjuDeFjpiDGw"
COLLECTION_NAME = "docs"

app = FastAPI()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

class QueryRequest(BaseModel):
    query: str

@app.post("/vector_search")
def vector_search(req: QueryRequest):
    vec = model.encode(req.query).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=3
    )
    return [hit.payload for hit in hits]
