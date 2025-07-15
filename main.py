from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# Qdrant設定
QDRANT_URL = os.getenv("QDRANT_URL", "https://4045ecb7-a2b4-4770-ab63-2f073a8bbab3.eu-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bwPIRAB_SZ6_Y_7x1bugApmqxJyWRlfbjuDeFjpiDGw")
COLLECTION_NAME = "docs"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 埋め込みモデル
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    context: str

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    vec = model.encode(req.query).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=3,
        with_payload=True
    )
    docs = [hit.payload.get("text", "") for hit in hits]
    context = "\n---\n".join(docs)
    return {"context": context}
