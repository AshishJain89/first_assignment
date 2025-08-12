from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from backend import embeddings, vectorstore, generator, logger
from backend import processing  # Import processing separately if it exists as backend/processing.py

router = APIRouter()

class IngestResponse(BaseModel):
    ingested_chunks: int
    message: str

@router.on_event("startup")
def startup_event():
    vectorstore.load_index()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), source: Optional[str] = Form(None)):
    """Ingest a PDF file and store embeddings in the vector store."""
    content = await file.read()
    text = processing.pdf_to_text(content)
    chunks = processing.chunk_text(text)
    if not chunks:
        return {"ingested_chunks": 0, "message": "No text extracted"}
    embs = embeddings.embed_texts(chunks)
    metas = [{"text": chunks[i], "source": source or file.filename} for i in range(len(chunks))]
    vectorstore.add_vectors(embs, metas)
    return {"ingested_chunks": len(chunks), "message": f"Indexed {len(chunks)} chunks"}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@router.post("/query")
async def query_endpoint(req: QueryRequest):
    q_emb = embeddings.embed_texts([req.query])[0]
    retrieved = vectorstore.search_vectors(q_emb, top_k=req.top_k * 2)
    reranked = vectorstore.rerank(req.query, retrieved)[:req.top_k]
    answer = generator.generate_answer(req.query, reranked)
    logger.log_interaction({
        "query": req.query,
        "top_k": req.top_k,
        "retrieved": [{"id": r.get("id"), "score": r.get("score"), "source": r.get("source")} for r in reranked],
        "answer": answer[:1000]
    })
    return {"answer": answer, "retrieved": reranked}

@router.get("/health")
async def health():
    return {"status": "ok", "vectors": len(vectorstore.metadata)}
