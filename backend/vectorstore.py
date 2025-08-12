import os, json, numpy as np
from backend.embeddings import embed_texts, _normalize
import faiss


DATA_DIR = os.getenv("RAG_DATA_DIR", "./rag_data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")

EMBEDDING_DIM = 384  # default for all-MiniLM-L6-v2

os.makedirs(DATA_DIR, exist_ok=True)

metadata = []
index = faiss.IndexFlatIP(EMBEDDING_DIM)

def add_vectors(vectors: np.ndarray, metas):
    global index, metadata
    if faiss is None:
        raise RuntimeError("faiss is required but not installed")
    if index is None:
        raise RuntimeError("faiss index is not initialized")
    vectors = _normalize(vectors)
    if vectors is None or len(vectors) == 0:
        raise ValueError("No vectors to add")
    index.add(vectors.astype('float32'))
    start_id = len(metadata)
    for i, m in enumerate(metas):
        metadata.append({"id": start_id + i, **m})
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index():
    global index, metadata
    if faiss is None:
        return
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"Loaded index with {len(metadata)} vectors")
        except Exception as e:
            print("Failed to load index, starting fresh:", e)

def search_vectors(query_vec: np.ndarray, top_k: int = 5):
    if faiss is None:
        raise RuntimeError("faiss is required but not installed")
    q = _normalize(query_vec.reshape(1, -1)).astype('float32')
    D, I = index.search(q, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results

def rerank(query: str, candidates):
    if not candidates:
        return []
    texts = [c["text"] for c in candidates]
    emb_q = embed_texts([query])
    emb_c = embed_texts(texts)
    emb_qn = _normalize(emb_q)
    emb_cn = _normalize(emb_c)
    sims = (emb_cn @ emb_qn.T).squeeze().tolist()
    for c, s in zip(candidates, sims):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
