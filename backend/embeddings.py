import os, numpy as np
from sentence_transformers import SentenceTransformer

try:
    import openai
except ImportError:
    openai = None

try:
    import cohere
except ImportError:
    cohere = None

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")

# Initialize default sentence-transformers model
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and openai is not None:
    openai.api_key = OPENAI_API_KEY

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY and cohere is not None:
    co_client = cohere.Client(COHERE_API_KEY)
else:
    co_client = None

def embed_texts(texts):
    provider = EMBEDDING_PROVIDER.lower()
    if provider == "sentence-transformers":
        embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    elif provider == "openai" and openai is not None and OPENAI_API_KEY:
        resp = openai.embeddings.create(input=texts, model="text-embedding-ada-002")
        embs = np.array([d.embedding for d in resp.data])
    elif provider == "cohere" and co_client is not None:
        resp = co_client.embed(texts=texts, model="embed-english-v2.0")
        embs = np.array(resp.embeddings)
    else:
        raise RuntimeError(f"Unknown or unconfigured embedding provider: {EMBEDDING_PROVIDER}")
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs

def _normalize(np_arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(np_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np_arr / norms
