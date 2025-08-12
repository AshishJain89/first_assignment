# RAG Chatbot — Modular Version

This is a modular Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI** (backend), **Streamlit** (frontend), and **FAISS** (vector store). It supports multiple embedding providers (SentenceTransformers, OpenAI, Cohere) and can be easily extended to Pinecone, Qdrant, or Weaviate.

---

## 📂 Project Structure
```
rag_chatbot/
├── backend/
│   ├── main.py           # FastAPI app entry point
│   ├── routes.py         # API endpoints for ingestion and querying
│   ├── vectorstore.py    # FAISS vector index operations
│   ├── embeddings.py     # Embedding helpers
│   ├── processing.py     # PDF parsing and text chunking
│   ├── generator.py      # LLM-based and fallback answer generation
│   ├── logger.py         # Query/response logging
├── frontend/
│   └── app.py            # Streamlit UI for user interaction
├── run_backend.py        # Shortcut to start backend
├── run_frontend.py       # Shortcut to start frontend
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (dummy values included)
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/AshishJain89/rag_chatbot.git
cd rag_chatbot
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
   - Copy `.env` to your project root
   - Add real API keys if using OpenAI/Cohere

---

## 🚀 Running the Chatbot

### 1️⃣ Start the Backend (FastAPI)
```bash
python run_backend.py
```
Backend will be available at: **http://localhost:8000**

### 2️⃣ Start the Frontend (Streamlit)
```bash
python run_frontend.py
```
Frontend will be available at: **http://localhost:8501**

---

## 📄 Usage

### Upload PDF (via Frontend)
- Go to the **Upload PDF** section in the frontend.
- Select a `.pdf` file.
- Click **Ingest PDF to backend** — the document is parsed, chunked, and stored in the FAISS index.

### Ask Questions (via Frontend)
- Enter your query in the text box.
- Click **Ask**.
- The system retrieves relevant chunks, reranks them, and generates an answer.

---

## 📡 API Examples

### Ingest PDF
**Using `curl`:**
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@example.pdf" \
     -F "source=example.pdf"
```

**Using Python `requests`:**
```python
import requests

files = {"file": ("example.pdf", open("example.pdf", "rb"), "application/pdf")}
resp = requests.post("http://localhost:8000/ingest", files=files)
print(resp.json())
```

### Query
**Using `curl`:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?", "top_k": 5}'
```

**Using Python `requests`:**
```python
import requests

payload = {"query": "What is this document about?", "top_k": 5}
resp = requests.post("http://localhost:8000/query", json=payload)
print(resp.json())
```

---

## 🔧 Configuration

Modify `.env` to change settings:
```
EMBEDDING_PROVIDER=sentence-transformers  # or openai, cohere
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_DATA_DIR=./rag_data
```

---

## 📦 Extending

- Replace FAISS with Pinecone, Qdrant, or Weaviate in `vectorstore.py`.
- Add additional LLM backends in `generator.py`.
- Improve PDF parsing and chunking in `processing.py`.

---

## 📝 Logging

All queries, answers, and retrieved contexts are logged in:
```
rag_data/logs.jsonl
```

---

## ⚠️ Notes
- This is **not** production-ready; add authentication, error handling, and scaling mechanisms for production.
- Avoid storing API keys in plain `.env` in production — use a secrets manager.
