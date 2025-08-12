from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router as rag_router
import uvicorn

app = FastAPI(title="RAG Chatbot Backend")

# Allow CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from backend/routes.py
app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
