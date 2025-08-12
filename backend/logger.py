import os
import time
import json

DATA_DIR = os.getenv("RAG_DATA_DIR", "./rag_data")
LOG_PATH = os.path.join(DATA_DIR, "logs.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

def log_interaction(entry: dict):
    entry["timestamp"] = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
