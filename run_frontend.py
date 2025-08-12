import os
import subprocess

frontend_file = 'app.py'

if __name__ == "__main__":
    # Ensure Streamlit secrets directory exists
    secrets_dir = os.path.join(r"rag_chatbot\frontend", ".streamlit")
    os.makedirs(secrets_dir, exist_ok=True)
    secrets_path = os.path.join(secrets_dir, "secrets.toml")
    if not os.path.exists(secrets_path):
        with open(secrets_path, "w", encoding="utf-8") as f:
            f.write('BACKEND_URL = "http://localhost:8000"\n')
    # Launch Streamlit
    subprocess.run(["streamlit", "run", os.path.join(r"rag_chatbot\frontend", frontend_file)])
