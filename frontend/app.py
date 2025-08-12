import streamlit as st
import requests

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.title("RAG Chatbot — Streamlit UI")

st.header("Upload PDF")
uploaded = st.file_uploader("Upload PDF to ingest", type=["pdf"])
if uploaded is not None:
    if st.button("Ingest PDF to backend"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        resp = requests.post(f"{BACKEND_URL}/ingest", files=files)
        st.write(resp.json())

st.write("---")

st.header("Ask a Question")
query = st.text_input("Your question:")
if st.button("Ask") and query:
    payload = {"query": query, "top_k": 5}
    resp = requests.post(f"{BACKEND_URL}/query", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        st.subheader("Answer")
        st.text(data.get("answer"))
        st.subheader("Retrieved Chunks")
        for r in data.get("retrieved", []):
            st.write(f"Source: {r.get('source')}")
            st.write(r.get("text"))
            st.write("---")
    else:
        st.error(f"Backend error: {resp.status_code}")