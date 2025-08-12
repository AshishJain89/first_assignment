import os, sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def generate_answer_groq(query: str, context_chunks, model="meta-llama/llama-4-maverick-17b-128e-instruct") -> str:
    if groq_client is None:
        raise RuntimeError("Groq API key not set")
    prompt = "You are a helpful assistant. Use the provided context to answer the question.\\n\\n"
    for i, c in enumerate(context_chunks):
        prompt += f"[CONTEXT {i}] {c['text']}\\n--\\n"
    prompt += f"\\nQUESTION: {query}\\n\\nAnswer concisely and cite contexts by [CONTEXT i]."
    
    completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    content = completion.choices[0].message.content
    return content if content is not None else ""


def generate_answer_fallback(query: str, context_chunks) -> str:
    combined = "\n\n".join([c["text"] for c in context_chunks[:5]])
    return (
        f"[Fallback answer]\nUsing the top {min(5, len(context_chunks))} retrieved chunks as context:\n\n{combined}\n\n"
        f"QUESTION: {query}\n\nSUGGESTED ANSWER: Based on the provided contexts, here is a concise answer."
    )

def generate_answer(query: str, context_chunks):
    try:
        if groq_client is not None:
            return generate_answer_groq(query, context_chunks)
    except Exception as e:
        print("Groq generation failed:", e, file=sys.stderr)
    return generate_answer_fallback(query, context_chunks)

