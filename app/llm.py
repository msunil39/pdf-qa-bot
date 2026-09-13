import requests
import os
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided document context.

Rules:
- Answer only from the context provided below.
- If the answer is not in the context, say "I could not find this information in the document."
- Be concise and accurate.
- Do not make up information.
"""


def build_prompt(context_chunks: list[str], question: str) -> str:
    """Build the full prompt from retrieved context and user question."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Context from document:
{context}

Question: {question}

Answer:"""


def ask(context_chunks: list[str], question: str) -> dict:

    if not context_chunks:
        return {
            "answer": "No relevant context found in the document for this question.",
            "model": OLLAMA_MODEL,
            "context_chunks_used": 0,
        }

    prompt = build_prompt(context_chunks, question)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data   = response.json()
        answer = data.get("response", "").strip()

        return {
            "answer":               answer,
            "model":                OLLAMA_MODEL,
            "context_chunks_used":  len(context_chunks),
        }

    except requests.exceptions.ConnectionError:
        logger.error("Ollama not running. Start with: ollama serve")
        return {
            "answer": "Error: Ollama is not running. Please start it with `ollama serve`.",
            "model":  OLLAMA_MODEL,
            "context_chunks_used": 0,
        }
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return {
            "answer": f"Error communicating with Ollama: {str(e)}",
            "model":  OLLAMA_MODEL,
            "context_chunks_used": 0,
        }


def check_ollama_health() -> dict:

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_ready = any(OLLAMA_MODEL in m for m in models)
        return {
            "ollama_running": True,
            "model":          OLLAMA_MODEL,
            "model_ready":    model_ready,
            "available_models": models,
        }
    except Exception as e:
        return {
            "ollama_running": False,
            "model":          OLLAMA_MODEL,
            "model_ready":    False,
            "error":          str(e),
        }
