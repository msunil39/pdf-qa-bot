import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import os
import shutil

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR       = os.getenv("CHROMA_DIR", "./chroma_db")

_embedder = None
_client   = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_or_create_collection(doc_id: str):

    client = get_client()

    safe_id = "doc-" + "".join(c if c.isalnum() else "-" for c in doc_id)[:60]
    return client.get_or_create_collection(
        name=safe_id,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(doc_id: str, chunks: list[dict]) -> int:

    if not chunks:
        return 0

    embedder   = get_embedder()
    collection = get_or_create_collection(doc_id)

    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    ids        = [f"{doc_id}_chunk_{c['chunk_id']}" for c in chunks]
    metadatas  = [{"chunk_id": c["chunk_id"], "char_count": c["char_count"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks for doc_id='{doc_id}'")
    return len(chunks)


def retrieve(doc_id: str, query: str, top_k: int = 4) -> list[str]:

    embedder   = get_embedder()
    collection = get_or_create_collection(doc_id)

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    docs = results.get("documents", [[]])[0]
    logger.info(f"Retrieved {len(docs)} chunks for query: '{query[:60]}'")
    return docs


def delete_document(doc_id: str):

    client = get_client()
    safe_id = "doc-" + "".join(c if c.isalnum() else "-" for c in doc_id)[:60]
    try:
        client.delete_collection(safe_id)
        logger.info(f"Deleted collection for doc_id='{doc_id}'")
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")


def list_documents() -> list[str]:

    client = get_client()
    collections = client.list_collections()
    return [c.name.replace("doc-", "", 1) for c in collections]
