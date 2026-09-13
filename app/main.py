import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.parser import extract_text, chunk_text
from app.vector_store import embed_and_store, retrieve, delete_document, list_documents
from app.llm import ask, check_ollama_health
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Q&A Bot API",
    description=(
        "Upload any PDF and ask questions about it. "
        "Uses sentence-transformers for embeddings, ChromaDB for retrieval, "
        "and Ollama (Mistral) as the local LLM. 100% offline - no API key needed."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: Optional[int] = 4


class UploadResponse(BaseModel):
    doc_id:        str
    filename:      str
    total_chars:   int
    chunks_stored: int
    message:       str


class AskResponse(BaseModel):
    doc_id:               str
    question:             str
    answer:               str
    model:                str
    context_chunks_used:  int


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "PDF Q&A Bot is running"}


@app.get("/health/ollama", tags=["Health"])
def ollama_health():
    return check_ollama_health()


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(..., description="PDF file to upload")):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    text = extract_text(pdf_bytes)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF")

    chunks   = chunk_text(text)
    doc_id   = str(uuid.uuid4())[:8]
    stored   = embed_and_store(doc_id, chunks)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        total_chars=len(text),
        chunks_stored=stored,
        message=f"PDF processed. Use doc_id '{doc_id}' to ask questions.",
    )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    context_chunks = retrieve(
        doc_id=request.doc_id,
        query=request.question,
        top_k=request.top_k or 4,
    )

    if not context_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with doc_id '{request.doc_id}'. Upload a PDF first."
        )

    result = ask(context_chunks, request.question)

    return AskResponse(
        doc_id=request.doc_id,
        question=request.question,
        answer=result["answer"],
        model=result["model"],
        context_chunks_used=result["context_chunks_used"],
    )


@app.get("/documents", tags=["Documents"])
def get_documents():
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_doc(doc_id: str):
    delete_document(doc_id)
    return {"message": f"Document '{doc_id}' deleted successfully"}
