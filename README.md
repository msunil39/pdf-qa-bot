# PDF Q&A Bot

A local RAG application that lets you upload a PDF and ask questions about it.

The PDF is processed locally, its text is stored as embeddings in ChromaDB, and Ollama is used to generate the answers. No OpenAI API or other cloud service is required.

## Features

* Upload PDF files
* Extract and split PDF text into chunks
* Create embeddings locally
* Store embeddings in ChromaDB
* Ask questions about uploaded documents
* Use Mistral through Ollama for answers
* Support multiple documents
* Everything runs locally

## Project Structure

```text id="7p5f1m"
pdf-qa-bot/
├── app/
│   ├── parser.py
│   ├── vector_store.py
│   ├── llm.py
│   └── main.py
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

You need Python and Ollama installed.

Install Ollama and download the Mistral model:

```bash id="g7e0u9"
ollama pull mistral
```

Start Ollama:

```bash id="j6zq3r"
ollama serve
```

## Setup

Clone the repository:

```bash id="t3k6b1"
git clone <repo-url>
cd pdf-qa-bot
```

Create a virtual environment:

```bash id="5s7d2a"
python -m venv venv
```

Windows:

```bash id="g2w8zq"
venv\Scripts\activate
```

Mac/Linux:

```bash id="c4v8ys"
source venv/bin/activate
```

Install the dependencies:

```bash id="5kq9pl"
pip install -r requirements.txt
```

Start the API:

```bash id="n6t1vx"
uvicorn app.main:app --reload
```

Open the API docs:

```text id="m0g5fe"
http://127.0.0.1:8000/docs
```

## How to Use

Upload a PDF:

```bash id="x5b3rz"
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@my_document.pdf"
```

The response gives you a `doc_id`.

Use that ID to ask a question:

```bash id="e8j2wn"
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "your-doc-id",
    "question": "What are the main topics covered in this document?"
  }'
```

Other available endpoints:

```text id="3s0v4c"
GET     /health/ollama
GET     /documents
DELETE  /documents/{doc_id}
```

## Tests

```bash id="7b1w5r"
pytest -v
```

## Tech Used

* Python
* FastAPI
* PyMuPDF
* sentence-transformers
* ChromaDB
* Ollama
* Mistral
* Pytest

## Why I Built This

I wanted to understand how a basic RAG application works without relying on a cloud API. This project gave me a chance to work with PDF processing, embeddings, vector search, and a local LLM in one application.
