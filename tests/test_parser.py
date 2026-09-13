import pytest
from app.parser import clean_text, chunk_text

SAMPLE_TEXT = (
    "Python is a high-level programming language. "
    "It is widely used in data science and machine learning. "
    "FastAPI is a modern web framework for building APIs with Python. "
    "Kafka is a distributed streaming platform used for real-time data pipelines. "
    "Elasticsearch is a distributed search and analytics engine. "
    "DuckDB is an in-process SQL OLAP database management system. "
    "Apache Airflow is a platform to programmatically author workflows. "
    "Docker is used to containerize applications for consistent deployment. "
    "Git is a distributed version control system widely used by developers. "
    "Linux is the most popular operating system for servers and cloud computing. "
) * 5


def test_clean_text_strips_whitespace():
    result = clean_text("  hello   world  ")
    assert result == "hello world"


def test_clean_text_collapses_newlines():
    result = clean_text("line1\n\n\n\nline2")
    assert result.count("\n") <= 2


def test_chunk_text_returns_list():
    chunks = chunk_text(SAMPLE_TEXT)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_has_required_keys():
    chunks = chunk_text(SAMPLE_TEXT)
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "char_count" in chunk


def test_chunk_text_respects_size():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=200, overlap=50)
    for chunk in chunks:
        assert chunk["char_count"] <= 250


def test_chunk_text_ids_sequential():
    chunks = chunk_text(SAMPLE_TEXT)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(ids)))


def test_chunk_text_empty_input():
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_text_short_input():
    chunks = chunk_text("Short text.")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0


def test_chunk_text_overlap():
    text = "A" * 1000
    chunks_no_overlap = chunk_text(text, chunk_size=200, overlap=0)
    chunks_overlap    = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks_overlap) >= len(chunks_no_overlap)
