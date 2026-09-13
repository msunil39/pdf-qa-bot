import pytest
from unittest.mock import patch, MagicMock
from app.llm import build_prompt, ask, check_ollama_health


def test_build_prompt_contains_question():
    prompt = build_prompt(["Some context here."], "What is Python?")
    assert "What is Python?" in prompt


def test_build_prompt_contains_context():
    prompt = build_prompt(["FastAPI is a web framework."], "What is FastAPI?")
    assert "FastAPI is a web framework." in prompt


def test_build_prompt_multiple_chunks():
    chunks = ["Chunk one content.", "Chunk two content."]
    prompt = build_prompt(chunks, "Tell me more.")
    assert "Chunk one content." in prompt
    assert "Chunk two content." in prompt


def test_ask_empty_context():
    result = ask([], "What is Kafka?")
    assert "No relevant context" in result["answer"]
    assert result["context_chunks_used"] == 0


def test_ask_ollama_not_running():
    with patch("app.llm.requests.post") as mock_post:
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        result = ask(["Some context"], "What is this?")
        assert "not running" in result["answer"].lower() or "error" in result["answer"].lower()


def test_ask_successful_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Python is a programming language."}
    mock_response.raise_for_status = MagicMock()

    with patch("app.llm.requests.post", return_value=mock_response):
        result = ask(["Python context here."], "What is Python?")
        assert result["answer"] == "Python is a programming language."
        assert result["context_chunks_used"] == 1


def test_check_ollama_health_not_running():
    with patch("app.llm.requests.get") as mock_get:
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        result = check_ollama_health()
        assert result["ollama_running"] is False


def test_check_ollama_health_running():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "mistral:latest"}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("app.llm.requests.get", return_value=mock_response):
        result = check_ollama_health()
        assert result["ollama_running"] is True
        assert result["model_ready"] is True
