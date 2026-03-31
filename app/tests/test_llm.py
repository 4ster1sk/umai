import os
from unittest.mock import MagicMock, patch

import pytest

SAMPLE_DATA = {
    "おやつ": [
        {"date": "2025-10-16", "item": "赤福"},
        {"date": "2025-11-22", "item": "福砂屋 カステラ"},
    ]
}


def _collect(gen) -> str:
    return "".join(gen)


def test_anthropic_provider_called(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["おいしい", "傾向です"])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    with patch("anthropic.Anthropic", return_value=mock_client):
        import importlib
        import app.llm as llm_module
        importlib.reload(llm_module)
        result = _collect(llm_module.stream_analyze(SAMPLE_DATA))

    assert result == "おいしい傾向です"
    mock_client.messages.stream.assert_called_once()


def test_openai_provider_called(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    chunk1 = MagicMock()
    chunk1.choices[0].delta.content = "おすすめ"
    chunk2 = MagicMock()
    chunk2.choices[0].delta.content = "です"

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=iter([chunk1, chunk2]))
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_stream

    with patch("openai.OpenAI", return_value=mock_client):
        import importlib
        import app.llm as llm_module
        importlib.reload(llm_module)
        result = _collect(llm_module.stream_analyze(SAMPLE_DATA))

    assert result == "おすすめです"


def test_gemini_provider_called(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    chunk1 = MagicMock()
    chunk1.text = "提案"
    chunk2 = MagicMock()
    chunk2.text = "します"

    mock_model = MagicMock()
    mock_model.generate_content.return_value = iter([chunk1, chunk2])

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        import importlib
        import app.llm as llm_module
        importlib.reload(llm_module)
        result = _collect(llm_module.stream_analyze(SAMPLE_DATA))

    assert result == "提案します"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    import importlib
    import app.llm as llm_module
    importlib.reload(llm_module)

    with pytest.raises(ValueError, match="未対応のプロバイダー"):
        _collect(llm_module.stream_analyze(SAMPLE_DATA))


def test_custom_model_used(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "claude-opus-4-6")

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["ok"])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    with patch("anthropic.Anthropic", return_value=mock_client):
        import importlib
        import app.llm as llm_module
        importlib.reload(llm_module)
        _collect(llm_module.stream_analyze(SAMPLE_DATA))

    _, kwargs = mock_client.messages.stream.call_args
    assert kwargs["model"] == "claude-opus-4-6"
