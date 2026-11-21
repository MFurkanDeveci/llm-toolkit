"""Ollama sağlayıcısının testleri. Gerçek ollama yanıt tipleri kullanılır, ağa çıkılmaz."""

import ollama
import pytest

from llm_toolkit.datatypes import Message
from llm_toolkit.errors import PermanentLLMError
from llm_toolkit.providers.ollama_client import OllamaClient, to_ollama_messages
from llm_toolkit.schemas import ReviewAnalysis


def chat_response(content: str = "", tool_calls: list | None = None, done: bool = True):
    data = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        "done": done,
    }
    if done:
        data |= {"prompt_eval_count": 30, "eval_count": 12}
    return ollama.ChatResponse.model_validate(data)


class FakeOllama:
    def __init__(self, result):
        self.result = result
        self.kwargs: dict = {}

    def chat(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_complete_text_tokens_and_zero_cost():
    sdk = FakeOllama(chat_response("Merhaba"))
    response = OllamaClient("llama3.2:3b", sdk=sdk).complete([Message(role="user", content="x")])
    assert response.text == "Merhaba"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (30, 12)
    assert response.cost_usd == 0.0


def test_tool_calls_get_generated_ids():
    calls = [{"function": {"name": "predict_iris", "arguments": {"sepal_length": 6.7}}}]
    sdk = FakeOllama(chat_response(tool_calls=calls))
    response = OllamaClient("llama3.2:3b", sdk=sdk).complete([Message(role="user", content="x")])
    assert response.tool_calls[0].id == "call_0"
    assert response.tool_calls[0].arguments == {"sepal_length": 6.7}


def test_complete_json_uses_format_and_num_predict():
    sdk = FakeOllama(chat_response("{}"))
    OllamaClient("llama3.2:3b", sdk=sdk, max_tokens=500).complete_json(
        [Message(role="user", content="x")], ReviewAnalysis
    )
    assert sdk.kwargs["format"]["required"] == ["sentiment", "score", "topics", "summary"]
    assert sdk.kwargs["options"] == {"num_predict": 500}


def test_system_message_goes_first():
    converted = to_ollama_messages([Message(role="user", content="soru")], system="Kısa ol.")
    assert converted[0] == {"role": "system", "content": "Kısa ol."}


def test_stream_yields_text_then_response():
    chunks = [chat_response("Mer", done=False), chat_response("haba", done=True)]
    sdk = FakeOllama(iter(chunks))
    parts = list(OllamaClient("llama3.2:3b", sdk=sdk).stream([Message(role="user", content="x")]))
    assert parts[:2] == ["Mer", "haba"]
    assert parts[-1].text == "Merhaba"
    assert parts[-1].usage.output_tokens == 12


def test_server_down_gives_clear_message():
    sdk = FakeOllama(ConnectionError("Failed to connect"))
    with pytest.raises(PermanentLLMError, match="ollama serve"):
        OllamaClient("llama3.2:3b", sdk=sdk).complete([Message(role="user", content="x")])


def test_missing_model_is_permanent():
    sdk = FakeOllama(ollama.ResponseError("model 'yok' not found", status_code=404))
    with pytest.raises(PermanentLLMError, match="404"):
        OllamaClient("yok", sdk=sdk).complete([Message(role="user", content="x")])
