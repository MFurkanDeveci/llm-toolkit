"""Claude sağlayıcısının testleri. Gerçek SDK yanıt tipleri kullanılır, ağa çıkılmaz."""

import anthropic
import httpx
import pytest
from anthropic.types import Message as AnthropicMessage

from llm_toolkit.datatypes import Message, ToolCall
from llm_toolkit.errors import PermanentLLMError, TransientLLMError
from llm_toolkit.providers.anthropic_client import AnthropicClient, to_anthropic_messages
from llm_toolkit.schemas import ReviewAnalysis


def api_message(content: list[dict], stop_reason: str = "end_turn") -> AnthropicMessage:
    """API'nin döndürdüğü JSON'dan SDK'nın kendi Message nesnesini üretir."""
    return AnthropicMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
    )


def api_error(cls, status: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("hata", response=httpx.Response(status, request=request), body=None)


class FakeMessages:
    def __init__(self, results):
        self.results = list(results)
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeSDK:
    def __init__(self, *results):
        self.messages = FakeMessages(results)


THINKING = {"type": "thinking", "thinking": "Kullanıcı selam veriyor.", "signature": "sig123"}


def test_text_skips_thinking_blocks():
    # Haiku 5.5'te ilk blok genelde düşünme bloğudur; content[0].text bu yüzden güvenilmez
    sdk = FakeSDK(api_message([THINKING, {"type": "text", "text": "Merhaba!"}]))
    response = AnthropicClient("claude-haiku-5-5", sdk=sdk).complete(
        [Message(role="user", content="Selam")]
    )
    assert response.text == "Merhaba!"
    assert response.provider_blocks[0]["type"] == "thinking"


def test_cost_is_computed_from_usage():
    sdk = FakeSDK(api_message([{"type": "text", "text": "ok"}]))
    response = AnthropicClient("claude-haiku-5-5", sdk=sdk).complete(
        [Message(role="user", content="x")]
    )
    # 1000 girdi * 0.10$/M + 200 çıktı * 0.50$/M = 0.0001 + 0.0001
    assert response.cost_usd == pytest.approx(0.0002)


def test_request_contains_effort_and_default_max_tokens():
    sdk = FakeSDK(api_message([{"type": "text", "text": "ok"}]))
    AnthropicClient("claude-haiku-5-5", sdk=sdk, effort="low", max_tokens=4096).complete(
        [Message(role="user", content="x")], system="Kısa cevap ver."
    )
    sent = sdk.messages.kwargs[0]
    assert sent["output_config"] == {"effort": "low"}
    assert sent["max_tokens"] == 4096
    assert sent["system"] == "Kısa cevap ver."


def test_tool_use_is_parsed():
    sdk = FakeSDK(
        api_message(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "predict_iris",
                    "input": {"sepal_length": 6.7},
                }
            ],
            stop_reason="tool_use",
        )
    )
    response = AnthropicClient("claude-haiku-5-5", sdk=sdk).complete(
        [Message(role="user", content="x")]
    )
    assert response.tool_calls == [
        ToolCall(id="toolu_1", name="predict_iris", arguments={"sepal_length": 6.7})
    ]


def test_complete_json_sends_schema():
    sdk = FakeSDK(api_message([{"type": "text", "text": "{}"}]))
    AnthropicClient("claude-haiku-5-5", sdk=sdk).complete_json(
        [Message(role="user", content="x")], ReviewAnalysis
    )
    fmt = sdk.messages.kwargs[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == {"sentiment", "score", "topics", "summary"}


def test_tool_results_grouped_and_blocks_preserved():
    blocks = [THINKING, {"type": "tool_use", "id": "t1", "name": "a", "input": {}}]
    converted = to_anthropic_messages(
        [
            Message(role="user", content="soru"),
            Message(role="assistant", provider_blocks=blocks),
            Message(role="tool", content="sonuç 1", tool_call_id="t1", tool_name="a"),
            Message(role="tool", content="sonuç 2", tool_call_id="t2", tool_name="a"),
        ]
    )
    assert converted[1] == {"role": "assistant", "content": blocks}  # düşünme bloğu korunur
    assert len(converted) == 3  # iki araç sonucu tek user mesajında
    assert [b["tool_use_id"] for b in converted[2]["content"]] == ["t1", "t2"]


def test_rate_limit_is_retried(no_sleep):
    delays, sleep = no_sleep
    sdk = FakeSDK(
        api_error(anthropic.RateLimitError, 429), api_message([{"type": "text", "text": "ok"}])
    )
    response = AnthropicClient("claude-haiku-5-5", sdk=sdk, sleep=sleep).complete(
        [Message(role="user", content="x")]
    )
    assert response.text == "ok"
    assert len(delays) == 1


def test_auth_error_is_not_retried(no_sleep):
    delays, sleep = no_sleep
    sdk = FakeSDK(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(PermanentLLMError):
        AnthropicClient("claude-haiku-5-5", sdk=sdk, sleep=sleep).complete(
            [Message(role="user", content="x")]
        )
    assert delays == []


def test_gives_up_after_max_attempts(no_sleep):
    delays, sleep = no_sleep
    errors = [api_error(anthropic.InternalServerError, 500) for _ in range(3)]
    sdk = FakeSDK(*errors)
    with pytest.raises(TransientLLMError):
        AnthropicClient("claude-haiku-5-5", sdk=sdk, sleep=sleep, max_attempts=3).complete(
            [Message(role="user", content="x")]
        )
    assert len(delays) == 2  # 3 deneme arasında 2 bekleme
