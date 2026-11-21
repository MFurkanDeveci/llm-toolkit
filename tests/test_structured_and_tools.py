"""Sağlayıcıdan bağımsız katmanın testleri: yapılandırılmış çıktı ve tool döngüsü."""

import json

import httpx
import pytest
from conftest import FakeLLM, make_response

from llm_toolkit.datatypes import ToolCall
from llm_toolkit.errors import StructuredOutputError, ToolLoopError
from llm_toolkit.iris_tools import make_iris_tools
from llm_toolkit.schemas import ReviewAnalysis
from llm_toolkit.structured import extract
from llm_toolkit.tools import execute_tool, run_with_tools

VALID = {"sentiment": "olumlu", "score": 5, "topics": ["kargo"], "summary": "Memnun."}


def test_extract_valid_first_try():
    llm = FakeLLM([make_response(json.dumps(VALID))])
    result, responses = extract(llm, "yorum", ReviewAnalysis)
    assert result.score == 5
    assert len(responses) == 1


def test_extract_retries_with_error_feedback():
    invalid = {**VALID, "score": 9}  # JSON biçimi doğru ama 1-5 kuralına uymuyor
    llm = FakeLLM([make_response(json.dumps(invalid)), make_response(json.dumps(VALID))])
    result, responses = extract(llm, "yorum", ReviewAnalysis)
    assert result.score == 5
    assert len(responses) == 2
    feedback = llm.calls[1][-1].content  # ikinci denemede modele giden son mesaj
    assert "score" in feedback


def test_extract_gives_up():
    llm = FakeLLM([make_response("bu json değil"), make_response("hala değil")])
    with pytest.raises(StructuredOutputError):
        extract(llm, "yorum", ReviewAnalysis, max_attempts=2)


def iris_service(handler) -> list:
    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return make_iris_tools("http://test", http=http)


def fake_predict(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body["petal_length"] <= 0:
        return httpx.Response(422, json={"detail": "petal_length > 0 olmalı"})
    return httpx.Response(200, json={"label": "virginica", "probabilities": {"virginica": 0.96}})


def test_tool_loop_calls_tool_then_answers():
    tools = iris_service(fake_predict)
    args = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    llm = FakeLLM(
        [
            make_response(tool_calls=[ToolCall(id="c1", name="predict_iris", arguments=args)]),
            make_response("Bu çiçek %96 olasılıkla virginica."),
        ]
    )
    answer, responses, messages = run_with_tools(llm, "Hangi tür?", tools)
    assert "virginica" in answer
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert "0.96" in messages[2].content  # servis cevabı modele iletildi


def test_tool_errors_are_returned_to_model_not_raised():
    tools = {t.name: t for t in iris_service(fake_predict)}
    unknown = execute_tool(tools, ToolCall(id="1", name="yok", arguments={}))
    bad_args = execute_tool(tools, ToolCall(id="2", name="predict_iris", arguments={"x": 1}))
    rejected = execute_tool(
        tools,
        ToolCall(
            id="3",
            name="predict_iris",
            arguments={"sepal_length": 1, "sepal_width": 1, "petal_length": -1, "petal_width": 1},
        ),
    )
    assert unknown.startswith("HATA") and "predict_iris" in unknown
    assert bad_args.startswith("HATA: Argümanlar")
    assert rejected.startswith("HATA: Servis")


def test_service_down_is_reported_to_model():
    def down(request):
        raise httpx.ConnectError("bağlantı reddedildi")

    tools = {t.name: t for t in iris_service(down)}
    result = execute_tool(tools, ToolCall(id="1", name="iris_model_info", arguments={}))
    assert "ConnectError" in result


def test_tool_loop_has_step_limit():
    call = ToolCall(id="c", name="iris_model_info", arguments={})
    tools = iris_service(lambda r: httpx.Response(200, json={}))
    llm = FakeLLM([make_response(tool_calls=[call]) for _ in range(3)])
    with pytest.raises(ToolLoopError):
        run_with_tools(llm, "?", tools, max_steps=3)
