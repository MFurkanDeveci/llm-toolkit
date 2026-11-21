"""Tekrar deneme, fiyatlandırma ve komut satırı testleri."""

import json

import pytest
from conftest import FakeLLM, make_response

from llm_toolkit import cli
from llm_toolkit.datatypes import Usage
from llm_toolkit.errors import PermanentLLMError, TransientLLMError
from llm_toolkit.pricing import cost_usd
from llm_toolkit.retry import with_retries


def test_retry_succeeds_after_transient_errors(no_sleep):
    delays, sleep = no_sleep
    attempts = iter([TransientLLMError("429"), TransientLLMError("529"), "tamam"])

    def flaky():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    assert with_retries(flaky, max_attempts=3, base_delay=1, sleep=sleep) == "tamam"
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 1.5 and 1.0 <= delays[1] <= 3.0  # 1 sn, sonra 2 sn (± jitter)


def test_retry_does_not_retry_permanent_errors(no_sleep):
    delays, sleep = no_sleep

    def broken():
        raise PermanentLLMError("401")

    with pytest.raises(PermanentLLMError):
        with_retries(broken, sleep=sleep)
    assert delays == []


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-haiku-5-5", 0.10 + 0.50),  # 1M girdi + 1M çıktı
        ("anthropic", "claude-sonnet-5-5", 2.00 + 10.00),
        ("ollama", "llama3.2:3b", 0.0),
        ("anthropic", "bilinmeyen-model", 0.0),
    ],
)
def test_cost(provider, model, expected):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(provider, model, usage) == pytest.approx(expected)


def run_cli(monkeypatch, llm, argv):
    monkeypatch.setattr(cli, "create_client", lambda settings: llm)
    cli.main(argv)


def test_cli_ask(monkeypatch, capsys):
    run_cli(monkeypatch, FakeLLM([make_response("Merhaba dünya")]), ["ask", "selam"])
    out, err = capsys.readouterr()
    assert out.strip() == "Merhaba dünya"
    assert "maliyet=$0.001000" in err  # kullanım bilgisi stderr'de, cevaba karışmıyor


def test_cli_ask_stream(monkeypatch, capsys):
    run_cli(monkeypatch, FakeLLM([make_response("bir iki")]), ["ask", "selam", "--stream"])
    assert capsys.readouterr().out.strip() == "biriki"


def test_cli_extract(monkeypatch, capsys):
    valid = {"sentiment": "karışık", "score": 3, "topics": ["kargo"], "summary": "Orta."}
    run_cli(monkeypatch, FakeLLM([make_response(json.dumps(valid))]), ["extract", "yorum"])
    assert json.loads(capsys.readouterr().out)["sentiment"] == "karışık"


def test_cli_bench(monkeypatch, capsys):
    llm = FakeLLM([make_response("ok") for _ in range(4)])
    run_cli(monkeypatch, llm, ["bench", "--n", "2"])
    assert "Hızlanma" in capsys.readouterr().out


def test_cli_turns_errors_into_message(monkeypatch):
    class Broken(FakeLLM):
        def complete(self, *a, **k):
            raise PermanentLLMError("AuthenticationError: invalid x-api-key")

    with pytest.raises(SystemExit, match="PermanentLLMError"):
        run_cli(monkeypatch, Broken([]), ["ask", "selam"])


def test_missing_api_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir("/")  # proje klasöründeki olası .env dosyası okunmasın
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["--provider", "anthropic", "ask", "selam"])
