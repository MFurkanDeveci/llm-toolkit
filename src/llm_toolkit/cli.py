"""Komut satırı arayüzü: `uv run llm <komut> ...`

Örnekler:
    uv run llm ask "RAG nedir?" --stream
    uv run llm extract "Kargo geç geldi ama ürün harika."
    uv run llm agent "6.7, 3.0, 5.2, 2.3 ölçülerindeki çiçek hangi tür?"
    uv run llm bench --n 5
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from llm_toolkit.config import Settings
from llm_toolkit.datatypes import LLMResponse, Message
from llm_toolkit.errors import LLMError
from llm_toolkit.iris_tools import make_iris_tools
from llm_toolkit.providers import LLMClient, create_client
from llm_toolkit.schemas import ReviewAnalysis
from llm_toolkit.structured import extract
from llm_toolkit.tools import run_with_tools

AGENT_SYSTEM = (
    "Sen bir botanik asistanısın. Çiçek türü sorularında mutlaka predict_iris aracını kullan; "
    "tahmin etme. Cevabı Türkçe, kısa ve olasılığıyla birlikte ver."
)
BENCH_QUESTIONS = [
    "Overfitting'i tek cümleyle açıkla.",
    "Precision ile recall farkı nedir? Tek cümle.",
    "Cross-validation neden yapılır? Tek cümle.",
    "Feature scaling neden gerekir? Tek cümle.",
    "Data leakage nedir? Tek cümle.",
    "Bias-variance dengesi nedir? Tek cümle.",
    "Regularization ne işe yarar? Tek cümle.",
    "Confusion matrix nedir? Tek cümle.",
]


def print_usage(responses: Sequence[LLMResponse]) -> None:
    """Bir işin toplam maliyetini ve süresini yazar. stderr'e yazar ki cevap metnine karışmasın."""
    tokens_in = sum(r.usage.input_tokens for r in responses)
    tokens_out = sum(r.usage.output_tokens for r in responses)
    cost = sum(r.cost_usd for r in responses)
    latency = sum(r.latency_s for r in responses)
    first = responses[0]
    print(
        f"\n[{first.provider}/{first.model}] çağrı={len(responses)} "
        f"token: girdi={tokens_in} çıktı={tokens_out} | "
        f"maliyet=${cost:.6f} | süre={latency:.2f} sn",
        file=sys.stderr,
    )


def cmd_ask(client: LLMClient, args: argparse.Namespace) -> None:
    messages = [Message(role="user", content=args.question)]
    if not args.stream:
        response = client.complete(messages)
        print(response.text)
        print_usage([response])
        return
    final = None
    for part in client.stream(messages):
        if isinstance(part, LLMResponse):
            final = part  # Akış bitti; son parça toplam bilgiyi taşır
        else:
            print(part, end="", flush=True)  # flush: parçayı beklemeden ekrana bas
    print()
    if final:
        print_usage([final])


def cmd_extract(client: LLMClient, args: argparse.Namespace) -> None:
    result, responses = extract(
        client,
        f"Aşağıdaki müşteri yorumunu analiz et:\n\n{args.text}",
        ReviewAnalysis,
        system="Sen bir e-ticaret müşteri yorumu analistisin.",
    )
    print(result.model_dump_json(indent=2))
    print_usage(responses)


def cmd_agent(client: LLMClient, args: argparse.Namespace, settings: Settings) -> None:
    tools = make_iris_tools(settings.ml_service_url)
    answer, responses, messages = run_with_tools(client, args.question, tools, system=AGENT_SYSTEM)
    if args.verbose:
        for m in messages:
            calls = f" araç_çağrısı={[(c.name, c.arguments) for c in m.tool_calls]}"
            print(f"[{m.role}] {m.content[:200]}{calls if m.tool_calls else ''}", file=sys.stderr)
    print(answer)
    print_usage(responses)


def cmd_bench(client: LLMClient, args: argparse.Namespace) -> None:
    """Aynı soruları önce sırayla, sonra eşzamanlı gönderip süreyi karşılaştırır."""
    questions = BENCH_QUESTIONS[: args.n]

    def ask(q: str) -> LLMResponse:
        return client.complete([Message(role="user", content=q)], max_tokens=args.max_tokens)

    start = time.perf_counter()
    sequential = [ask(q) for q in questions]
    t_seq = time.perf_counter() - start

    start = time.perf_counter()
    # LLM çağrısının süresinin neredeyse tamamı ağ cevabını beklemekle geçer (I/O-bound).
    # Thread'ler beklerken diğer istekler de yola çıkar; toplam süre en yavaş isteğe yaklaşır.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        concurrent = list(pool.map(ask, questions))
    t_con = time.perf_counter() - start

    print(f"{len(questions)} soru | sırayla: {t_seq:.2f} sn | eşzamanlı: {t_con:.2f} sn")
    print(f"Hızlanma: {t_seq / t_con:.1f}x")
    print_usage(sequential + concurrent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm", description="Modül 2 LLM araç seti")
    parser.add_argument("--provider", choices=["anthropic", "ollama"], help="Ayarı geçersiz kıl")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="Soru sor")
    p.add_argument("question")
    p.add_argument("--stream", action="store_true", help="Cevabı geldikçe yazdır")

    p = sub.add_parser("extract", help="Müşteri yorumundan yapılandırılmış analiz çıkar")
    p.add_argument("text")

    p = sub.add_parser("agent", help="Modül 1 servisini araç olarak kullanan asistan")
    p.add_argument("question")
    p.add_argument("-v", "--verbose", action="store_true", help="Konuşma adımlarını göster")

    p = sub.add_parser("bench", help="Sıralı ve eşzamanlı çağrı süresini karşılaştır")
    p.add_argument("--n", type=int, default=5, choices=range(1, len(BENCH_QUESTIONS) + 1))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if args.provider:
        settings = settings.model_copy(update={"provider": args.provider})
    try:
        client = create_client(settings)
        if args.command == "ask":
            cmd_ask(client, args)
        elif args.command == "extract":
            cmd_extract(client, args)
        elif args.command == "agent":
            cmd_agent(client, args, settings)
        elif args.command == "bench":
            cmd_bench(client, args)
    except LLMError as exc:
        sys.exit(f"HATA ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    main()
