"""İşlem araçları: dünyayı değiştiren araçlar (Ders 5.5).

Arama sadece okur; yanlış bir arama en fazla biraz para harcatır. İade başlatmak ise gerçek
bir işlemdir: yanlış siparişe iade açmak müşteriyi ve şirketi etkiler. Bu yüzden agent bu
araçları kendi başına çalıştıramaz; kullanıcı onaylamalıdır (agent.resume_agent).

Terminalde dene (Claude Haiku kullanır): uv run python -m llm_toolkit.actions
"""

from pathlib import Path

from pydantic import BaseModel, Field

from llm_toolkit.agent import resume_agent, run_agent
from llm_toolkit.config import Settings
from llm_toolkit.datatypes import Tool
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.providers import create_client
from llm_toolkit.vector_store import VectorStore


class RefundArgs(BaseModel):
    order_id: str = Field(
        pattern=r"^KK-\d{6}$",
        description="Sipariş numarası, örnek: KK-123456. Kullanıcıdan öğren; asla uydurma.",
    )
    reason: str = Field(
        min_length=3, max_length=200, description="İade sebebi, kullanıcının anlattığı gibi kısaca"
    )


class RefundRequest(BaseModel):
    refund_id: str
    order_id: str
    reason: str


class RefundDesk:
    """Sahte iade sistemi. Gerçek bir şirkette burası sipariş sistemine API çağrısı yapardı."""

    def __init__(self) -> None:
        self.requests: list[RefundRequest] = []

    def start_refund(self, args: RefundArgs) -> str:
        # Aynı işlem iki kez yapılmasın (örn. kullanıcı iki kez onaylarsa)
        for request in self.requests:
            if request.order_id == args.order_id:
                return f"Bu sipariş için zaten bir iade talebi var: {request.refund_id}."
        request = RefundRequest(
            refund_id=f"IADE-{len(self.requests) + 1:04d}",
            order_id=args.order_id,
            reason=args.reason,
        )
        self.requests.append(request)
        return (
            f"İade talebi oluşturuldu. Takip numarası: {request.refund_id}. "
            "İade kargo kodu müşteriye e-postayla gönderilecek."
        )

    def as_tool(self) -> Tool:
        return Tool(
            name="iade_baslat",
            description=(
                "Bir sipariş için iade talebi oluşturur. GERÇEK bir işlemdir ve kullanıcı "
                "onaylamadan çalışmaz. Sadece kullanıcı açıkça iade istediğinde ve sipariş "
                "numarasını verdiğinde çağır."
            ),
            parameters=RefundArgs,
            function=self.start_refund,
        )


def main() -> int:
    settings = Settings().model_copy(
        update={"provider": "anthropic", "anthropic_model": "claude-haiku-5-5"}
    )
    store = VectorStore.load(Path("data/index"))
    embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
    client = create_client(settings)
    desk = RefundDesk()
    actions = [desk.as_tool()]

    run = run_agent(input("Soru: "), store, embedder, client, actions=actions)
    while run.stop_reason == "onay":
        for call in run.pending:
            print(f"\nAgent şu işlemi yapmak istiyor: {call.name} {call.arguments}")
        approved = input("Onaylıyor musun? (e/h): ").strip().lower() == "e"
        run = resume_agent(run, approved, store, embedder, client, actions=actions)

    print(f"\nCevap: {run.answer}")
    print(f"[{run.model_calls} çağrı · {len(run.searches)} arama · ${run.cost_usd:.6f}]")
    print(f"Oluşturulan iadeler: {[r.refund_id for r in desk.requests] or 'yok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
