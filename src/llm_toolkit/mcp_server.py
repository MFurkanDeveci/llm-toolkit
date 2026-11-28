"""MCP sunucusu (Ders 5.6): belge aramamızı standart bir araç olarak yayınlar.

Şimdiye kadar belgelerde_ara aracını sadece kendi agent'ımız kullanabiliyordu. MCP (Model
Context Protocol), araçları uygulamadan bağımsız, standart bir yolla sunmayı sağlar: aynı
sunucuyu Claude Desktop, VS Code ya da MCP destekleyen herhangi bir uygulama kullanabilir.

Sadece okuyan arama aracı yayınlanır. iade_baslat bilerek yayınlanmaz: onu kimin, hangi onayla
çalıştıracağına bizim uygulamamız karar veriyor (Ders 5.5); başka bir uygulamaya devredilmez.

Çalıştırma: uv run python -m llm_toolkit.mcp_server
Bu sunucu stdin/stdout üzerinden konuşur (stdio). stdout protokole ait olduğu için bu dosyada
asla print() kullanılmaz; günlükler logging ile stderr'e yazılır.
"""

import logging
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from llm_toolkit.agent import SearchArgs, make_search_tool
from llm_toolkit.config import Settings
from llm_toolkit.embeddings import OllamaEmbedder
from llm_toolkit.vector_store import Embedder, VectorStore

logger = logging.getLogger(__name__)

# Sunucuyu başka bir uygulama başlatır ve çalışma klasörü proje kökü olmayabilir; bu yüzden
# indeksin yerini bu dosyanın konumundan buluyoruz: src/llm_toolkit/ -> proje kökü
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
MAX_K = 5  # Bir aramada en fazla kaç parça dönsün (istemci ne isterse istesin)


def build_server(store: VectorStore, embedder: Embedder) -> MCPServer:
    """Arama aracını taşıyan MCP sunucusunu kurar. Testler sahte indeks ve embedder verir."""
    server = MCPServer(
        "kuzey-kitap",
        version="0.1.0",
        instructions="Kuzey Kitap yardım belgelerinde (iade, kargo, üyelik) arama yapar.",
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def belgelerde_ara(
        query: str = Field(description="Aranacak ifade; kısa ve net, örn. 'iade süresi'"),
        k: int = Field(default=3, description=f"Kaç parça dönsün (1-{MAX_K})"),
    ) -> str:
        """Kuzey Kitap yardım belgelerinde anlamca arama yapar.

        En ilgili parçaları kaynak kimlikleriyle döndürür, örn. <kaynak id="iade.md #0">.
        Cevap verirken bu kimlikleri kaynak olarak gösterebilirsin.
        """
        k = max(1, min(k, MAX_K))
        logger.info("Arama: %r (k=%d)", query, k)
        # Agent'ın kullandığı aracın aynısı: aynı arama, aynı çıktı biçimi
        tool = make_search_tool(store, embedder, k, log=[])
        return tool.function(SearchArgs(query=query))

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)  # Varsayılan olarak stderr'e yazar
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Her Ollama isteğini yazmasın
    settings = Settings()
    store = VectorStore.load(INDEX_DIR)
    embedder = OllamaEmbedder(store.model, host=settings.ollama_host, timeout_s=settings.timeout_s)
    logger.info("İndeks yüklendi: %d parça, %s", len(store.chunks), INDEX_DIR)
    build_server(store, embedder).run(transport="stdio")


if __name__ == "__main__":
    main()
