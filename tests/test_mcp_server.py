"""MCP sunucusu testleri: sunucuya bellekte bağlanan gerçek bir MCP istemcisiyle, ağa çıkmadan."""

import anyio
import numpy as np
from mcp import Client

from llm_toolkit.chunking import Chunk
from llm_toolkit.embeddings import EmbeddingResult
from llm_toolkit.mcp_server import build_server
from llm_toolkit.vector_store import VectorStore


class KeywordEmbedder:
    model = "keyword-test"
    WORDS = ["iade", "kargo", "puan"]

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors = np.array([[float(w in t.lower()) for w in self.WORDS] for t in texts])
        return EmbeddingResult(vectors=vectors, model=self.model, input_tokens=4, latency_s=0.01)


TEXTS = ["İade süresi 14 gündür.", "Kargo ücreti 39,90 TL'dir.", "Her alışverişte puan kazanılır."]
CHUNKS = [Chunk(source=f"d{i}.md", index=0, text=t) for i, t in enumerate(TEXTS)]
STORE = VectorStore(CHUNKS, KeywordEmbedder().embed(TEXTS).vectors, "keyword-test")
SERVER = build_server(STORE, KeywordEmbedder())


def call(tool: str, arguments: dict):
    """MCP istemcisiyle sunucuya bağlanır ve bir aracı çağırır (Claude Desktop'ın yaptığı gibi)."""

    async def main():
        async with Client(SERVER) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(main)


def test_only_the_read_only_search_tool_is_published():
    async def main():
        async with Client(SERVER) as client:
            return (await client.list_tools()).tools

    tools = anyio.run(main)
    assert [t.name for t in tools] == ["belgelerde_ara"]  # iade_baslat bilerek yayınlanmıyor
    assert tools[0].annotations.read_only_hint is True
    assert "query" in tools[0].input_schema["properties"]


def test_search_returns_sources_with_ids():
    result = call("belgelerde_ara", {"query": "kargo ücreti", "k": 1})
    assert not result.is_error
    text = result.content[0].text
    assert '<kaynak id="d1.md #0">' in text
    assert "39,90 TL" in text


def test_k_is_clamped_to_a_safe_range():
    many = call("belgelerde_ara", {"query": "iade", "k": 999}).content[0].text
    assert many.count("<kaynak") == 3  # En fazla MAX_K, ama indekste 3 parça var
    one = call("belgelerde_ara", {"query": "iade", "k": 0}).content[0].text
    assert one.count("<kaynak") == 1  # En az 1


def test_missing_argument_is_an_error_not_a_crash():
    result = call("belgelerde_ara", {})
    assert result.is_error  # Hata istemciye iletilir; sunucu çökmez
