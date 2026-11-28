"""MCP sunucusunu dışarıdan dene: sunucuyu ayrı bir süreç olarak başlatır ve stdio üzerinden
konuşur. Claude Desktop gibi uygulamalar da sunucuyu tam olarak böyle kullanır.

Çalıştırma: uv run python -m llm_toolkit.mcp_check "İade süresi ne kadar?"
"""

import sys
import time

import anyio
from mcp import Client, StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "llm_toolkit.mcp_server"])


async def check(query: str) -> None:
    start = time.perf_counter()
    async with Client(SERVER) as client:
        print(f"Bağlandı ({time.perf_counter() - start:.2f} sn)")
        for tool in (await client.list_tools()).tools:
            print(f"Araç: {tool.name} · salt okunur: {tool.annotations.read_only_hint}")
        start = time.perf_counter()
        result = await client.call_tool("belgelerde_ara", {"query": query, "k": 2})
        print(f"Arama: {time.perf_counter() - start:.2f} sn · hata: {result.is_error}\n")
        print(result.content[0].text)


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "İade süresi ne kadar?"
    anyio.run(check, query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
