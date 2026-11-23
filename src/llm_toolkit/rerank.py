"""Reranking: aramanın bulduğu ilk birkaç adayı bir LLM'e okutup yeniden sıralamak.

İki aşamalı arama:
1. Hızlı ve ucuz arama (vektör, BM25 ya da hibrit) tüm parçalardan N aday seçer.
2. Daha yavaş ama daha dikkatli bir model sadece bu N adayı okuyup soruyla ilgisine puan verir.

Reranker sadece adayların sırasını değiştirebilir. Doğru parça ilk N adayın içinde yoksa
reranker onu hiç görmez; bu yüzden ulaşılabilecek en iyi sonuç ilk aşamanın recall@N'idir.

Ollama'nın standart bir rerank uç noktası olmadığı için cross-encoder yerine LLM kullanıyoruz.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from llm_toolkit.chunking import Chunk
from llm_toolkit.datatypes import LLMResponse
from llm_toolkit.evalset import EvalQuestion
from llm_toolkit.hybrid import BM25
from llm_toolkit.providers.base import LLMClient
from llm_toolkit.retrieval_eval import RetrievalResult, rank_chunks, score_ranking
from llm_toolkit.structured import extract
from llm_toolkit.vector_store import VectorStore

RERANK_SYSTEM = """Bir arama sistemindeki parçaların soruyla ilgisini puanlıyorsun.
Her parça için 0-3 arası bir puan ver:
3 = soruyu doğrudan cevaplıyor
2 = cevabın bir kısmını içeriyor
1 = konuyla ilgili ama soruyu cevaplamıyor
0 = ilgisiz
Her parçaya, kendi id'siyle, tam olarak bir puan ver."""


class RelevanceScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="Parçanın id'si")
    score: int = Field(ge=0, le=3, description="0 = ilgisiz, 3 = soruyu doğrudan cevaplıyor")


class RerankOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[RelevanceScore]


def build_rerank_prompt(query: str, chunks: list[Chunk]) -> str:
    passages = "\n\n".join(f'<parca id="{i}">\n{c.text}\n</parca>' for i, c in enumerate(chunks))
    return f"<soru>{query}</soru>\n\n{passages}"


def rerank(
    query: str, chunks: list[Chunk], client: LLMClient
) -> tuple[list[int], list[LLMResponse]]:
    """Parçaları modelin verdiği puana göre sıralar; yeni sırayı (indeksler) döndürür.

    Eşit puanlı parçalar eski sıralarını korur. Modelin puanlamayı unuttuğu parçalar sona,
    uydurduğu (var olmayan) id'ler yok sayılır.
    """
    if not chunks:
        return [], []
    output, responses = extract(
        client, build_rerank_prompt(query, chunks), RerankOutput, system=RERANK_SYSTEM
    )
    scores = {s.id: s.score for s in output.scores if 0 <= s.id < len(chunks)}
    order = sorted(range(len(chunks)), key=lambda i: -scores.get(i, -1))  # sorted kararlıdır
    return order, responses


class RerankRun(BaseModel):
    results: list[RetrievalResult]
    cost_usd: float
    latency_s: float


def evaluate_reranking(
    questions: list[EvalQuestion],
    store: VectorStore,
    bm25: BM25,
    query_vectors: np.ndarray,
    client: LLMClient,
    *,
    base_method: str = "hibrit",
    candidates: int = 6,
    k: int = 3,
) -> RerankRun:
    """Her cevaplanabilir soruda: ilk aşamadan N aday al, rerank et, recall@k ve sırayı ölç."""
    results, cost, latency = [], 0.0, 0.0
    for question, vector in zip(questions, query_vectors, strict=True):
        if not question.answerable:
            continue
        ranking = rank_chunks(base_method, store, bm25, vector, question.question)
        top = ranking[:candidates]
        order, responses = rerank(question.question, [store.chunks[i] for i in top], client)
        # Adayların sırası değişir; geri kalan parçalar ilk aşamadaki sıralarında kalır
        new_ranking = [top[i] for i in order] + ranking[candidates:]
        first_rank, recall = score_ranking(new_ranking, store.chunks, question.evidence, k)
        results.append(
            RetrievalResult(
                question_id=question.id,
                category=question.category,
                method=f"{base_method}+rerank",
                first_rank=first_rank,
                recall=recall,
            )
        )
        cost += sum(r.cost_usd for r in responses)
        latency += sum(r.latency_s for r in responses)
    return RerankRun(results=results, cost_usd=cost, latency_s=latency)
