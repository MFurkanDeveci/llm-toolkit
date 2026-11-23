"""Altın soru seti: cevabı belgelerde kesin olarak bilinen sorular.

Her soruda:
- evidence: cevabı taşıyan ve belgelerde BİREBİR geçen kısa ifadeler. "Arama doğru parçayı
  buldu mu?" sorusu, parça numarasına değil bu ifadelere bakılarak cevaplanır; böylece parça
  boyunu değiştirsek de soru seti geçerli kalır.
- answer_keywords: doğru bir cevapta geçmesi gereken ifadeler. "a|b" = a ya da b yeterli.
- category "yok": belgelerde cevabı olmayan soru. Doğru davranış "bilgi yok" demektir.

Komut satırından kontrol: uv run python -m llm_toolkit.evalset
"""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from llm_toolkit.chunking import split_sentences

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
DOCS_DIR = Path("data/docs")

# olgu: tek bilgi · anlam: soruda belgedeki kelimeler geçmiyor · kesin: sayı/tutar
# coklu: birden fazla bilgi gerekiyor · yok: belgelerde cevap yok
Category = Literal["olgu", "anlam", "kesin", "coklu", "yok"]


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: Category
    evidence: list[str]
    answer_keywords: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """JSON Lines dosyasını okur: her satır bir soru. Boş satırları atlar."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]


def load_documents(folder: Path = DOCS_DIR) -> dict[str, str]:
    """Klasördeki .md ve .txt dosyalarını {dosya adı: metin} olarak okur."""
    paths = sorted([*folder.glob("*.md"), *folder.glob("*.txt")])
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def find_problems(questions: list[EvalQuestion], documents: dict[str, str]) -> list[str]:
    """Soru setindeki hataları listeler. Boş liste = set kullanıma hazır."""
    problems = [
        f"{qid}: kimlik {count} kez kullanılmış"
        for qid, count in Counter(q.id for q in questions).items()
        if count > 1
    ]
    sentences = [s for text in documents.values() for s in split_sentences(text, 10_000)]
    for q in questions:
        if q.category == "yok" and (q.evidence or q.answer_keywords):
            problems.append(f"{q.id}: 'yok' sorusunda kanıt ve beklenen ifade boş olmalı")
        if q.category != "yok" and not (q.evidence and q.answer_keywords):
            problems.append(f"{q.id}: kanıt ve beklenen ifade gerekli")
        for evidence in q.evidence:
            # Kanıt tek bir cümlenin içinde olmalı: parçalar cümle sınırında kesildiği için
            # böylece parça boyu ne olursa olsun kanıt bölünmeden bir parçaya düşer
            if not any(evidence in sentence for sentence in sentences):
                problems.append(f"{q.id}: kanıt tek bir cümlede bulunamadı: {evidence!r}")
    return problems


def main() -> int:
    questions = load_questions()
    counts = Counter(q.category for q in questions)
    print(f"{len(questions)} soru: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    problems = find_problems(questions, load_documents())
    for problem in problems:
        print("SORUN:", problem)
    print(f"{len(problems)} sorun var." if problems else "Soru seti kullanıma hazır.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
