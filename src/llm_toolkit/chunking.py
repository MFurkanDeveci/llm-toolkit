"""Parçalama (chunking): uzun metni aramaya uygun, cümle sınırında kesilmiş parçalara böler."""

import re

from pydantic import BaseModel

# Nokta, ünlem veya soru işaretinden sonra gelen boşluk = cümle sonu.
# Basit bir kural: "Dr. Ali" gibi kısaltmalarda yanılır; bu derste yeterli.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    source: str  # Parçanın geldiği doküman (dosya adı). Cevapta kaynak göstermek için.
    index: int  # Doküman içindeki sıra numarası
    text: str


def split_sentences(text: str, max_chars: int) -> list[str]:
    """Metni cümlelere ayırır. max_chars'tan uzun cümleyi harf sayısına göre keser."""
    text = " ".join(text.split())  # Satır sonları ve fazla boşluklar tek boşluk olur
    sentences: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if not sentence:
            continue
        for start in range(0, len(sentence), max_chars):
            sentences.append(sentence[start : start + max_chars])
    return sentences


def chunk_text(
    text: str, source: str, *, max_chars: int = 800, overlap_chars: int = 150
) -> list[Chunk]:
    """Cümleleri max_chars'ı aşmayacak şekilde parçalarda toplar.

    Örtüşme (overlap): yeni parça, önceki parçanın son cümleleriyle (toplam en fazla
    overlap_chars) başlar. Böylece iki parçanın sınırına denk gelen bilgi kaybolmaz.
    """
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars, 0 ile max_chars arasında olmalı (max_chars hariç).")

    groups: list[list[str]] = []
    current: list[str] = []
    for sentence in split_sentences(text, max_chars):
        if current and _length(current + [sentence]) > max_chars:
            groups.append(current)
            current = _tail(current, overlap_chars)
            # Örtüşme + yeni cümle sığmıyorsa örtüşmeyi baştan kısalt
            while current and _length(current + [sentence]) > max_chars:
                current.pop(0)
        current.append(sentence)
    if current:
        groups.append(current)

    return [Chunk(source=source, index=i, text=" ".join(g)) for i, g in enumerate(groups)]


def _length(sentences: list[str]) -> int:
    """Cümleler aralarında birer boşlukla birleşince oluşacak metnin uzunluğu."""
    return len(" ".join(sentences))


def _tail(sentences: list[str], budget: int) -> list[str]:
    """Sondan başlayarak, toplamı budget'ı aşmayan cümleleri seçer."""
    tail: list[str] = []
    for sentence in reversed(sentences):
        if _length([sentence, *tail]) > budget:
            break
        tail.insert(0, sentence)
    return tail
