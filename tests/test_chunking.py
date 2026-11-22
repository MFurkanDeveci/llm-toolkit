"""Parçalama testleri: ağ ve model gerekmez, saf Python."""

import pytest

from llm_toolkit.chunking import chunk_text, split_sentences

TEXT = (
    "RAG, modelin cevap vermeden önce ilgili belgeleri bulmasıdır. "
    "Önce belgeler küçük parçalara bölünür. Her parça bir vektöre çevrilir. "
    "Soru gelince ona en yakın parçalar bulunur. Bu parçalar modele bağlam olarak verilir. "
    "Model cevabı bu bağlama dayanarak yazar ve kaynak gösterir."
)


def test_short_text_is_single_chunk():
    chunks = chunk_text("Kısa bir metin.", source="a.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "Kısa bir metin."
    assert (chunks[0].source, chunks[0].index) == ("a.txt", 0)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ", source="bos.txt") == []


def test_chunks_respect_max_chars():
    chunks = chunk_text(TEXT, source="rag.txt", max_chars=120, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_sentence_is_lost():
    chunks = chunk_text(TEXT, source="rag.txt", max_chars=120, overlap_chars=40)
    for sentence in split_sentences(TEXT, max_chars=120):
        assert any(sentence in c.text for c in chunks)


def test_without_overlap_chunks_rebuild_the_text():
    chunks = chunk_text(TEXT, source="rag.txt", max_chars=120, overlap_chars=0)
    assert " ".join(c.text for c in chunks) == TEXT


def test_overlap_repeats_last_sentence_of_previous_chunk():
    chunks = chunk_text(TEXT, source="rag.txt", max_chars=120, overlap_chars=60)
    last_sentence = split_sentences(chunks[0].text, max_chars=120)[-1]
    assert chunks[1].text.startswith(last_sentence)


def test_too_long_sentence_is_cut():
    parts = split_sentences("a" * 250, max_chars=100)
    assert [len(p) for p in parts] == [100, 100, 50]


def test_whitespace_and_newlines_are_normalized():
    assert split_sentences("Bir.\n\n  İki   üç.", max_chars=100) == ["Bir.", "İki üç."]


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_text(TEXT, source="rag.txt", max_chars=100, overlap_chars=100)
