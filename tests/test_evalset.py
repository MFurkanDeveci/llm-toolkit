"""Altın soru seti testleri: hem kodu hem de gerçek soru setinin kendisini kontrol eder."""

from pathlib import Path

from llm_toolkit.evalset import EvalQuestion, find_problems, load_documents, load_questions

ROOT = Path(__file__).parents[1]  # Proje kökü (tests klasörünün bir üstü)
DOCS = {"a.md": "İade süresi 14 gündür. Kargo ücreti 39,90 TL'dir."}


def question(**overrides) -> EvalQuestion:
    data = {
        "id": "q1",
        "question": "İade kaç gün?",
        "category": "olgu",
        "evidence": ["İade süresi 14 gündür"],
        "answer_keywords": ["14 gün"],
    }
    return EvalQuestion(**(data | overrides))


def test_load_questions_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(question().model_dump_json() + "\n\n", encoding="utf-8")
    assert [q.id for q in load_questions(path)] == ["q1"]


def test_valid_question_has_no_problems():
    assert find_problems([question()], DOCS) == []


def test_finds_each_kind_of_mistake():
    problems = find_problems(
        [
            question(id="yazim", evidence=["İade süresi 15 gündür"]),  # Belgede yok
            question(id="iki-cumle", evidence=["14 gündür. Kargo"]),  # Cümle sınırını aşıyor
            question(id="ifadesiz", answer_keywords=[]),  # Beklenen ifade yok
            question(id="yok-ama-kanitli", category="yok"),  # Cevaplanamaz ama kanıtı var
            question(id="ikiz"),
            question(id="ikiz"),
        ],
        DOCS,
    )
    for expected in ["yazim", "iki-cumle", "ifadesiz", "yok-ama-kanitli", "ikiz"]:
        assert any(p.startswith(expected) for p in problems), expected


def test_unanswerable_question():
    assert not question(category="yok", evidence=[], answer_keywords=[]).answerable


# --- Gerçek soru seti: veri de kod gibi test edilir ---


def test_real_question_set_is_valid():
    questions = load_questions(ROOT / "data/eval/questions.jsonl")
    documents = load_documents(ROOT / "data/docs")
    assert find_problems(questions, documents) == []


def test_real_question_set_covers_every_category_and_document():
    questions = load_questions(ROOT / "data/eval/questions.jsonl")
    documents = load_documents(ROOT / "data/docs")
    assert {q.category for q in questions} == {"olgu", "anlam", "kesin", "coklu", "yok"}
    for name, text in documents.items():
        assert any(ev in " ".join(text.split()) for q in questions for ev in q.evidence), name
