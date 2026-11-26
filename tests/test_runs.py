"""Deney kaydı testleri: kaydet, geri oku, tabloya çevir, Pareto sınırını bul."""

from datetime import datetime

from llm_toolkit.answer_eval import AnswerResult, JudgeVerdict
from llm_toolkit.runs import compare_runs, load_runs, pareto_front, runs_table, save_run


def result(qid: str, passed: bool, cost: float = 0.01) -> AnswerResult:
    return AnswerResult(
        question_id=qid,
        category="olgu",
        question=f"{qid}?",
        answer="14 gün" if passed else "bilmiyorum",
        keyword_score=1.0 if passed else 0.0,
        refused=False,
        verdict=JudgeVerdict(correct=passed, faithful=True, reason="test"),
        answer_cost=cost,
        judge_cost=0.0,
        latency_s=1.0,
    )


SETUP = {"cevap_modeli": "claude-haiku-5-5", "yargıç": "claude-sonnet-5-5", "k": 3}


def test_save_and_load_roundtrip(tmp_path):
    path = save_run(
        [result("a", True), result("b", False)],
        SETUP,
        tmp_path,
        now=datetime(2026, 11, 23, 14, 5, 9),
    )
    assert path.name == "20261123-140509.json"
    [run] = load_runs(tmp_path)
    assert run.setup == SETUP
    assert run.summary["ifade_başarısı"] == 0.5
    assert run.results[1].verdict.correct is False


def test_runs_are_sorted_oldest_first_and_broken_files_skipped(tmp_path):
    save_run([result("a", True)], SETUP, tmp_path, now=datetime(2026, 11, 24))
    save_run([result("a", False)], SETUP, tmp_path, now=datetime(2026, 11, 23))
    (tmp_path / "bozuk.json").write_text("{yarım", encoding="utf-8")
    runs = load_runs(tmp_path)
    assert [r.run_id for r in runs] == ["20261123-000000", "20261124-000000"]


def test_missing_folder_means_no_runs(tmp_path):
    assert load_runs(tmp_path / "yok") == []


def test_runs_table_uses_judge_score_as_quality(tmp_path):
    save_run([result("a", True), result("b", False)], SETUP, tmp_path, now=datetime(2026, 11, 23))
    [row] = runs_table(load_runs(tmp_path))
    assert row["kalite"] == 0.5  # Yargıç varsa kalite = yargıcın "doğru" oranı
    assert row["cevap_modeli"] == "claude-haiku-5-5"
    assert row["soru_başı_cevap_maliyeti"] == 0.01  # (0.01 + 0.01) / 2 soru


def test_pareto_ignores_judge_cost(tmp_path):
    # Aynı cevaplar, biri pahalı yargıçla ölçülmüş: yargıç maliyeti Pareto'yu etkilememeli
    cheap = [result("a", True, cost=0.001)]
    judged = [r.model_copy(update={"judge_cost": 0.5}) for r in cheap]
    save_run(judged, SETUP, tmp_path, now=datetime(2026, 11, 23))
    save_run([result("a", True, cost=0.002)], SETUP, tmp_path, now=datetime(2026, 11, 24))
    rows = runs_table(load_runs(tmp_path))
    assert [r["pareto"] for r in rows] == [True, False]
    assert rows[0]["yargıç_maliyeti"] == 0.5


def test_pareto_front():
    # (kalite, maliyet): 2. nokta 1.'den hem daha kaliteli hem daha ucuz -> 1. elenir
    points = [(0.6, 0.05), (0.8, 0.01), (0.9, 0.10), (0.7, 0.10)]
    assert pareto_front(points) == [False, True, True, False]


def test_compare_runs_marks_changed_questions(tmp_path):
    save_run([result("a", True), result("b", False)], SETUP, tmp_path, now=datetime(2026, 11, 23))
    save_run([result("a", True), result("b", True)], SETUP, tmp_path, now=datetime(2026, 11, 24))
    old, new = load_runs(tmp_path)
    rows = {r["soru"]: r for r in compare_runs(old, new)}
    assert rows["a"]["değişim"] == ""
    assert rows["b"]["değişim"] == "düzeldi"


# --- Sabit RAG ve agent yan yana (Ders 5.4b) ---


def test_old_runs_count_as_fixed_rag_and_agent_runs_keep_their_method(tmp_path):
    old = [result("a", True), result("b", True)]  # Eski kayıt: setup'ta "yöntem" yok
    agent = [r.model_copy(update={"searches": n}) for r, n in zip(old, [1, 2], strict=True)]
    save_run(old, SETUP, tmp_path, now=datetime(2026, 11, 23))
    save_run(agent, {"yöntem": "agent", **SETUP}, tmp_path, now=datetime(2026, 11, 24))
    rows = runs_table(load_runs(tmp_path))
    assert [r["yöntem"] for r in rows] == ["sabit RAG", "agent"]
    assert [r["ort_arama"] for r in rows] == [1.0, 1.5]
    assert list(rows[0])[:2] == ["çalıştırma", "yöntem"]  # Tabloda hep ikinci sütun
