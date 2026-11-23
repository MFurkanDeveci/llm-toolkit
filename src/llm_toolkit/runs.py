"""Deney kaydı: her cevap ölçümünü ayarları ve sonuçlarıyla birlikte bir JSON dosyasına yazar.

Ölçümü bir kez yapıp unutmak yerine kaydediyoruz; böylece "Sonnet'e geçince ne değişti?",
"dün iyiydi, bugün neden kötü?" gibi soruları eski çalıştırmalarla karşılaştırarak
cevaplayabiliyoruz. Her çalıştırma data/eval/runs/ altında ayrı bir dosya.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from llm_toolkit.answer_eval import AnswerResult, summarize_answers

logger = logging.getLogger(__name__)

RUNS_DIR = Path("data/eval/runs")


class RunRecord(BaseModel):
    run_id: str  # Dosya adıyla aynı: tarih-saat, örn. 20261123-140509
    created_at: datetime
    setup: dict[str, Any]  # Hangi ayarlarla çalıştı: model, yargıç, k...
    summary: dict[str, Any]  # summarize_answers'ın "tümü" satırı
    results: list[AnswerResult]  # Soru soru sonuçlar (cevap metinleri dahil)


def save_run(
    results: list[AnswerResult],
    setup: dict[str, Any],
    folder: Path = RUNS_DIR,
    *,
    now: datetime | None = None,
) -> Path:
    """Çalıştırmayı kaydeder ve dosya yolunu döndürür."""
    now = now or datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}"
    summary = next(row for row in summarize_answers(results) if row["tür"] == "tümü")
    record = RunRecord(run_id=run_id, created_at=now, setup=setup, summary=summary, results=results)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_runs(folder: Path = RUNS_DIR) -> list[RunRecord]:
    """Tüm çalıştırmaları eskiden yeniye sıralı okur. Okunamayan dosyaları atlar."""
    runs = []
    for path in sorted(folder.glob("*.json")):
        try:
            runs.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except ValidationError:
            logger.warning("Okunamayan çalıştırma dosyası atlandı: %s", path)
    return sorted(runs, key=lambda r: r.created_at)


def quality(summary: dict[str, Any]) -> float:
    """Tek bir kalite sayısı: yargıç varsa onun "doğru" oranı, yoksa ifade kontrolü."""
    judged = summary.get("yargıç_doğru")
    return judged if judged is not None else summary["ifade_başarısı"]


def pareto_front(points: list[tuple[float, float]]) -> list[bool]:
    """(kalite, maliyet) noktalarından hangileri Pareto sınırında?

    Bir nokta, başka hiçbir nokta ondan hem daha kaliteli (ya da eşit) hem daha ucuz (ya da
    eşit) değilse sınırdadır. Sınırda olmayan bir ayarı seçmek için sebep yoktur: aynı ya da
    daha az parayla daha iyi bir seçenek zaten var.
    """
    return [
        not any(
            q2 >= q and c2 <= c and (q2, c2) != (q, c)
            for j, (q2, c2) in enumerate(points)
            if j != i
        )
        for i, (q, c) in enumerate(points)
    ]


def answer_cost_per_question(run: RunRecord) -> float:
    """Kullanıcıya hizmet verirken ödenecek maliyet: arama + cevap, soru başına.

    Yargıç maliyeti buna dahil değil: yargıç sadece ölçüm yaparken çalışır.
    """
    return sum(r.answer_cost for r in run.results) / len(run.results)


def runs_table(runs: list[RunRecord]) -> list[dict[str, Any]]:
    """Her çalıştırma için bir satır: ayarlar + özet ölçütler + Pareto bilgisi.

    Pareto sınırı (kalite, soru başı cevap maliyeti) ile çizilir; yargıç maliyeti ayrı sütunda.
    """
    points = [(quality(r.summary), answer_cost_per_question(r)) for r in runs]
    front = pareto_front(points)
    return [
        {
            "çalıştırma": r.run_id,
            **r.setup,
            "kalite": q,
            "ifade_başarısı": r.summary["ifade_başarısı"],
            "yargıç_sadık": r.summary["yargıç_sadık"],
            "soru_başı_cevap_maliyeti": c,
            "yargıç_maliyeti": sum(x.judge_cost for x in r.results),
            "ort_süre": r.summary["ort_süre"],
            "pareto": on_front,
        }
        for r, (q, c), on_front in zip(runs, points, front, strict=True)
    ]


def passed(result: AnswerResult) -> bool:
    """Tek bir sorunun başarılı sayılıp sayılmadığı: yargıç varsa onun kararı."""
    return result.verdict.correct if result.verdict else result.keyword_pass


def compare_runs(old: RunRecord, new: RunRecord) -> list[dict[str, Any]]:
    """İki çalıştırmayı soru soru karşılaştırır: hangi soru düzeldi, hangisi bozuldu?"""
    old_by_id = {r.question_id: r for r in old.results}
    rows = []
    for r in new.results:
        before = old_by_id.get(r.question_id)
        if before is None:
            continue  # Yeni eklenmiş soru: eski çalıştırmada yok
        was, now = passed(before), passed(r)
        change = "" if was == now else ("düzeldi" if now else "bozuldu")
        rows.append(
            {
                "soru": r.question_id,
                "tür": r.category,
                "önce": was,
                "sonra": now,
                "değişim": change,
                "önceki cevap": before.answer,
                "yeni cevap": r.answer,
            }
        )
    return rows
