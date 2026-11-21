"""Yapılandırılmış çıktı için örnek şema: müşteri yorumu analizi."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["olumlu", "olumsuz", "karışık"] = Field(description="Genel duygu")
    score: int = Field(ge=1, le=5, description="Tahmini puan: 1 çok kötü, 5 çok iyi")
    topics: list[str] = Field(max_length=5, description="Yorumda geçen ana konular, en fazla 5")
    summary: str = Field(max_length=200, description="Yorumun tek cümlelik Türkçe özeti")
