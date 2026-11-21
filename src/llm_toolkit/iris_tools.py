"""Modül 1'deki tahmin servisini LLM'in kullanabileceği araçlara dönüştürür."""

import httpx
from pydantic import BaseModel, Field

from llm_toolkit.datatypes import Tool


class IrisMeasurements(BaseModel):
    sepal_length: float = Field(description="Çanak yaprak uzunluğu, cm")
    sepal_width: float = Field(description="Çanak yaprak genişliği, cm")
    petal_length: float = Field(description="Taç yaprak uzunluğu, cm")
    petal_width: float = Field(description="Taç yaprak genişliği, cm")


class NoArguments(BaseModel):
    pass


def make_iris_tools(base_url: str, http: httpx.Client | None = None) -> list[Tool]:
    http = http or httpx.Client(base_url=base_url, timeout=5)

    def predict(args: IrisMeasurements) -> str:
        response = http.post("/predict", json=args.model_dump())
        if response.status_code == 422:
            return f"HATA: Servis ölçüleri reddetti: {response.text}"
        response.raise_for_status()
        return response.text

    def model_info(args: NoArguments) -> str:
        response = http.get("/model/info")
        response.raise_for_status()
        return response.text

    return [
        Tool(
            name="predict_iris",
            description=(
                "Dört ölçüsü (cm) verilen bir iris çiçeğinin türünü tahmin eder. "
                "Tür adını ve her tür için olasılığı döndürür."
            ),
            parameters=IrisMeasurements,
            function=predict,
        ),
        Tool(
            name="iris_model_info",
            description=(
                "Tahmin modelinin sürümünü, tanıdığı türleri ve doğruluk metriklerini verir."
            ),
            parameters=NoArguments,
            function=model_info,
        ),
    ]
