"""Structured outputs for the agent ensemble (week 8) via pydantic.

One ``Brief`` envelope carries exactly one decision variant depending on the routed
intent:
  * predictor  -> PriceForecast    (5-arm price comparison)
  * substitute -> SubstitutionSet
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field

Intent = Literal["predictor", "substitute"]

SignalLabel = Literal[
    "festival_discount", "demand_spike",
    "complaint_policy", "catalog_substitution", "noise",
]

Confidence = Literal["low", "medium", "high"]


class Signal(BaseModel):
    label: SignalLabel
    snippet: str
    confidence: Confidence = "medium"
    source: str = ""


# --- Deal surface ----------------------------------------------------------------
class PriceForecast(BaseModel):
    kind: Literal["price_forecast"] = "price_forecast"
    sku: str = ""
    title: str = ""
    currency: str = "INR"
    low_inr: float = Field(ge=0)
    high_inr: float = Field(ge=0)
    point_inr: float = Field(ge=0)
    unit_price_inr: float | None = None
    unit: str | None = None
    festival_context: str | None = None
    rationale: str = ""
    estimator: str = ""
    # Per-arm price comparison: {arm_name: {"price": float|None, "status": str}, ...}
    # plus an "ensemble_reasoning" string. Populated by the predictor surface.
    comparison: dict = Field(default_factory=dict)
    # RAG retrieval results displayed in the UI to show retrieval is happening.
    rag_context: dict | None = None


# --- Substitution surface --------------------------------------------------------
class SubstituteCandidate(BaseModel):
    sku: str
    title: str
    platform: str = ""
    price_inr: float = Field(ge=0)
    unit_price_inr: float | None = None
    in_stock: bool = True
    rating: float = 0.0
    score: float = Field(ge=0, le=1)
    reason: str = ""


class SubstitutionSet(BaseModel):
    kind: Literal["substitution_set"] = "substitution_set"
    original_sku: str = ""
    original_title: str = ""
    reason_for_substitution: str = ""
    candidates: list[SubstituteCandidate] = []
    value_improvement_pct: float | None = None


Decision = Union[PriceForecast, SubstitutionSet]


class Brief(BaseModel):
    """Single envelope returned by the engine for any intent."""
    intent: Intent
    query: str
    signals: list[Signal] = []
    decision: Decision | None = Field(default=None, discriminator="kind")
    trend_strength: Confidence = "low"
    citations: list[str] = []
    notes: str = ""
