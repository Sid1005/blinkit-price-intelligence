"""Item — a priced product data-point, mirroring llm_eng week6/pricer/items.py.

Part A (Amazon) uses currency="$"; Part B (Blinkit) uses currency="₹". The
prompt template and push/pull-to-hub mechanics are shared by both parts.
"""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

QUESTION = "What does this cost to the nearest {unit}?"


class Item(BaseModel):
    """A single priced product, ready for the price-prediction prompt."""

    title: str
    category: str
    price: float
    currency: str = "$"
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def _unit_and_prefix(self) -> tuple[str, str]:
        if self.currency == "$":
            return "dollar", "Price is $"
        return "rupee", "Price is ₹"

    def make_prompt(self, text: str) -> None:
        unit, prefix = self._unit_and_prefix()
        question = QUESTION.format(unit=unit)
        self.prompt = f"{question}\n\n{text}\n\n{prefix}{round(self.price)}.00"

    def test_prompt(self) -> str:
        _, prefix = self._unit_and_prefix()
        return self.prompt.split(prefix)[0] + prefix

    def __repr__(self) -> str:
        return f"<{self.title} = {self.currency}{self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list["Item"], val: list["Item"], test: list["Item"]) -> None:
        """Push Item lists to the caller's Hugging Face Hub namespace."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list["Item"], list["Item"], list["Item"]]:
        """Load a curated dataset back from the Hub and reconstruct Items."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
