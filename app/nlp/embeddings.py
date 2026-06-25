"""Sentence-transformers embeddings (week 3) — shared by the RAG store."""
from __future__ import annotations

from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.HF_EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


class STEmbeddingFunction:
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return embed(input)

    def name(self) -> str:  # chroma >=0.5 requires a name
        return "sentence-transformers-all-MiniLM-L6-v2"

    def embed_documents(self, input):  # noqa: A002
        return embed(list(input))

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return embed_one(input)
        return embed(list(input))

    # Chroma >=1.x persistence hooks (recreate the EF from stored config).
    def get_config(self) -> dict:
        return {"model_name": config.HF_EMBED_MODEL}

    @staticmethod
    def build_from_config(cfg: dict) -> "STEmbeddingFunction":
        return STEmbeddingFunction()
