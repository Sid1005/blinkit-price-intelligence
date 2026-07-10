"""Resolve the user's own Hugging Face namespace and push/pull curated datasets.

Never pushes to ed-donner/* or any shared namespace — always ``<user>/...``
where ``<user>`` is resolved from HF_TOKEN via whoami() at runtime.
"""
from __future__ import annotations

from functools import lru_cache

from huggingface_hub import HfApi

from app import config
from app.pricer.items import Item


@lru_cache(maxsize=1)
def whoami() -> str:
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing from the environment or .env file.")
    api = HfApi(token=config.HF_TOKEN)
    return api.whoami()["name"]


def dataset_repo(name: str) -> str:
    """e.g. dataset_repo('amazon-pricer-lite') -> '<user>/amazon-pricer-lite'."""
    return f"{whoami()}/{name}"


def push(name: str, train: list[Item], val: list[Item], test: list[Item]) -> str:
    repo_id = dataset_repo(name)
    Item.push_to_hub(repo_id, train, val, test)
    return repo_id


def pull(name: str) -> tuple[list[Item], list[Item], list[Item]]:
    return Item.from_hub(dataset_repo(name))
