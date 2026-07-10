"""Dedicated Chroma index over the real-data Blinkit substitution docs.

Deliberately separate from ``app/rag/store.py``'s ``india_commerce_kb``
collection: that one also embeds a synthesized view of the old synthetic
``commerce_data`` catalog (``_catalog_documents()``), which would contaminate
the "real data only" story for this Part B substitution arm. This collection
indexes only ``data/blinkit/substitutions/*.md`` — the 50 real in-aisle
grounding docs built from the 921-SKU scrape, plus ``_no_substitute.md``.
``index.md`` (a pure table of contents with no substitute reasoning) is
skipped — it has nothing worth retrieving.
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

from app import config
from app.nlp.embeddings import STEmbeddingFunction

COLLECTION = "blinkit_substitutions"
DOCS_DIR = Path("data/blinkit/substitutions")


def _chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Paragraph chunking with overlap, sized up from app/rag/store.py's 600/100
    since each bucket doc's per-SKU substitute block reads best kept whole."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + p).strip() if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_index(docs_dir: Path | None = None, reset: bool = True) -> int:
    docs_dir = docs_dir or DOCS_DIR
    cl = _client()
    if reset:
        try:
            cl.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001
            pass
    coll = cl.get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    ids, docs, metas = [], [], []
    for fp in sorted(docs_dir.glob("*.md")):
        if fp.name == "index.md":
            continue
        for i, ch in enumerate(_chunk(fp.read_text(encoding="utf-8"))):
            ids.append(f"{fp.stem}-{i}")
            docs.append(ch)
            metas.append({"source": fp.name, "chunk": i})
    if docs:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def retrieve(query: str, k: int = 5) -> list[dict]:
    coll = _client().get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    if coll.count() == 0:  # lazy build so a fresh checkout works without a manual step
        build_index()
        coll = _client().get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    res = coll.query(query_texts=[query], n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        hits.append({"id": cid, "text": doc, "source": meta.get("source"),
                    "distance": round(dist, 4)})
    return hits


if __name__ == "__main__":
    n = build_index()
    print(f"indexed {n} chunks from {DOCS_DIR}/ into collection '{COLLECTION}'")
    for h in retrieve("substitute for Haldiram's Aloo Bhujia", k=3):
        print(round(h["distance"], 3), h["source"], "-", h["text"][:80].replace("\n", " "))
