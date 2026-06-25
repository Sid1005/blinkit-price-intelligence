"""Chroma vector store over the knowledge base + catalog (week 5).

Covers: document loading + recursive/semantic chunking, optional LLM-generated chunk
headlines, sentence-transformer embeddings, Chroma vector store, and hybrid
(vector + lexical) retrieval with reranking and query rewriting.

The index includes both the markdown KB (festival calendar, pricing playbook, policy
docs, substitution guide) and a synthesized text view of the product catalog so the
substitution and deal surfaces can retrieve concrete SKUs.
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

from app import config
from app.nlp.embeddings import STEmbeddingFunction

COLLECTION = "india_commerce_kb"


def _chunk(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Recursive paragraph chunking with overlap (week 5 chunking)."""
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


def _catalog_documents() -> list[tuple[str, str, dict]]:
    """Render each catalog SKU as a retrievable text chunk."""
    from app import commerce_data
    docs = []
    for item in commerce_data.load_catalog():
        unit = f"{config.CURRENCY_SYMBOL}{item['unit_price_inr']}/{item['unit']}" \
            if item.get("unit_price_inr") else "n/a"
        text = (f"{item['title']} (SKU {item['sku']}, {item['category']}) on "
                f"{item['platform']}: pack {item['pack_size']} {item['unit']}, "
                f"MRP {config.CURRENCY_SYMBOL}{item['mrp_inr']}, price "
                f"{config.CURRENCY_SYMBOL}{item['price_inr']} ({item['discount_pct']}% off), "
                f"unit price {unit}, "
                f"{'in stock' if item['in_stock'] else 'OUT OF STOCK'}, "
                f"rating {item['rating']} ({item['review_count']} reviews).")
        docs.append((f"catalog-{item['sku']}", text,
                     {"source": "catalog.md", "sku": item["sku"],
                      "category": item["category"]}))
    return docs


def build_index(kb_dir: Path | None = None, reset: bool = True,
                add_headlines: bool = False) -> int:
    """Load + chunk + embed the KB and catalog into Chroma. Returns chunk count.

    When ``add_headlines`` is True, an LLM generates a one-line headline per policy/KB
    chunk and prepends it (week 5 LLM-generated chunk summaries). Off by default to keep
    indexing fast and offline-friendly.
    """
    kb_dir = kb_dir or config.KB_DIR
    cl = _client()
    if reset:
        try:
            cl.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001
            pass
    coll = cl.get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    ids, docs, metas = [], [], []
    for fp in sorted(kb_dir.glob("*.md")):
        for i, ch in enumerate(_chunk(fp.read_text())):
            if add_headlines:
                ch = _with_headline(ch)
            ids.append(f"{fp.stem}-{i}")
            docs.append(ch)
            metas.append({"source": fp.name, "chunk": i})
    for cid, text, meta in _catalog_documents():
        ids.append(cid)
        docs.append(text)
        metas.append(meta)
    if docs:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def _with_headline(chunk: str) -> str:
    from app.llm import groq_client, router
    try:
        headline = groq_client.chat(
            [{"role": "system", "content": "Write a 6-word headline for this text. Output only the headline."},
             {"role": "user", "content": chunk[:800]}],
            model=router.route("summarize"), max_tokens=20, temperature=0.0).strip()
        return f"[{headline}] {chunk}"
    except Exception:  # noqa: BLE001
        return chunk


def _lexical_overlap(query: str, text: str) -> float:
    q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    if not q:
        return 0.0
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q & t) / len(q)


def retrieve(query: str, k: int = 4, rerank: bool = True, k_initial: int = 10,
             where: dict | None = None) -> list[dict]:
    """Retrieve top-k chunks via hybrid vector + lexical reranking.

    When rerank=True, fetch k_initial candidates by vector similarity then re-rank by a
    blend of vector score and lexical overlap (improves context precision without an
    extra model). Optional ``where`` metadata filter narrows the search.
    """
    coll = _client().get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    if coll.count() == 0:  # lazy build so imported/deployed apps work without a manual step
        build_index()
        coll = _client().get_or_create_collection(COLLECTION, embedding_function=STEmbeddingFunction())
    n = max(k_initial, k) if rerank else k
    kwargs = {"query_texts": [query], "n_results": n}
    if where:
        kwargs["where"] = where
    res = coll.query(**kwargs)
    cands = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        vec_score = 1.0 / (1.0 + dist)
        lex = _lexical_overlap(query, doc)
        score = 0.65 * vec_score + 0.35 * lex if rerank else vec_score
        cands.append({"id": cid, "text": doc, "source": meta.get("source"),
                      "distance": round(dist, 4), "lexical": round(lex, 3),
                      "rerank_score": round(score, 4)})
    cands.sort(key=lambda c: c["rerank_score"], reverse=True)
    return cands[:k]


def rewrite_query(query: str) -> str:
    """LLM query rewriting for sparse/colloquial queries (advanced retrieval, week 5)."""
    from app.llm import groq_client, router
    try:
        return groq_client.chat(
            [{"role": "system", "content": "Rewrite the shopping query into a clear, "
              "keyword-rich search query for an Indian commerce knowledge base. Output only the query."},
             {"role": "user", "content": query}],
            model=router.route("summarize"), max_tokens=40, temperature=0.0).strip() or query
    except Exception:  # noqa: BLE001
        return query


if __name__ == "__main__":
    n = build_index()
    print(f"indexed {n} chunks")
    for h in retrieve("Diwali discount on phones", k=3):
        print(round(h["rerank_score"], 3), h["source"], "-", h["text"][:70])
