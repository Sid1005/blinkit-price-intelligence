"""RAG generation grounded in retrieved context (week 5).

Uses LangChain prompt templating to assemble a grounded prompt, retrieves from the
Chroma store, and generates with Groq. Returns the answer + cited context so the eval
harness can score faithfulness / context precision / groundedness.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.llm import groq_client, router
from app.rag import store

RAG_SYSTEM = (
    "You are India Commerce SignalForge's grounded analyst for the Indian marketplace. "
    "Answer ONLY from the provided context. Cite the source filename in brackets after "
    "each claim, e.g. [return_refund_policy.md]. Prices are in INR (\u20b9) and are demo "
    "data, not live quotes. If the context does not contain the answer, say "
    "'insufficient evidence'."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nGrounded answer:"),
])


def answer(question: str, k: int = 4, model: str | None = None,
           rewrite: bool = False) -> dict:
    q = store.rewrite_query(question) if rewrite else question
    hits = store.retrieve(q, k=k)
    if not hits:
        return {"question": question, "answer": "insufficient evidence",
                "contexts": [], "sources": []}
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    msgs = _PROMPT.format_messages(context=context, question=question)
    payload = [{"role": "system" if m.type == "system" else "user", "content": m.content}
               for m in msgs]
    ans = groq_client.chat(payload, model=model or router.route("synthesis"),
                           temperature=0.1, max_tokens=600)
    return {"question": question, "answer": ans, "contexts": hits,
            "sources": sorted({h["source"] for h in hits})}
