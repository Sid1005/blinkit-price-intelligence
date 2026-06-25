"""Tool / function-calling layer (week 2).

Defines JSON tool schemas and a dispatcher used by the scout agent's Groq
tool-calling loop. Tools:
  * catalog_lookup    — find a SKU in the curated catalog.
  * normalize_unit    — per-kg / per-litre / per-piece / per-100ml unit price.
  * find_substitutes  — ranked in-catalog alternatives for a SKU.
  * policy_lookup     — retrieve a policy/KB snippet for grounding.
  * web_search        — Tavily web evidence for live-ish price/deal context.
  * draft_notification — compose a deal/escalation alert payload.
"""
from __future__ import annotations

import json

from app import commerce_data, config
from app.ingest import scraper
from app.llm import groq_client
from app.rag import store

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "catalog_lookup",
        "description": "Look up a product SKU in the curated Indian catalog by name or SKU id.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "normalize_unit",
        "description": "Compute a normalised unit price (per kg/litre/piece/100ml).",
        "parameters": {"type": "object", "properties": {
            "price_inr": {"type": "number"}, "pack_size": {"type": "number"},
            "unit": {"type": "string"}}, "required": ["price_inr", "pack_size", "unit"]}}},
    {"type": "function", "function": {
        "name": "find_substitutes",
        "description": "Find ranked in-catalog substitute SKUs for a given product.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "policy_lookup",
        "description": "Retrieve a policy/knowledge-base snippet for grounding a resolution.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (Tavily) for Indian commerce price/deal evidence.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


# --- tool implementations --------------------------------------------------------
def normalize_unit(price_inr: float, pack_size: float, unit: str) -> dict:
    """Return a comparable unit price. Supports kg, litre, piece, ml, g."""
    unit = (unit or "").lower().strip()
    if not pack_size or pack_size <= 0:
        return {"error": "pack_size must be > 0"}
    if unit in ("kg", "litre", "l", "piece", "pc", "pcs"):
        norm_unit = "litre" if unit == "l" else ("piece" if unit in ("pc", "pcs") else unit)
        return {"unit_price_inr": round(price_inr / pack_size, 2),
                "basis": f"per {norm_unit}"}
    if unit in ("ml", "g"):
        per_100 = round(price_inr / pack_size * 100, 2)
        return {"unit_price_inr": per_100, "basis": f"per 100{unit}"}
    return {"unit_price_inr": round(price_inr / pack_size, 2), "basis": f"per {unit}"}


def find_substitutes(query: str, k: int = 3) -> list[dict]:
    """Rank in-catalog substitutes for a SKU using the substitution guide weights."""
    catalog = commerce_data.load_catalog()
    original = commerce_data.find_sku(query)
    if not original:
        return []
    group = original.get("substitute_group")
    pool = [c for c in catalog if c["sku"] != original["sku"] and (
        (group and c.get("substitute_group") == group) or
        (not group and c["category"] == original["category"]))]
    ranked = []
    for c in pool:
        ranked.append({**c, "_score": _substitute_score(original, c)})
    ranked.sort(key=lambda c: c["_score"], reverse=True)
    return ranked[:k]


def _substitute_score(original: dict, cand: dict) -> float:
    """Weighted blend per substitution_guide.md: fit/value/availability/quality."""
    fit = 1.0 if cand.get("substitute_group") == original.get("substitute_group") and \
        original.get("substitute_group") else (0.6 if cand["category"] == original["category"] else 0.2)
    o_up = original.get("unit_price_inr") or original["price_inr"]
    c_up = cand.get("unit_price_inr") or cand["price_inr"]
    value = 1.0 if c_up <= o_up else max(0.0, 1.0 - (c_up - o_up) / max(o_up, 1))
    availability = 1.0 if cand["in_stock"] else 0.0
    quality = min(1.0, cand["rating"] / 5.0)
    return round(0.30 * fit + 0.30 * value + 0.20 * availability + 0.20 * quality, 4)


def policy_lookup(query: str, k: int = 3) -> list[dict]:
    return store.retrieve(query, k=k)


_REQUIRED_ARGS = {
    "catalog_lookup": ["query"], "find_substitutes": ["query"],
    "policy_lookup": ["query"], "web_search": ["query"],
    "normalize_unit": ["price_inr", "pack_size", "unit"],
}


def dispatch(name: str, args: dict) -> str:
    if not isinstance(args, dict):
        return json.dumps({"error": "tool args must be an object"})
    missing = [k for k in _REQUIRED_ARGS.get(name, []) if k not in args]
    if missing:
        return json.dumps({"error": f"missing args for {name}: {missing}"})
    if name == "catalog_lookup":
        item = commerce_data.find_sku(args["query"])
        return json.dumps(item or {"error": "not found"}, ensure_ascii=False)
    if name == "normalize_unit":
        return json.dumps(normalize_unit(args["price_inr"], args["pack_size"], args["unit"]))
    if name == "find_substitutes":
        return json.dumps(find_substitutes(args["query"]), ensure_ascii=False)
    if name == "policy_lookup":
        return json.dumps(policy_lookup(args["query"]), ensure_ascii=False)
    if name == "web_search":
        return json.dumps(scraper.gather_evidence(args["query"], max_results=3), ensure_ascii=False)
    return json.dumps({"error": f"unknown tool {name}"})


def run_with_tools(user_msg: str, model: str | None = None, max_rounds: int = 3) -> dict:
    """Minimal tool-calling loop using Groq function calling (week 2 agent intro)."""
    messages = [
        {"role": "system", "content": "You are India Commerce SignalForge's scout. Use "
         "tools to gather catalog, unit-price, substitute, policy, and web evidence "
         "before answering. Prices are INR demo data. Be concise."},
        {"role": "user", "content": user_msg},
    ]
    calls = []
    cl = groq_client.client()
    for _ in range(max_rounds):
        resp = cl.chat.completions.create(
            model=model or config.GROQ_MODELS["strong"], messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.2, max_tokens=700)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content, "tool_calls": calls}
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            result = dispatch(tc.function.name, args)
            calls.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.function.name, "content": result})
    final = groq_client.chat(messages, model=model, max_tokens=500)
    return {"answer": final, "tool_calls": calls}
