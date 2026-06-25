"""Web evidence ingestion + cleaning + LLM summarization (week 1).

Covers: web search/scraping (Tavily primary, BeautifulSoup fallback), summarization
(Groq), system/user prompt engineering, markdown output, and strict JSON extraction
of pricing/SKU signals.

Tavily is the primary evidence source because Blinkit/Amazon-style pages are
JS-heavy and bot-protected; Tavily returns clean, ranked snippets. A direct
``requests`` + BeautifulSoup path is kept as an offline-friendly fallback.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from app import config
from app.llm import groq_client, router

HEADERS = {"User-Agent": "Mozilla/5.0 (SignalForge India commerce research agent)"}

ANALYST_SYSTEM = (
    "You are India Commerce SignalForge, an Indian-marketplace pricing & product "
    "analyst. You read messy Blinkit/Zepto/Amazon.in/Flipkart product and pricing "
    "pages and extract crisp, evidence-grounded signals in INR. You never invent "
    "prices; if a page lacks pricing or stock info, you say so explicitly. Treat all "
    "scraped figures as unverified demo evidence, not live quotes."
)

_tavily_client = None


def _tavily():
    global _tavily_client
    if _tavily_client is None:
        if not config.TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY missing — set it in the repo-root .env")
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _tavily_client


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    source: str = "tavily"


@dataclass
class Evidence:
    """A ranked web-evidence snippet for the scout agent."""
    title: str
    url: str
    content: str
    score: float = 0.0


def search(query: str, max_results: int = 5,
           include_domains: list[str] | None = None) -> list[Evidence]:
    """Tavily web search for Indian commerce evidence (prices, deals, reviews).

    Defaults to Indian marketplace + news domains so results stay on-topic.
    """
    include_domains = include_domains or [
        "amazon.in", "flipkart.com", "blinkit.com", "bigbasket.com",
        "zepto.com", "gadgets360.com", "smartprix.com", "91mobiles.com",
    ]
    try:
        res = _tavily().search(query, max_results=max_results,
                               include_domains=include_domains,
                               search_depth="advanced")
    except Exception:  # noqa: BLE001 — network/domain filters can fail; retry broad
        try:
            res = _tavily().search(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001
            return [Evidence(title="search failed", url="", content=str(e)[:200], score=0.0)]
    out = []
    for r in res.get("results", []):
        out.append(Evidence(title=(r.get("title") or "")[:200], url=r.get("url", ""),
                            content=(r.get("content") or "")[:1500],
                            score=_safe_float(r.get("score"))))
    return out


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def extract(url: str) -> ScrapedPage:
    """Extract clean page text via Tavily, falling back to requests+BeautifulSoup."""
    try:
        res = _tavily().extract(url)
        results = res.get("results") or []
        if results:
            raw = results[0].get("raw_content", "") or ""
            return ScrapedPage(url=url, title=url, text=raw[:20000], source="tavily")
    except Exception:  # noqa: BLE001 — fall through to direct fetch
        pass
    return _scrape_direct(url)


def _scrape_direct(url: str, timeout: int = 12) -> ScrapedPage:
    """Direct requests + BeautifulSoup fallback (offline-friendly)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — network is best-effort
        return ScrapedPage(url=url, title="", text="", ok=False, error=str(e),
                           source="requests")
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    links = [a.get("href") for a in soup.find_all("a", href=True)][:40]
    return ScrapedPage(url=url, title=title, text=text[:20000], links=links,
                       source="requests")


# Backwards-compatible alias used by tools/agents.
def scrape(url: str) -> ScrapedPage:
    return extract(url)


def summarize(page: ScrapedPage, model: str | None = None) -> str:
    """Summarize a page into a markdown India-commerce signal brief."""
    user = (
        f"Source URL: {page.url}\nTitle: {page.title}\n\n"
        f"PAGE TEXT (truncated):\n{page.text[:8000]}\n\n"
        "Produce a markdown brief with sections: **Summary** (2-3 sentences), "
        "**Price signals (INR)** (bullets with \u20b9 amounts, or 'none found'), "
        "**Stock/availability** (bullets), **Festival/sale context** (bullets), "
        "**Confidence** (low/med/high with one reason)."
    )
    messages = [{"role": "system", "content": ANALYST_SYSTEM},
                {"role": "user", "content": user}]
    return groq_client.chat(messages, model=model or router.route("summarize"),
                            temperature=0.2, max_tokens=700)


def extract_signals_json(page_or_text, model: str | None = None) -> dict:
    """Strict-JSON extraction of links / SKUs / price signals (week 1 JSON output)."""
    text = page_or_text.text if isinstance(page_or_text, ScrapedPage) else str(page_or_text)
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM + " Output strict JSON only."},
        {"role": "user", "content": (
            "From the text below, extract JSON with keys: "
            "skus (list of {name, price_inr|null, mrp_inr|null, platform|null}), "
            "deal_signals (list of short strings), urls (list of strings). "
            "Use null when unknown; never invent prices.\n\nTEXT:\n" + text[:6000])},
    ]
    return groq_client.chat_json(messages, model=model or router.route("summarize"))


def stream_summary(page: ScrapedPage, model: str | None = None):
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": f"Summarize this page as a 4-bullet INR price brief:\n\n{page.text[:6000]}"},
    ]
    yield from groq_client.stream(messages, model=model or router.route("summarize"), max_tokens=500)


def gather_evidence(query: str, max_results: int = 5) -> list[dict]:
    """Convenience: search -> normalize to evidence records the scout agent consumes."""
    results = search(query, max_results=max_results)
    return [{"text": e.content, "source": e.url or "tavily", "title": e.title,
             "score": e.score, "date": None} for e in results if e.content]


if __name__ == "__main__":
    os.environ.setdefault("TAVILY_API_KEY", config.TAVILY_API_KEY)
    ev = gather_evidence("iPhone 15 price Amazon India Big Billion Days", max_results=3)
    print(json.dumps(ev, indent=2)[:2000])
