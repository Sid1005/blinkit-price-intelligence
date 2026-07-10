#!/usr/bin/env python3
"""Blinkit Product Scraper using Tavily search/extract and Groq LLM extraction.

This script searches for Blinkit products via Tavily, extracts the contents of
the product and category pages, and uses Groq (llama-3.3-70b-versatile) to
parse clean, structured product signals (pricing, units, stock, discounts).

Two modes:
  --query "chips"   single ad-hoc query (legacy behavior, small sample)
  --full            sweep ~60 Blinkit category queries to build the real,
                    non-synthetic catalog (data/blinkit/blinkit_products.json),
                    checkpointing after every category so a crash/rate-limit
                    doesn't lose progress and re-runs can --resume.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app import config
from app.llm import groq_client

# Colors for CLI output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ~65 Blinkit category queries spanning grocery, snacks/drinks, personal care,
# household, baby/pet, electronics/stationery and pharma — chosen to mirror
# Blinkit's real category taxonomy so the scrape stays representative rather
# than skewed toward a handful of snack categories.
CATEGORY_QUERIES: List[str] = [
    # Grocery & kitchen staples
    "fruits and vegetables", "atta rice and dal", "masala and spices",
    "dry fruits and nuts", "edible oil and ghee", "sugar jaggery and salt",
    "meat fish and eggs", "dairy bread and eggs", "paneer and curd",
    "butter and cheese", "bakery and biscuits", "breakfast and cereals",
    "sauces and spreads", "pickles and chutneys", "frozen food",
    "gourmet and world food", "organic and healthy food",
    # Snacks & drinks
    "chips and namkeen", "chocolates and candies", "sweets and mithai",
    "ice cream and desserts", "cold drinks and soft drinks",
    "fruit juices and energy drinks", "tea and coffee",
    "noodles and pasta", "biscuits and cookies", "cakes and pastries",
    "water bottles and cans", "paan corner",
    # Beauty & personal care
    "skin care products", "hair care products", "bath and body wash",
    "beauty and cosmetics", "oral care and toothpaste",
    "feminine hygiene products", "perfumes and deodorants",
    "men's grooming products", "health supplements and wellness",
    "sexual wellness products",
    # Household
    "cleaning essentials", "detergents and dishwash",
    "home and kitchen appliances", "mops brooms and scrubs",
    "air fresheners and fragrances", "pooja and religious needs",
    "disposable and party supplies",
    # Baby & pet care
    "baby food and diapers", "baby care products",
    "pet food and accessories",
    # Electronics & stationery
    "electronics accessories", "mobile accessories and chargers",
    "audio devices earbuds", "stationery and office supplies",
    "toys and games",
    # Pharma & wellness
    "pharma and wellness products", "ayurveda and herbal products",
    "health devices and monitors", "vitamins and supplements",
    # Misc grocery
    "rice and dal varieties", "spices and masala powders",
    "namkeen and snacks", "milk and milk products",
    "bread and buns", "cooking sauces and ketchup",
    "instant food and ready to eat", "energy and sports drinks",
    "coffee beans and powder",
]


def search_blinkit(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for products on Blinkit using Tavily."""
    print(f"{Colors.BLUE}Searching Tavily for '{query}' on blinkit.com...{Colors.ENDC}")
    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is missing from the environment or .env file.")

    from tavily import TavilyClient
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    # Restrict to blinkit.com
    search_query = f"{query} site:blinkit.com"
    try:
        response = client.search(
            query=search_query,
            max_results=max_results,
            include_domains=["blinkit.com"]
        )
        return response.get("results", [])
    except Exception as e:
        print(f"{Colors.FAIL}Error searching Tavily: {e}{Colors.ENDC}")
        # Try fallback without domain filter if the domain-restricted one failed
        try:
            print(f"{Colors.YELLOW}Retrying search without domain filter...{Colors.ENDC}")
            response = client.search(query=search_query, max_results=max_results)
            return response.get("results", [])
        except Exception as e_inner:
            print(f"{Colors.FAIL}Fallback search also failed: {e_inner}{Colors.ENDC}")
            return []


def extract_page_content(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract contents of URLs using Tavily Extract."""
    if not urls:
        return []

    print(f"{Colors.BLUE}Extracting page contents for {len(urls)} URLs...{Colors.ENDC}")
    from tavily import TavilyClient
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    try:
        response = client.extract(urls=urls)
        return response.get("results", [])
    except Exception as e:
        print(f"{Colors.FAIL}Error extracting pages: {e}{Colors.ENDC}")
        # Fallback to single page extractions if batch fails
        results = []
        for url in urls:
            try:
                print(f"{Colors.YELLOW}Retrying extract for single URL: {url}{Colors.ENDC}")
                res = client.extract(urls=[url])
                results.extend(res.get("results", []))
            except Exception as ex:
                print(f"{Colors.FAIL}Failed to extract {url}: {ex}{Colors.ENDC}")
        return results


def parse_products_with_llm(raw_text: str, source_url: str, category_hint: str | None = None) -> List[Dict[str, Any]]:
    """Use Groq JSON-mode to extract structured products from raw markdown/text."""
    print(f"{Colors.BLUE}Parsing products from {source_url} using Groq...{Colors.ENDC}")

    system_prompt = (
        "You are India Commerce SignalForge, an expert marketplace pricing analyst. "
        "Your task is to parse raw text/markdown from a Blinkit web page and extract "
        "structured product information. Be precise and ground every pricing signal "
        "in the text. Never invent or guess prices or details. If a value is missing or "
        "not explicitly mentioned, use null."
    )

    user_prompt = (
        f"From the web page content below, extract all products, variants, and similar products "
        f"shown. Output a JSON object containing a 'products' key, which is a list of objects "
        f"representing each product item found.\n\n"
        f"Each product item must have the following structure:\n"
        f"- product_name (string, full name of product, e.g., 'Lay\\'s India\\'s Magic Masala Potato Chips')\n"
        f"- brand (string or null, e.g., 'Lay\\'s')\n"
        f"- category (string or null, the product category/aisle shown on the page, e.g., 'Chips & Crisps')\n"
        f"- price_inr (number or null, the actual current selling price in INR, e.g., 25)\n"
        f"- mrp_inr (number or null, the original maximum retail price or cross-out price, e.g., 25 or 30)\n"
        f"- discount_percent (number or null, discount percentage if mentioned, e.g., 10)\n"
        f"- unit (string or null, the quantity/weight, e.g., '58 g' or '107 g' or 'Pack of 2')\n"
        f"- in_stock (boolean, default true; set to false if text explicitly states 'Out of stock' or 'Sold out' for this variant)\n"
        f"- rating (number or null, average rating out of 5 if shown)\n"
        f"- url (string, source URL: '{source_url}')\n\n"
        f"Web Page Content:\n"
        f"====================\n"
        f"{raw_text[:12000]}\n"
        f"====================\n\n"
        f"Ensure the output contains only a strict JSON object with a single 'products' list key."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Use strong Llama 3.3 model for accurate JSON extraction
        response_dict = groq_client.chat_json(messages, model=config.GROQ_MODELS["strong"])
        products = response_dict.get("products", [])

        # Post-process to ensure all fields exist and are normalized
        normalized = []
        for p in products:
            if not p.get("product_name"):
                continue
            p["url"] = p.get("url") or source_url
            if not p.get("category") and category_hint:
                p["category"] = category_hint

            # Clean numeric values
            for field in ["price_inr", "mrp_inr", "discount_percent", "rating"]:
                val = p.get(field)
                if val is not None:
                    try:
                        p[field] = float(val) if '.' in str(val) else int(val)
                    except (ValueError, TypeError):
                        p[field] = None

            # Drop rows with no numeric price — we only keep real priced SKUs.
            if p.get("price_inr") is None:
                continue

            # If discount percent is not present but price and mrp are, compute it
            if p.get("discount_percent") is None and p.get("price_inr") and p.get("mrp_inr"):
                price = p["price_inr"]
                mrp = p["mrp_inr"]
                if mrp > price:
                    p["discount_percent"] = round(((mrp - price) / mrp) * 100)

            normalized.append(p)
        return normalized
    except Exception as e:
        print(f"{Colors.FAIL}Error parsing page with LLM: {e}{Colors.ENDC}")
        return []


def generate_markdown_report(products: List[Dict[str, Any]], query: str) -> str:
    """Create a gorgeous Markdown report summarizing the scraped products."""
    md = [
        f"# Blinkit Scraped Products Report",
        f"",
        f"This report lists products scraped from **Blinkit** using the **Tavily API** and structured by **SignalForge**.",
        f"",
        f"**Search Query:** `{query}`",
        f"**Total Products Found:** {len(products)}",
        f"",
        f"## Product Catalog",
        f"",
        f"| Brand | Product Name | Unit | Price (INR) | MRP (INR) | Discount | Stock Status | Source Link |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for p in products:
        brand = p.get("brand") or "N/A"
        name = p.get("product_name", "Unknown Product")
        unit = p.get("unit") or "N/A"

        price = f"₹{p['price_inr']}" if p.get("price_inr") is not None else "N/A"
        mrp = f"₹{p['mrp_inr']}" if p.get("mrp_inr") is not None else "N/A"

        discount = f"{p['discount_percent']}% OFF" if p.get("discount_percent") else "-"
        stock = "🟢 In Stock" if p.get("in_stock", True) else "🔴 Out of Stock"

        url_text = f"[Link]({p['url']})" if p.get("url") else "N/A"

        md.append(f"| {brand} | {name} | {unit} | {price} | {mrp} | {discount} | {stock} | {url_text} |")

    return "\n".join(md)


def _dedupe_key(p: Dict[str, Any]) -> tuple:
    return (p["product_name"].strip().lower(), (p.get("unit") or "").strip().lower())


def run_full_sweep(output_json: str, output_md: str, max_results: int,
                    sleep_s: float, resume: bool, limit: int | None) -> List[Dict[str, Any]]:
    """Sweep CATEGORY_QUERIES against real Blinkit pages, checkpointing after each category."""
    all_products: List[Dict[str, Any]] = []
    seen = set()
    done_categories: set = set()

    if resume and os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for p in existing:
            key = _dedupe_key(p)
            if key not in seen:
                seen.add(key)
                all_products.append(p)
            done_categories.add((p.get("category") or "").strip().lower())
        print(f"{Colors.YELLOW}Resuming: loaded {len(all_products)} existing products, "
              f"{len(done_categories)} categories already touched.{Colors.ENDC}")

    queries = CATEGORY_QUERIES[:limit] if limit else CATEGORY_QUERIES
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)

    for i, query in enumerate(queries, 1):
        print(f"\n{Colors.HEADER}{Colors.BOLD}[{i}/{len(queries)}] Category: {query}{Colors.ENDC}")
        if resume and query.strip().lower() in done_categories:
            print(f"{Colors.YELLOW}Already covered in a previous run — skipping.{Colors.ENDC}")
            continue

        try:
            results = search_blinkit(query, max_results=max_results)
            urls = [r["url"] for r in results if r.get("url")]
            if not urls:
                print(f"{Colors.YELLOW}No results for '{query}', skipping.{Colors.ENDC}")
                continue

            extracted = extract_page_content(urls)
            new_count = 0
            for page in extracted:
                raw_text = page.get("raw_content") or ""
                url = page.get("url") or ""
                if not raw_text:
                    continue
                products = parse_products_with_llm(raw_text, url, category_hint=query)
                for p in products:
                    key = _dedupe_key(p)
                    if key not in seen:
                        seen.add(key)
                        all_products.append(p)
                        new_count += 1

            print(f"{Colors.GREEN}+{new_count} new unique products (total: {len(all_products)}){Colors.ENDC}")

        except Exception as e:  # noqa: BLE001 — one bad category shouldn't kill the sweep
            print(f"{Colors.FAIL}Category '{query}' failed: {e}{Colors.ENDC}")

        # Checkpoint after every category so a crash/rate-limit never loses progress.
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)

        if sleep_s > 0 and i < len(queries):
            time.sleep(sleep_s)

    report_md = generate_markdown_report(all_products, "full category sweep")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    return all_products


def main():
    parser = argparse.ArgumentParser(description="Scrape products from Blinkit using Tavily search/extract and Groq LLM.")
    parser.add_argument("--query", type=str, default=None, help="Single ad-hoc product query (legacy small-sample mode)")
    parser.add_argument("--full", action="store_true", help="Sweep ~65 Blinkit category queries for the full real catalog")
    parser.add_argument("--max-results", type=int, default=4, help="Max search results (URLs) per query/category")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds to sleep between categories in --full mode")
    parser.add_argument("--resume", action="store_true", help="Resume a --full sweep, skipping categories already in the output file")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N categories (for testing --full cheaply)")
    parser.add_argument("--output-json", type=str, default="data/blinkit/blinkit_products.json", help="Path to save the JSON output")
    parser.add_argument("--output-md", type=str, default="data/blinkit/blinkit_products.md", help="Path to save the Markdown report")
    args = parser.parse_args()

    print(f"{Colors.HEADER}{Colors.BOLD}=== Blinkit Product Scraper (Tavily + Groq) ==={Colors.ENDC}\n")

    if args.full:
        products = run_full_sweep(args.output_json, args.output_md, args.max_results,
                                   args.sleep, args.resume, args.limit)
        if not products:
            print(f"{Colors.FAIL}Full sweep produced zero products.{Colors.ENDC}")
            sys.exit(1)
        print(f"\n{Colors.GREEN}{Colors.BOLD}Full sweep complete: {len(products)} unique real, priced SKUs.{Colors.ENDC}")
        print(f"Saved to: {Colors.BOLD}{args.output_json}{Colors.ENDC}")
        return

    query = args.query or "chips"

    # Step 1: Search
    results = search_blinkit(query, max_results=args.max_results)
    if not results:
        print(f"{Colors.FAIL}No search results found for '{query}' on Blinkit.{Colors.ENDC}")
        sys.exit(1)

    urls = [r["url"] for r in results if r.get("url")]
    print(f"{Colors.GREEN}Found {len(urls)} product pages/links on blinkit.com{Colors.ENDC}")
    for url in urls:
        print(f" - {url}")

    # Step 2: Extract Content
    extracted = extract_page_content(urls)
    if not extracted:
        print(f"{Colors.FAIL}Failed to extract content from any of the URLs.{Colors.ENDC}")
        sys.exit(1)

    # Step 3: LLM Parsing and Data Aggregation
    all_products = []
    seen_combinations = set()  # Avoid duplicates based on name + unit

    for page in extracted:
        raw_text = page.get("raw_content") or ""
        url = page.get("url") or ""
        if not raw_text:
            continue

        products = parse_products_with_llm(raw_text, url, category_hint=query)
        for p in products:
            combo = _dedupe_key(p)
            if combo not in seen_combinations:
                seen_combinations.add(combo)
                all_products.append(p)

    if not all_products:
        print(f"{Colors.FAIL}No products could be parsed from the pages.{Colors.ENDC}")
        sys.exit(1)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Successfully extracted {len(all_products)} unique products/SKUs!{Colors.ENDC}\n")

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)

    # Save JSON output
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)
    print(f"Saved raw JSON catalog to: {Colors.BOLD}{args.output_json}{Colors.ENDC}")

    # Save Markdown report
    report_md = generate_markdown_report(all_products, query)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved Markdown report to: {Colors.BOLD}{args.output_md}{Colors.ENDC}")

    # Print beautiful summary table to console
    print(f"\n{Colors.BOLD}--- Scraped Products Summary ({query}) ---{Colors.ENDC}")
    print(f"{'Brand':<15} | {'Product Name':<45} | {'Unit':<10} | {'Price':<8} | {'Stock':<12}")
    print("-" * 90)
    for p in all_products[:15]:  # Show top 15 in console
        brand = (p.get("brand") or "N/A")[:15]
        name = p.get("product_name")[:45]
        unit = p.get("unit") or "N/A"
        price = f"₹{p['price_inr']}" if p.get("price_inr") is not None else "N/A"
        stock = "In Stock" if p.get("in_stock", True) else "Out of Stock"
        print(f"{brand:<15} | {name:<45} | {unit:<10} | {price:<8} | {stock:<12}")

    if len(all_products) > 15:
        print(f"... and {len(all_products) - 15} more items.")

    print(f"\n{Colors.GREEN}Done!{Colors.ENDC}")


if __name__ == "__main__":
    main()
