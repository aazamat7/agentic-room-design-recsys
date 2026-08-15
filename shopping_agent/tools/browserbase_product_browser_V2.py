# shopping_agent/tools/browserbase_product_browser_V2.py
# ============================================================
# Browserbase Product Retrieval Tool V2
#
# Fixes:
#   1. Uses one Browserbase session per site/search URL.
#      If World Market / Article / Amazon kills a browser context, it only kills that site.
#   2. Search-page extraction only. No product-page enrichment in this retrieval step.
#   3. CSV fallback is always merged and returned.
#   4. No scoring, no ranking, no Gemini, no semantic filtering.
#   5. Returns unranked candidates for ProductVerificationAgent and CrossEncoderRerankingAgent.
# ============================================================

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import (
    parse_qsl,
    quote_plus,
    unquote,
    urlencode,
    urlparse,
    urlunparse,
)

from browserbase import Browserbase
from dotenv import load_dotenv
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)


# ============================================================
# 1. Site configuration
# ============================================================

SITE_SEARCH_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # Article intentionally excluded from DEFAULT_US_SITES.
    # Keep available as fallback/manual use only.
    "article": {
        "name": "Article",
        "search_url": "https://www.article.com/search?query={query}",
        "allowed_domains": ["article.com"],
        "product_url_patterns": ["/product/", "/products/"],
    },
    "rugs_usa": {
        "name": "Rugs USA",
        "search_url": "https://www.rugsusa.com/search?query={query}",
        "allowed_domains": ["rugsusa.com"],
        "product_url_patterns": ["/products/"],
    },
    "nathan_james": {
        "name": "Nathan James",
        "search_url": "https://nathanjames.com/search?q={query}",
        "allowed_domains": ["nathanjames.com"],
        "product_url_patterns": ["/products/"],
    },
    "west_elm": {
        "name": "West Elm",
        "search_url": "https://www.westelm.com/search/results.html?words={query}",
        "allowed_domains": ["westelm.com"],
        "product_url_patterns": ["/products/"],
    },
    "amazon": {
        "name": "Amazon",
        "search_url": "https://www.amazon.com/s?k={query}",
        "allowed_domains": ["amazon.com"],
        "product_url_patterns": ["/dp/", "/gp/product/"],
    },
    "castlery": {
        "name": "Castlery",
        "search_url": "https://www.castlery.com/us/search?query={query}",
        "allowed_domains": ["castlery.com"],
        "product_url_patterns": ["/us/products/", "/products/"],
    },
    "world_market": {
        "name": "World Market",
        "search_url": "https://www.worldmarket.com/search?q={query}",
        "allowed_domains": ["worldmarket.com"],
        "product_url_patterns": ["/p/", "/product/"],
    },
}

# Ordered from likely safer/useful to heavier/noisier.
# Article removed from default because your logs show Article caused context closures.
DEFAULT_US_SITES = [
    "rugs_usa",
    "nathan_james",
    "west_elm",
    "amazon",
    "castlery",
    "world_market",
]

DEFAULT_CSV_PATH = "data/browserbase_results_final_with_ho_05.csv"

CSV_REJECT_TERMS = [
    "touch up kit",
    "touch-up kit",
    "furniture touch up",
    "furniture care",
    "repair marker",
    "wood marker",
    "replacement part",
    "assembly instructions",
]


# ============================================================
# 2. Helpers
# ============================================================

def clean_text(value: Optional[str], max_len: int = 600) -> str:
    if value is None:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value))
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def extract_query_from_input(query: Any) -> str:
    data = safe_json_loads(query)

    if isinstance(data, dict):
        for key in ["browser_query", "query", "user_query", "search_query"]:
            value = data.get(key)
            if value:
                return clean_text(str(value), max_len=1000)

        for nested_key in ["planner_output", "result", "output"]:
            nested = data.get(nested_key)
            if nested:
                nested_query = extract_query_from_input(nested)
                if nested_query:
                    return nested_query

        if data.get("interpreted_need"):
            return clean_text(str(data["interpreted_need"]), max_len=1000)

    return clean_text(str(query), max_len=1000)


def normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        return ""

    parsed = urlparse(url)

    # Amazon sponsored redirects.
    if "amazon.com" in parsed.netloc.lower():
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        redirect_url = params.get("url")
        if redirect_url and redirect_url.startswith("/"):
            return normalize_url("https://www.amazon.com" + unquote(redirect_url))

    clean_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l.startswith("utm_") or key_l in {
            "gclid",
            "fbclid",
            "msclkid",
            "yclid",
            "igshid",
        }:
            continue
        clean_params.append((key, value))

    return urlunparse(
        parsed._replace(
            fragment="",
            query=urlencode(clean_params, doseq=True),
        )
    )


def host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def domain_allowed(url: str, allowed_domains: List[str]) -> bool:
    h = host(url)
    return bool(h) and any(
        domain.lower().replace("www.", "") in h
        for domain in allowed_domains
    )


def extract_price(text: str) -> Optional[str]:
    if not text:
        return None

    price_regex = re.compile(
        r"((?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?|"
        r"(?:\$|USD|£|GBP|€|EUR)\s?[\d,]+(?:\.\d{1,2})?)",
        flags=re.IGNORECASE,
    )
    match = price_regex.search(text)
    return clean_text(match.group(1), max_len=80) if match else None


def query_tokens(query: str) -> List[str]:
    stopwords = {
        "the", "and", "for", "with", "under", "over", "from", "into",
        "best", "find", "show", "give", "want", "need", "room", "home",
        "living", "dining", "decor", "furniture", "dollar", "dollars",
        "usd", "in", "on", "of", "to", "a", "an",
    }
    tokens = [
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", query or "")
        if len(t) >= 3
    ]
    return [t for t in tokens if t not in stopwords]


# ============================================================
# 3. CSV augmentation
# ============================================================

def resolve_csv_path(csv_data_path: Optional[str] = None) -> Optional[Path]:
    candidates = [
        csv_data_path,
        os.environ.get("PRODUCT_CANDIDATE_CSV_PATH"),
        DEFAULT_CSV_PATH,
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path

    return None


def first_non_empty(row: Dict[str, Any], names: List[str]) -> str:
    lower_map = {k.lower(): k for k in row.keys()}
    for name in names:
        actual = lower_map.get(name.lower())
        if actual is None:
            continue
        value = row.get(actual)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def row_matches_query_broadly(row: Dict[str, Any], query: str) -> bool:
    tokens = query_tokens(query)
    if not tokens:
        return True
    text = " ".join(str(v or "") for v in row.values()).lower()
    return any(token in text for token in tokens)


def csv_row_to_product(row: Dict[str, Any], retrieval_rank: int, query: str) -> Optional[Dict[str, Any]]:
    product_url = normalize_url(
        first_non_empty(
            row,
            ["product_url", "url", "link", "href", "product_link", "productUrl", "product url"],
        )
    )
    if not product_url:
        return None

    title = clean_text(
        first_non_empty(
            row,
            [
                "title", "product_title", "product_name", "name",
                "candidate_title", "page_title", "anchorText", "anchor_text",
            ],
        ),
        max_len=180,
    )

    description = clean_text(
        first_non_empty(
            row,
            [
                "description", "product_description", "candidate_text",
                "rawText", "raw_text", "page_description",
                "page_text_excerpt", "text",
            ],
        ),
        max_len=450,
    )

    combined = f"{title} {description}".lower()
    if any(term in combined for term in CSV_REJECT_TERMS):
        return None

    image_url = normalize_url(
        first_non_empty(
            row,
            [
                "image_url", "product_image", "candidate_image_url",
                "page_image_url", "image", "img_url",
            ],
        )
    ) or None

    price_text = clean_text(
        first_non_empty(row, ["price_text", "price", "sale_price", "current_price", "amount"]),
        max_len=80,
    ) or extract_price(f"{title} {description}")

    source_site = clean_text(
        first_non_empty(row, ["source_site", "site", "site_name", "source", "retailer", "merchant"]),
        max_len=80,
    ) or host(product_url)

    category = clean_text(
        first_non_empty(row, ["category", "product_category", "required_categories", "taxonomy", "type"]),
        max_len=120,
    ) or None

    if not title:
        title = product_url

    return {
        "title": title,
        "description": description,
        "price_text": price_text or None,
        "product_url": product_url,
        "image_url": image_url,
        "source_site": source_site,
        "category": category,
        "relevance_score": 0,
        "why_it_matches": (
            "Retrieved from CSV candidate cache for high-recall retrieval. "
            "Semantic fit is handled by downstream reranking."
        ),
        "retrieval_rank": retrieval_rank,
        "retrieval_source": "csv_candidate_cache",
        "search_phrase": query,
        "site_key": "csv_cache",
    }


def load_csv_products(
    query: str,
    csv_data_path: Optional[str] = None,
    max_csv_candidates: int = 150,
) -> List[Dict[str, Any]]:
    csv_path = resolve_csv_path(csv_data_path)
    if not csv_path:
        print("[CSV] No CSV candidate cache found.")
        return []

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with csv_path.open("r", encoding="latin-1", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"[WARN] Could not read CSV candidate cache {csv_path}: {exc}")
        return []

    if not rows:
        return []

    matched_rows = [row for row in rows if row_matches_query_broadly(row, query)]
    selected_rows = matched_rows if matched_rows else rows

    products: List[Dict[str, Any]] = []
    for row in selected_rows:
        product = csv_row_to_product(row, retrieval_rank=len(products) + 1, query=query)
        if product:
            products.append(product)
        if len(products) >= max_csv_candidates:
            break

    print(
        f"[CSV] Loaded {len(products)} candidates from {csv_path}. "
        f"matched_rows={len(matched_rows)} total_rows={len(rows)}"
    )
    return products


# ============================================================
# 4. Browser search-page extraction
# ============================================================

PRODUCT_LINK_EXTRACTION_JS = """
() => {
  const abs = (url) => {
    try {
      if (!url) return null;
      return new URL(url, location.href).href;
    } catch {
      return null;
    }
  };

  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();

  const getImage = (container, anchor) => {
    const img =
      anchor.querySelector("img") ||
      container.querySelector("img") ||
      container.parentElement?.querySelector("img");

    if (!img) return null;

    const srcset = img.getAttribute("srcset") || "";
    const dataSrcset = img.getAttribute("data-srcset") || "";

    return abs(
      img.currentSrc ||
      img.src ||
      img.getAttribute("data-src") ||
      img.getAttribute("data-original") ||
      img.getAttribute("data-lazy-src") ||
      img.getAttribute("data-testid-src") ||
      srcset.split(" ")[0] ||
      dataSrcset.split(" ")[0]
    );
  };

  const anchors = Array.from(document.querySelectorAll("a[href]"));
  const results = [];

  for (const a of anchors) {
    const href = abs(
      a.getAttribute("href") ||
      a.getAttribute("data-href") ||
      a.href
    );

    if (!href || href.startsWith("javascript:") || href.startsWith("mailto:")) {
      continue;
    }

    const container =
      a.closest("article") ||
      a.closest("li") ||
      a.closest("[data-testid]") ||
      a.closest("[data-test]") ||
      a.closest("[data-cy]") ||
      a.closest("[class*='product']") ||
      a.closest("[class*='Product']") ||
      a.closest("[class*='card']") ||
      a.closest("[class*='Card']") ||
      a.closest("div") ||
      a;

    const imageUrl = getImage(container, a);

    const anchorText = clean(
      a.innerText ||
      a.textContent ||
      a.getAttribute("aria-label") ||
      a.getAttribute("title") ||
      ""
    );

    const rawText = clean(
      container.innerText ||
      container.textContent ||
      anchorText ||
      ""
    ).slice(0, 1800);

    results.push({
      href,
      anchorText: anchorText.slice(0, 400),
      imageUrl,
      rawText
    });
  }

  return results;
}
"""


class BrowserbaseLinkRetriever:
    REJECT_URL_TERMS = [
        "login", "signin", "signup", "register", "account", "cart", "basket",
        "wishlist", "help", "support", "customer-service", "return", "policy",
        "privacy", "terms", "blog", "ideas", "inspiration", "stores", "locator",
        "contact", "about-us", "careers", "track-order", "order-history",
        "gift-card", "registry", "financing", "assembly", "delivery",
    ]

    GENERIC_PRODUCT_URL_HINTS = [
        "/p/", "/product", "/products", "/dp/", "/gp/product/", "/ip/",
        "/item", "/listing", "/pd/", "sku=", "pid=", "productid",
    ]

    def __init__(
        self,
        browserbase_api_key: str,
        sites: Optional[List[str]] = None,
        max_links_per_site: int = 25,
        scroll_rounds: int = 5,
        use_proxy: bool = False,
    ):
        self.bb = Browserbase(api_key=browserbase_api_key)
        self.sites = sites or DEFAULT_US_SITES
        self.max_links_per_site = max_links_per_site
        self.scroll_rounds = scroll_rounds
        self.use_proxy = use_proxy

    def build_search_urls(self, query: str) -> List[Dict[str, Any]]:
        urls = []
        encoded = quote_plus(query)

        for site_key in self.sites:
            site = SITE_SEARCH_TEMPLATES[site_key]
            urls.append(
                {
                    "site_key": site_key,
                    "site_name": site["name"],
                    "search_url": site["search_url"].format(query=encoded),
                    "allowed_domains": site["allowed_domains"],
                    "product_url_patterns": site.get("product_url_patterns", []),
                    "query": query,
                }
            )

        return urls

    def looks_like_product(self, item: Dict[str, Any], allowed_domains: List[str], patterns: List[str]) -> bool:
        url = normalize_url(item.get("href"))
        if not url:
            return False

        url_l = url.lower()
        text = f"{item.get('anchorText') or ''} {item.get('rawText') or ''}".lower()

        if not domain_allowed(url, allowed_domains):
            return False

        if any(term in url_l for term in self.REJECT_URL_TERMS):
            return False

        if any(term in url_l for term in ["/search", "search?", "/category", "/categories", "/collections", "/collection", "/pages/"]):
            return False

        if any(pattern.lower() in url_l for pattern in patterns):
            return True

        if any(hint in url_l for hint in self.GENERIC_PRODUCT_URL_HINTS):
            return True

        # High-recall fallback.
        if item.get("imageUrl") and len(text.strip()) >= 12:
            return True

        return False

    async def collect_one_search_url(self, search: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        One Browserbase session per search URL.

        This is the main fix: if one site closes the context, other sites still work.
        """
        session_kwargs: Dict[str, Any] = {
            "timeout": 900,
            "browser_settings": {
                "blockAds": True,
                "recordSession": True,
                "logSession": True,
                "viewport": {"width": 1440, "height": 1100},
            },
        }

        if self.use_proxy:
            session_kwargs["proxies"] = True

        session = self.bb.sessions.create(**session_kwargs)
        print(f"[Browserbase] Session for {search['site_name']}: https://browserbase.com/sessions/{session.id}")

        playwright = await async_playwright().start()
        browser = None

        try:
            browser = await playwright.chromium.connect_over_cdp(session.connect_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()

            print(f"[Search] {search['site_name']}: {search['query']}")
            try:
                await page.goto(search["search_url"], wait_until="domcontentloaded", timeout=45_000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass
            except Exception as exc:
                print(f"[WARN] Search page failed for {search['site_name']}: {exc}")
                return []

            # Best-effort close cookie/popups.
            for selector in [
                "button:has-text('Accept')",
                "button:has-text('Accept All')",
                "button:has-text('I Accept')",
                "button:has-text('Got it')",
                "button:has-text('No thanks')",
                "button:has-text('Close')",
                "button[aria-label='Close']",
                "[aria-label='close']",
            ]:
                try:
                    loc = page.locator(selector).first
                    if await loc.count() > 0:
                        await loc.click(timeout=1200)
                        await page.wait_for_timeout(300)
                except Exception:
                    pass

            for _ in range(self.scroll_rounds):
                if page.is_closed():
                    return []
                await page.mouse.wheel(0, 2400)
                await page.wait_for_timeout(700)

            if page.is_closed():
                return []

            try:
                raw_links = await page.evaluate(PRODUCT_LINK_EXTRACTION_JS)
            except Exception as exc:
                print(f"[WARN] Extraction failed for {search['site_name']}: {exc}")
                return []

            out: List[Dict[str, Any]] = []
            seen: set[str] = set()

            for item in raw_links or []:
                url = normalize_url(item.get("href"))
                if not url or url in seen:
                    continue

                item["href"] = url

                if not self.looks_like_product(
                    item,
                    allowed_domains=search["allowed_domains"],
                    patterns=search["product_url_patterns"],
                ):
                    continue

                seen.add(url)

                raw_text = clean_text(item.get("rawText"), max_len=1200)
                title = clean_text(item.get("anchorText") or raw_text, max_len=180)

                out.append(
                    {
                        "title": title or url,
                        "description": raw_text,
                        "price_text": extract_price(raw_text),
                        "product_url": url,
                        "image_url": normalize_url(item.get("imageUrl")) or None,
                        "source_site": search["site_name"],
                        "category": None,
                        "relevance_score": 0,
                        "why_it_matches": (
                            "Retrieved from ecommerce search page for high-recall retrieval. "
                            "Semantic fit is handled by downstream reranking."
                        ),
                        "retrieval_rank": len(out) + 1,
                        "retrieval_source": "browserbase_search_page_high_recall",
                        "search_phrase": search["query"],
                        "site_key": search["site_key"],
                    }
                )

                if len(out) >= self.max_links_per_site:
                    break

            return out

        finally:
            try:
                if browser:
                    await browser.close()
            except Exception:
                pass
            try:
                await playwright.stop()
            except Exception:
                pass

    async def collect(self, query: str) -> List[Dict[str, Any]]:
        search_urls = self.build_search_urls(query)
        all_products: List[Dict[str, Any]] = []

        # Sequential by default to reduce anti-bot/context instability.
        for search in search_urls:
            try:
                products = await self.collect_one_search_url(search)
                all_products.extend(products)
            except Exception as exc:
                print(f"[WARN] Site retrieval crashed for {search['site_name']}: {exc}")
                continue

            # Mild pacing between Browserbase sessions.
            await asyncio.sleep(0.7)

        return all_products


# ============================================================
# 5. Merge + ADK tool
# ============================================================

def merge_products(
    live_products: List[Dict[str, Any]],
    csv_products: List[Dict[str, Any]],
    max_results: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for source in [live_products, csv_products]:
        for item in source:
            url = normalize_url(item.get("product_url"))
            if not url or url in seen:
                continue
            item["product_url"] = url
            item["retrieval_rank"] = len(merged) + 1
            seen.add(url)
            merged.append(item)

            if len(merged) >= max_results:
                return merged

    return merged


async def browse_products_with_browserbase(
    query: str,
    country: str = "US",
    max_results: int = 50,
    sites: Optional[List[str]] = None,
    use_proxy: bool = False,
    use_csv_cache: bool = True,
    csv_data_path: Optional[str] = None,
    max_csv_candidates: int = 150,
    # Compatibility args accepted but intentionally ignored in V2.
    max_product_pages: int = 0,
    visit_product_pages: bool = False,
) -> Dict[str, Any]:
    """
    ADK tool entrypoint.

    Retrieval-only:
      - Browserbase search pages only
      - one Browserbase session per site
      - CSV fallback
      - no product-page enrichment
      - no ranking/scoring
    """
    load_dotenv()

    effective_query = extract_query_from_input(query)
    country = (country or "US").upper()

    if sites is None:
        sites = DEFAULT_US_SITES

    live_products: List[Dict[str, Any]] = []
    csv_products: List[Dict[str, Any]] = []
    notes: List[str] = []

    browserbase_api_key = os.environ.get("BROWSERBASE_API_KEY")

    if browserbase_api_key:
        try:
            retriever = BrowserbaseLinkRetriever(
                browserbase_api_key=browserbase_api_key,
                sites=sites,
                max_links_per_site=max(10, min(30, max_results)),
                scroll_rounds=5,
                use_proxy=use_proxy,
            )
            live_products = await retriever.collect(effective_query)
        except Exception as exc:
            notes.append(f"Browserbase retrieval failed: {type(exc).__name__}: {exc}")
            live_products = []
    else:
        notes.append("BROWSERBASE_API_KEY missing; using CSV-only retrieval.")

    if use_csv_cache:
        csv_products = load_csv_products(
            query=effective_query,
            csv_data_path=csv_data_path,
            max_csv_candidates=max_csv_candidates,
        )

    merged = merge_products(
        live_products=live_products,
        csv_products=csv_products,
        max_results=max_results,
    )

    notes.extend(
        [
            "High-recall retrieval completed.",
            "Browserbase V2 uses search pages only and one Browserbase session per site.",
            "No candidate scoring, ranking, or Gemini post-processing was performed.",
            "Product-page enrichment is intentionally disabled to avoid browser context crashes.",
            "Downstream verifier and cross-encoder should handle quality and relevance.",
            f"effective_query={effective_query}",
            f"live_candidates={len(live_products)}",
            f"csv_candidates={len(csv_products)}",
            f"merged_candidates={len(merged)}",
            f"sites={sites}",
        ]
    )

    return {
        "products": merged,
        "notes": notes,
    }


# ============================================================
# 6. Local smoke test
# ============================================================

async def _smoke_test() -> None:
    result = await browse_products_with_browserbase(
        query="pet friendly durable easy clean washable home decor furniture pet supplies accessories pet bed sofa cover rug storage scratch resistant under $500",
        country="US",
        max_results=50,
        sites=[
            "rugs_usa",
            "nathan_james",
            "west_elm",
            "amazon",
            "castlery",
            "world_market",
        ],
        use_proxy=False,
        use_csv_cache=True,
        csv_data_path="data/browserbase_results_final_with_ho_05.csv",
        max_csv_candidates=150,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_smoke_test())
