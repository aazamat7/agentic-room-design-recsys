# shopping_agent/tools/browserbase_product_browser_V4.py
# ============================================================
# Browserbase Product Retrieval Tool V4 FINAL
#
# Features:
#   - Strict product URL validation before and after enrichment.
#   - Isolated Browserbase session per product page.
#   - Product-page enrichment for canonical product_url, title, price, image, desc.
#   - CSV fallback/augmentation.
#   - Appends newly discovered valid live products to CSV cache.
#   - No Gemini call, no ranking, no semantic filtering.
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
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

from shopping_agent.tools.product_csv_cache import append_new_products_to_csv_cache


# ============================================================
# 1. Site configuration
# ============================================================

SITE_SEARCH_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # "rugs_usa": {
    #     "name": "Rugs USA",
    #     "search_url": "https://www.rugsusa.com/search?query={query}",
    #     "allowed_domains": ["rugsusa.com"],
    #     "product_url_patterns": ["/products/"],
    # },

    # New: IKEA US — safe to include in default.
    "ikea_us": {
        "name": "IKEA US",
        "search_url": "https://www.ikea.com/us/en/search/?q={query}",
        "allowed_domains": ["ikea.com"],
        "product_url_patterns": ["/us/en/p/"],
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
        "product_url_patterns": ["/p/"],
    },
    # Available but intentionally not default because Article caused context closures earlier.
    "article": {
        "name": "Article",
        "search_url": "https://www.article.com/search?query={query}",
        "allowed_domains": ["article.com"],
        "product_url_patterns": ["/product/"],
    },
}

DEFAULT_US_SITES = [
    "ikea_us",
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

PROMO_PRICE_LABELS = {
    "clearance",
    "limited time offer",
    "sale",
    "new",
    "best seller",
    "bestseller",
    "exclusive",
}


# ============================================================
# 2. Generic helpers
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

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
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


def host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def extract_amazon_asin(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path
    except Exception:
        return None

    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product/([A-Z0-9]{10})",
    ]

    for pattern in patterns:
        match = re.search(pattern, path, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        return ""

    parsed = urlparse(url)

    # Amazon sponsored redirect handling.
    if "amazon.com" in parsed.netloc.lower():
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        redirect_url = params.get("url")
        if redirect_url and redirect_url.startswith("/"):
            return normalize_url("https://www.amazon.com" + unquote(redirect_url))

    # Amazon canonicalization to ASIN.
    asin = extract_amazon_asin(url)
    if asin:
        return f"https://www.amazon.com/dp/{asin}"

    clean_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_l = key.lower()

        if key_l.startswith("utm_") or key_l in {
            "gclid",
            "fbclid",
            "msclkid",
            "yclid",
            "igshid",
            "ref",
            "ref_",
            "dib",
            "dib_tag",
            "keywords",
            "qid",
            "sr",
            "nsdoptoutparam",
        }:
            continue

        clean_params.append((key, value))

    return urlunparse(
        parsed._replace(
            fragment="",
            query=urlencode(clean_params, doseq=True),
        )
    )


def product_key(url: str) -> str:
    url = normalize_url(url)
    h = host(url)

    asin = extract_amazon_asin(url)
    if asin:
        return f"amazon:{asin}"

    parsed = urlparse(url)
    path = parsed.path.rstrip("/").lower()

    return f"{h}:{path}"


def domain_allowed(url: str, allowed_domains: List[str]) -> bool:
    h = host(url)

    return bool(h) and any(
        domain.lower().replace("www.", "") in h
        for domain in allowed_domains
    )


def is_numeric_price(value: Optional[str]) -> bool:
    if not value:
        return False

    text = str(value).strip().lower()

    if text in PROMO_PRICE_LABELS:
        return False

    return bool(
        re.search(
            r"(?:\$|usd|₹|rs\.?|inr|£|gbp|€|eur)\s?[\d,]+(?:\.\d{1,2})?",
            text,
            flags=re.IGNORECASE,
        )
    )


def normalize_price_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    text = clean_text(value, max_len=120)

    if not is_numeric_price(text):
        return None

    match = re.search(
        r"((?:\$|USD|₹|Rs\.?|INR|£|GBP|€|EUR)\s?[\d,]+(?:\.\d{1,2})?)",
        text,
        flags=re.IGNORECASE,
    )

    return clean_text(match.group(1), max_len=80) if match else None


def extract_price_regex(text: str) -> Optional[str]:
    if not text:
        return None

    price_patterns = [
        r"(?:sale price|price|now)\s*[:\-]?\s*((?:\$|USD)\s?[\d,]+(?:\.\d{1,2})?)",
        r"((?:\$|USD)\s?[\d,]+(?:\.\d{1,2})?)",
        r"((?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?)",
        r"((?:£|GBP|€|EUR)\s?[\d,]+(?:\.\d{1,2})?)",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_price_text(match.group(1))

    return None


def valid_image_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        return None

    url_l = url.lower()

    bad_terms = [
        "[object",
        "%5bobject",
        "key-rewards.svg",
        "icon",
        "logo",
        "sprite",
        "tracking",
        "uedata",
        "fls-na.amazon.com",
        "transparent",
        "placeholder",
        "1x1",
    ]

    if any(term in url_l for term in bad_terms):
        return None

    if url_l.endswith(".svg"):
        return None

    return normalize_url(url)


def is_obvious_non_product_url(url: str) -> bool:
    if not url:
        return True

    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")

    if path in {"", "/"}:
        return True

    non_product_terms = [
        "/search",
        "/category",
        "/categories",
        "/collections",
        "/collection",
        "/pages",
        "/shop-by-style",
        "/shop/",
        "/room-inspiration",
        "/ideas",
        "/blog",
        "/collaborations",
        "/sustainability",
        "/registry",
        "/customer-service",
        "/design-services",
        "/my-boards",
        "/stores",
        "/account",
        "/cart",
        "/checkout",
    ]

    return any(term in path for term in non_product_terms)


def is_valid_product_url_for_site(
    url: str,
    site_key: Optional[str],
    product_url_patterns: Optional[List[str]] = None,
) -> bool:
    url = normalize_url(url)
    path = urlparse(url).path.lower()

    if not url or is_obvious_non_product_url(url):
        return False

    if site_key == "amazon":
        return extract_amazon_asin(url) is not None

    if product_url_patterns:
        return any(pattern.lower() in path for pattern in product_url_patterns)

    return False


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
# 3. CSV augmentation read path
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


def csv_row_to_product(
    row: Dict[str, Any],
    retrieval_rank: int,
    query: str,
) -> Optional[Dict[str, Any]]:
    product_url = normalize_url(
        first_non_empty(
            row,
            [
                "product_url",
                "url",
                "link",
                "href",
                "product_link",
                "productUrl",
                "product url",
            ],
        )
    )

    if not product_url:
        return None

    title = clean_text(
        first_non_empty(
            row,
            [
                "title",
                "product_title",
                "product_name",
                "name",
                "candidate_title",
                "page_title",
                "anchorText",
                "anchor_text",
            ],
        ),
        max_len=180,
    )

    description = clean_text(
        first_non_empty(
            row,
            [
                "description",
                "product_description",
                "candidate_text",
                "rawText",
                "raw_text",
                "page_description",
                "page_text_excerpt",
                "text",
            ],
        ),
        max_len=450,
    )

    combined = f"{title} {description}".lower()

    if any(term in combined for term in CSV_REJECT_TERMS):
        return None

    image_url = valid_image_url(
        first_non_empty(
            row,
            [
                "image_url",
                "product_image",
                "candidate_image_url",
                "page_image_url",
                "image",
                "img_url",
            ],
        )
    )

    price_text = normalize_price_text(
        first_non_empty(
            row,
            [
                "price_text",
                "price",
                "sale_price",
                "current_price",
                "amount",
            ],
        )
    ) or extract_price_regex(f"{title} {description}")

    source_site = clean_text(
        first_non_empty(
            row,
            [
                "source_site",
                "site",
                "site_name",
                "source",
                "retailer",
                "merchant",
            ],
        ),
        max_len=80,
    ) or host(product_url)

    category = clean_text(
        first_non_empty(
            row,
            [
                "category",
                "product_category",
                "required_categories",
                "taxonomy",
                "type",
            ],
        ),
        max_len=120,
    ) or None

    if not title:
        title = product_url

    return {
        "title": title,
        "description": description,
        "price_text": price_text,
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
        "is_enriched_product_page": bool(price_text or image_url),
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
        product = csv_row_to_product(
            row=row,
            retrieval_rank=len(products) + 1,
            query=query,
        )

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
# 4. Browser JS
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
      a.closest("[data-component-type='s-search-result']") ||
      a.closest("[data-asin]") ||
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

    const dataAsin = container ? container.getAttribute("data-asin") : null;

    results.push({
      href,
      anchorText: anchorText.slice(0, 400),
      imageUrl,
      rawText,
      dataAsin
    });
  }

  return results;
}
"""


PRODUCT_PAGE_EXTRACTION_JS = """
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

  const meta = (selector) => {
    const el = document.querySelector(selector);
    return el ? el.getAttribute("content") : null;
  };

  const firstString = (value) => {
    if (!value) return null;

    if (typeof value === "string") return value;

    if (Array.isArray(value)) {
      for (const x of value) {
        const found = firstString(x);
        if (found) return found;
      }
      return null;
    }

    if (typeof value === "object") {
      for (const key of ["url", "contentUrl", "thumbnailUrl", "@id"]) {
        if (typeof value[key] === "string") return value[key];
      }
    }

    return null;
  };

  const textFromSelectors = (selectors) => {
    for (const selector of selectors) {
      const el = document.querySelector(selector);
      if (el) {
        const txt = clean(el.innerText || el.textContent || el.getAttribute("content") || "");
        if (txt) return txt;
      }
    }
    return "";
  };

  const parseJsonLdProducts = () => {
    const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
    const products = [];

    const walk = (obj) => {
      if (!obj) return;

      if (Array.isArray(obj)) {
        for (const x of obj) walk(x);
        return;
      }

      if (typeof obj !== "object") return;

      const type = obj["@type"];
      const types = Array.isArray(type) ? type : [type];

      if (types && types.some(t => String(t).toLowerCase() === "product")) {
        products.push(obj);
      }

      if (obj["@graph"]) walk(obj["@graph"]);

      for (const value of Object.values(obj)) {
        if (value && typeof value === "object") walk(value);
      }
    };

    for (const script of scripts) {
      try {
        const obj = JSON.parse(script.textContent || "");
        walk(obj);
      } catch {}
    }

    return products;
  };

  const jsonLdProducts = parseJsonLdProducts();
  const jsonProduct = jsonLdProducts.length ? jsonLdProducts[0] : {};

  const jsonOffers = Array.isArray(jsonProduct.offers)
    ? jsonProduct.offers[0]
    : (jsonProduct.offers || {});

  const jsonImage = firstString(jsonProduct.image);

  const jsonPrice =
    jsonOffers.price ||
    jsonOffers.lowPrice ||
    jsonOffers.highPrice ||
    null;

  const jsonCurrency = jsonOffers.priceCurrency || "";

  const title =
    clean(jsonProduct.name || "") ||
    meta('meta[property="og:title"]') ||
    meta('meta[name="twitter:title"]') ||
    textFromSelectors([
      "#productTitle",
      "[data-testid*='product-title']",
      "[class*='ProductTitle']",
      "[class*='product-title']",
      "h1"
    ]) ||
    document.title ||
    "";

  const description =
    clean(jsonProduct.description || "") ||
    meta('meta[name="description"]') ||
    meta('meta[property="og:description"]') ||
    meta('meta[name="twitter:description"]') ||
    textFromSelectors([
      "#feature-bullets",
      "#productDescription",
      "[data-testid*='description']",
      "[class*='description']",
      "[class*='Description']"
    ]);

  const imageUrl =
    abs(jsonImage) ||
    abs(meta('meta[property="og:image"]')) ||
    abs(meta('meta[property="og:image:secure_url"]')) ||
    abs(meta('meta[name="twitter:image"]')) ||
    abs(document.querySelector("img")?.currentSrc || document.querySelector("img")?.src);

  const canonicalUrl =
    abs(document.querySelector('link[rel="canonical"]')?.href) ||
    abs(firstString(jsonProduct.url)) ||
    location.href;

  let priceText = "";

  if (jsonPrice) {
    priceText = `${jsonCurrency ? jsonCurrency + " " : ""}${jsonPrice}`;
  }

  if (!priceText) {
    priceText = textFromSelectors([
      ".a-price .a-offscreen",
      "#priceblock_ourprice",
      "#priceblock_dealprice",
      "[data-testid*='price']",
      "[class*='sale-price']",
      "[class*='SalePrice']",
      "[class*='price']",
      "[class*='Price']",
      "[itemprop='price']"
    ]);
  }

  const bodyText = clean(document.body ? document.body.innerText : "").slice(0, 9000);

  return {
    finalUrl: location.href,
    canonicalUrl,
    title,
    description,
    imageUrl,
    priceText,
    bodyText,
    jsonLdProductFound: jsonLdProducts.length > 0
  };
}
"""


# ============================================================
# 5. Browserbase retriever
# ============================================================

class BrowserbaseProductRetriever:
    REJECT_URL_TERMS = [
        "login",
        "signin",
        "signup",
        "register",
        "account",
        "cart",
        "basket",
        "wishlist",
        "help",
        "support",
        "customer-service",
        "return",
        "policy",
        "privacy",
        "terms",
        "blog",
        "ideas",
        "inspiration",
        "stores",
        "locator",
        "contact",
        "about-us",
        "careers",
        "track-order",
        "order-history",
        "gift-card",
        "registry",
        "financing",
        "assembly",
        "delivery",
    ]

    def __init__(
        self,
        browserbase_api_key: str,
        sites: Optional[List[str]] = None,
        max_links_per_site: int = 12,
        max_product_pages: int = 24,
        scroll_rounds: int = 4,
        use_proxy: bool = False,
        enrich_product_pages: bool = True,
        product_page_concurrency: int = 2,
    ):
        self.bb = Browserbase(api_key=browserbase_api_key)
        self.sites = sites or DEFAULT_US_SITES
        self.max_links_per_site = max_links_per_site
        self.max_product_pages = max_product_pages
        self.scroll_rounds = scroll_rounds
        self.use_proxy = use_proxy
        self.enrich_product_pages = enrich_product_pages
        self.product_page_concurrency = product_page_concurrency

    def build_search_urls(self, query: str) -> List[Dict[str, Any]]:
        encoded = quote_plus(query)
        urls = []

        for site_key in self.sites:
            if site_key not in SITE_SEARCH_TEMPLATES:
                print(f"[WARN] Unknown site_key={site_key}; skipping.")
                continue

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

    def session_kwargs(self, timeout: int = 900) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "browser_settings": {
                "blockAds": True,
                "recordSession": True,
                "logSession": True,
                "viewport": {"width": 1440, "height": 1100},
            },
        }

        if self.use_proxy:
            kwargs["proxies"] = True

        return kwargs

    def looks_like_product(
        self,
        item: Dict[str, Any],
        search: Dict[str, Any],
    ) -> bool:
        url = normalize_url(item.get("href"))

        if not url:
            return False

        url_l = url.lower()

        if not domain_allowed(url, search["allowed_domains"]):
            return False

        if any(term in url_l for term in self.REJECT_URL_TERMS):
            return False

        if not is_valid_product_url_for_site(
            url=url,
            site_key=search["site_key"],
            product_url_patterns=search.get("product_url_patterns", []),
        ):
            return False

        return True

    async def collect_search_candidates_one_site(
        self,
        search: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        session = self.bb.sessions.create(**self.session_kwargs(timeout=900))

        print(
            f"[Browserbase] Search session for {search['site_name']}: "
            f"https://browserbase.com/sessions/{session.id}"
        )

        playwright = await async_playwright().start()
        browser = None

        try:
            browser = await playwright.chromium.connect_over_cdp(session.connect_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()

            print(f"[Search] {search['site_name']}: {search['query']}")

            try:
                await page.goto(
                    search["search_url"],
                    wait_until="domcontentloaded",
                    timeout=45_000,
                )

                try:
                    await page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass

            except Exception as exc:
                print(f"[WARN] Search page failed for {search['site_name']}: {exc}")
                return []

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
                print(f"[WARN] Link extraction failed for {search['site_name']}: {exc}")
                return []

            out: List[Dict[str, Any]] = []
            seen_keys: set[str] = set()

            for item in raw_links or []:
                url = normalize_url(item.get("href"))

                if not url:
                    continue

                item["href"] = url

                if not self.looks_like_product(item, search):
                    continue

                key = product_key(url)

                if key in seen_keys:
                    continue

                seen_keys.add(key)

                raw_text = clean_text(item.get("rawText"), max_len=1200)
                title = clean_text(item.get("anchorText") or raw_text, max_len=180)

                if search["site_key"] == "amazon":
                    # Avoid swatch/title noise like "Gold", "Blue", "+6".
                    if len(title) < 12 and raw_text:
                        title = clean_text(raw_text, max_len=180)

                out.append(
                    {
                        "title": title or url,
                        "description": raw_text,
                        "price_text": extract_price_regex(raw_text),
                        "product_url": url,
                        "image_url": valid_image_url(item.get("imageUrl")),
                        "source_site": search["site_name"],
                        "category": None,
                        "relevance_score": 0,
                        "why_it_matches": (
                            "Retrieved from ecommerce search page. "
                            "Product-page enrichment may update details."
                        ),
                        "retrieval_rank": len(out) + 1,
                        "retrieval_source": "browserbase_search_page_candidate",
                        "search_phrase": search["query"],
                        "site_key": search["site_key"],
                        "is_enriched_product_page": False,
                    }
                )

                if len(out) >= self.max_links_per_site:
                    break

            print(f"[Search] {search['site_name']} candidates={len(out)}")
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

    async def enrich_one_product_page(
        self,
        candidate: Dict[str, Any],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        product_url = normalize_url(candidate.get("product_url"))

        if not product_url:
            return None

        site_key = candidate.get("site_key")
        site_conf = SITE_SEARCH_TEMPLATES.get(site_key, {})

        session = self.bb.sessions.create(**self.session_kwargs(timeout=600))

        print(
            f"[Browserbase] Product session {idx}: "
            f"https://browserbase.com/sessions/{session.id}"
        )
        print(f"[Enrich] {idx}: {product_url}")

        playwright = await async_playwright().start()
        browser = None

        try:
            browser = await playwright.chromium.connect_over_cdp(session.connect_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                await page.goto(
                    product_url,
                    wait_until="domcontentloaded",
                    timeout=40_000,
                )

                try:
                    await page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass

            except Exception as exc:
                print(f"[WARN] Product page failed {product_url}: {exc}")
                fallback = dict(candidate)
                fallback["enrichment_error"] = f"{type(exc).__name__}: {exc}"
                return fallback

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

            try:
                detail = await page.evaluate(PRODUCT_PAGE_EXTRACTION_JS)
            except Exception as exc:
                print(f"[WARN] Product detail extraction failed {product_url}: {exc}")
                fallback = dict(candidate)
                fallback["enrichment_error"] = f"{type(exc).__name__}: {exc}"
                return fallback

            final_url = normalize_url(
                detail.get("canonicalUrl")
                or detail.get("finalUrl")
                or candidate.get("product_url")
            )

            # Critical: reject enrichment if page canonicalized to homepage/category/style.
            if not is_valid_product_url_for_site(
                url=final_url,
                site_key=site_key,
                product_url_patterns=site_conf.get("product_url_patterns", []),
            ):
                print(f"[DROP] Enriched URL is not a valid product page: {final_url}")
                return None

            title = clean_text(detail.get("title") or candidate.get("title"), max_len=180)

            description = clean_text(
                detail.get("description")
                or candidate.get("description")
                or detail.get("bodyText"),
                max_len=500,
            )

            price_text = normalize_price_text(detail.get("priceText")) or extract_price_regex(
                " ".join(
                    [
                        detail.get("priceText") or "",
                        detail.get("bodyText") or "",
                        candidate.get("description") or "",
                    ]
                )
            )

            image_url = valid_image_url(detail.get("imageUrl")) or valid_image_url(candidate.get("image_url"))

            enriched = dict(candidate)
            enriched.update(
                {
                    "title": title or candidate.get("title") or final_url,
                    "description": description or candidate.get("description") or "",
                    "price_text": price_text,
                    "product_url": final_url,
                    "image_url": image_url,
                    "retrieval_source": "browserbase_product_page_enriched",
                    "is_enriched_product_page": True,
                    "json_ld_product_found": bool(detail.get("jsonLdProductFound")),
                    "enrichment_error": None,
                }
            )

            return enriched

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

    async def collect_search_candidates(self, query: str) -> List[Dict[str, Any]]:
        search_urls = self.build_search_urls(query)
        all_candidates: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for search in search_urls:
            try:
                candidates = await self.collect_search_candidates_one_site(search)
            except Exception as exc:
                print(f"[WARN] Site retrieval crashed for {search['site_name']}: {exc}")
                candidates = []

            for candidate in candidates:
                url = normalize_url(candidate.get("product_url"))

                if not url:
                    continue

                key = product_key(url)

                if key in seen_keys:
                    continue

                seen_keys.add(key)
                all_candidates.append(candidate)

            await asyncio.sleep(0.6)

        for idx, candidate in enumerate(all_candidates, start=1):
            candidate["retrieval_rank"] = idx

        print(f"[Search] total_unique_candidates={len(all_candidates)}")
        return all_candidates

    async def enrich_product_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enrich_product_pages or self.max_product_pages <= 0:
            return candidates

        to_enrich = candidates[: self.max_product_pages]
        not_enriched = candidates[self.max_product_pages :]

        semaphore = asyncio.Semaphore(self.product_page_concurrency)

        async def worker(idx: int, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.enrich_one_product_page(candidate, idx=idx)

        enriched = await asyncio.gather(
            *[
                worker(idx, candidate)
                for idx, candidate in enumerate(to_enrich, start=1)
            ],
            return_exceptions=True,
        )

        products: List[Dict[str, Any]] = []

        for candidate, result in zip(to_enrich, enriched):
            if isinstance(result, Exception):
                fallback = dict(candidate)
                fallback["enrichment_error"] = f"{type(result).__name__}: {result}"
                products.append(fallback)
            elif result is None:
                # Drop non-product pages after enrichment.
                continue
            else:
                products.append(result)

        products.extend(not_enriched)
        return products

    async def collect(self, query: str) -> List[Dict[str, Any]]:
        candidates = await self.collect_search_candidates(query)
        products = await self.enrich_product_candidates(candidates)

        for idx, product in enumerate(products, start=1):
            product["retrieval_rank"] = idx

        print(
            "[Browserbase] live_products="
            f"{len(products)} enriched="
            f"{sum(1 for p in products if p.get('is_enriched_product_page'))}"
        )

        return products


# ============================================================
# 6. Merge + ADK tool
# ============================================================

def merge_products(
    live_products: List[Dict[str, Any]],
    csv_products: List[Dict[str, Any]],
    max_results: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    # Prioritize valid enriched products with price/image.
    def product_quality_key(p: Dict[str, Any]) -> tuple:
        return (
            0 if p.get("is_enriched_product_page") else 1,
            0 if p.get("price_text") else 1,
            0 if p.get("image_url") else 1,
            p.get("retrieval_rank") or 9999,
        )

    live_sorted = sorted(live_products, key=product_quality_key)

    for source in [live_sorted, csv_products]:
        for item in source:
            url = normalize_url(item.get("product_url"))

            if not url:
                continue

            # Do not let bad non-product pages through.
            if item.get("site_key") in SITE_SEARCH_TEMPLATES:
                site_conf = SITE_SEARCH_TEMPLATES[item.get("site_key")]
                if not is_valid_product_url_for_site(
                    url=url,
                    site_key=item.get("site_key"),
                    product_url_patterns=site_conf.get("product_url_patterns", []),
                ):
                    continue

            item["product_url"] = url
            item["image_url"] = valid_image_url(item.get("image_url"))
            item["price_text"] = normalize_price_text(item.get("price_text"))
            item["retrieval_rank"] = len(merged) + 1

            key = product_key(url)

            if key in seen_keys:
                continue

            seen_keys.add(key)
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
    max_product_pages: int = 24,
    visit_product_pages: bool = True,
    product_page_concurrency: int = 2,
    update_csv_cache: bool = True,
    max_csv_append: int = 100,
) -> Dict[str, Any]:
    """
    ADK tool entrypoint.

    Final V4 behavior:
      - search-page product link collection
      - isolated product-page enrichment
      - strict product URL validation
      - CSV read fallback
      - CSV append for newly discovered live products
      - no ranking/scoring/LLM filtering
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
            retriever = BrowserbaseProductRetriever(
                browserbase_api_key=browserbase_api_key,
                sites=sites,
                max_links_per_site=max(6, min(12, max_results)),
                max_product_pages=max_product_pages,
                scroll_rounds=4,
                use_proxy=use_proxy,
                enrich_product_pages=visit_product_pages,
                product_page_concurrency=product_page_concurrency,
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

    csv_append_result = {
        "csv_path": csv_data_path or os.environ.get("PRODUCT_CANDIDATE_CSV_PATH") or DEFAULT_CSV_PATH,
        "num_seen_existing": 0,
        "num_candidates_considered": 0,
        "num_appended": 0,
        "appended_urls": [],
    }

    # Read CSV first, then append live products for future runs.
    if update_csv_cache and live_products:
        csv_append_result = append_new_products_to_csv_cache(
            products=live_products,
            csv_data_path=csv_data_path,
            max_append=max_csv_append,
        )

    merged = merge_products(
        live_products=live_products,
        csv_products=csv_products,
        max_results=max_results,
    )

    notes.extend(
        [
            "High-recall retrieval completed.",
            "Browserbase V4 uses strict product URL validation before and after enrichment.",
            "Homepage, shop-by-style, category, collaboration, inspiration, and other non-product pages are dropped.",
            "Invalid images such as [object Object], SVG icons, logos, and tracking pixels are removed.",
            "Amazon URLs are canonicalized and deduped by ASIN.",
            "Promo labels such as CLEARANCE are not treated as prices.",
            "New valid live products are appended to the CSV cache for future runs.",
            "No candidate scoring, ranking, or Gemini post-processing was performed.",
            "Downstream verifier and cross-encoder should handle quality and semantic relevance.",
            f"effective_query={effective_query}",
            f"live_candidates={len(live_products)}",
            f"live_enriched_candidates={sum(1 for p in live_products if p.get('is_enriched_product_page'))}",
            f"csv_candidates={len(csv_products)}",
            f"merged_candidates={len(merged)}",
            f"csv_append_num_appended={csv_append_result.get('num_appended', 0)}",
            f"csv_append_path={csv_append_result.get('csv_path')}",
            f"sites={sites}",
            f"max_product_pages={max_product_pages}",
            f"product_page_concurrency={product_page_concurrency}",
        ]
    )

    return {
        "products": merged,
        "notes": notes,
        "csv_append_result": csv_append_result,
    }


# ============================================================
# 7. Local smoke test
# ============================================================

async def _smoke_test() -> None:
    result = await browse_products_with_browserbase(
        query="modern contemporary home decor accents wall art vases sculptures throw pillows table decor under $500",
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
        max_product_pages=24,
        visit_product_pages=True,
        product_page_concurrency=2,
        update_csv_cache=True,
        max_csv_append=100,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_smoke_test())
