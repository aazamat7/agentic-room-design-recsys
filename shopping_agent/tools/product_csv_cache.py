# shopping_agent/tools/product_csv_cache.py
# ============================================================
# CSV cache utilities for product candidate persistence.
#
# Purpose:
#   Append newly discovered live Browserbase products into the CSV cache
#   only if they are not already present.
#
# Rules:
#   - Never append rows that came from the CSV itself.
#   - Prefer enriched product-page rows.
#   - Dedupe by normalized product identity:
#       Amazon -> ASIN
#       Others -> domain + normalized product path
#   - Keep schema stable.
# ============================================================

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse


DEFAULT_CSV_PATH = "data/browserbase_results_final_with_ho_05.csv"

PRODUCT_CSV_COLUMNS = [
    "title",
    "description",
    "price_text",
    "product_url",
    "image_url",
    "source_site",
    "category",
    "retrieval_source",
    "search_phrase",
    "site_key",
    "is_enriched_product_page",
]


def clean_text(value: Optional[Any], max_len: int = 1000) -> str:
    if value is None:
        return ""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(value))
    text = " ".join(text.split())

    if len(text) > max_len:
        return text[: max_len - 3] + "..."

    return text


def extract_amazon_asin(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path
    except Exception:
        return None

    for pattern in [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product/([A-Z0-9]{10})",
    ]:
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

    if "amazon.com" in parsed.netloc.lower():
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        redirect_url = params.get("url")

        if redirect_url and redirect_url.startswith("/"):
            return normalize_url("https://www.amazon.com" + unquote(redirect_url))

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


def product_identity_key(url: Optional[str]) -> str:
    normalized = normalize_url(url)

    if not normalized:
        return ""

    asin = extract_amazon_asin(normalized)
    if asin:
        return f"amazon:{asin}"

    parsed = urlparse(normalized)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.rstrip("/").lower()

    return f"{domain}:{path}"


def resolve_csv_path(csv_data_path: Optional[str] = None) -> Path:
    candidate = (
        csv_data_path
        or os.environ.get("PRODUCT_CANDIDATE_CSV_PATH")
        or DEFAULT_CSV_PATH
    )

    return Path(candidate)


def read_existing_product_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    existing: set[str] = set()

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                url = (
                    row.get("product_url")
                    or row.get("url")
                    or row.get("link")
                    or row.get("href")
                )
                key = product_identity_key(url)

                if key:
                    existing.add(key)

    except Exception as exc:
        print(f"[CSV_APPEND][WARN] Could not read existing CSV keys: {exc}")

    return existing


def should_append_product(product: Dict[str, Any]) -> bool:
    """
    Conservative quality gate for cache writes.

    Retrieval can keep noisy candidates for recall, but cache should only persist
    reasonably useful rows.
    """
    if not isinstance(product, dict):
        return False

    if product.get("retrieval_source") == "csv_candidate_cache":
        return False

    url = normalize_url(product.get("product_url"))
    title = clean_text(product.get("title"), max_len=200)

    if not url or not title:
        return False

    bad_url_terms = [
        "/search",
        "/category",
        "/categories",
        "/collections",
        "/collection",
        "/shop-by-style",
        "/collaborations",
        "/blog",
        "/ideas",
        "/inspiration",
        "/cart",
        "/account",
    ]

    if any(term in url.lower() for term in bad_url_terms):
        return False

    bad_title_terms = {
        "debug info copied.",
        "shop by style",
        "home",
        "+6",
    }

    if title.lower().strip() in bad_title_terms:
        return False

    # Prefer enriched rows; allow non-enriched only if price or image exists.
    if not product.get("is_enriched_product_page"):
        if not product.get("price_text") and not product.get("image_url"):
            return False

    return True


def product_to_csv_row(product: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": clean_text(product.get("title"), max_len=220),
        "description": clean_text(product.get("description"), max_len=700),
        "price_text": clean_text(product.get("price_text"), max_len=80),
        "product_url": normalize_url(product.get("product_url")),
        "image_url": normalize_url(product.get("image_url")),
        "source_site": clean_text(product.get("source_site"), max_len=80),
        "category": clean_text(product.get("category"), max_len=120),
        "retrieval_source": clean_text(product.get("retrieval_source"), max_len=120),
        "search_phrase": clean_text(product.get("search_phrase"), max_len=500),
        "site_key": clean_text(product.get("site_key"), max_len=80),
        "is_enriched_product_page": str(bool(product.get("is_enriched_product_page"))),
    }


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRODUCT_CSV_COLUMNS)
        writer.writeheader()


def append_new_products_to_csv_cache(
    products: Iterable[Dict[str, Any]],
    csv_data_path: Optional[str] = None,
    max_append: int = 100,
) -> Dict[str, Any]:
    """
    Append new products to the cache CSV.

    Returns:
      {
        "csv_path": "...",
        "num_seen_existing": int,
        "num_candidates_considered": int,
        "num_appended": int,
        "appended_urls": [...]
      }
    """
    csv_path = resolve_csv_path(csv_data_path)
    ensure_csv_header(csv_path)

    existing_keys = read_existing_product_keys(csv_path)

    rows_to_append: List[Dict[str, str]] = []
    appended_urls: List[str] = []
    local_seen = set(existing_keys)
    considered = 0

    for product in products:
        considered += 1

        if not should_append_product(product):
            continue

        url = normalize_url(product.get("product_url"))
        key = product_identity_key(url)

        if not key or key in local_seen:
            continue

        local_seen.add(key)
        rows_to_append.append(product_to_csv_row(product))
        appended_urls.append(url)

        if len(rows_to_append) >= max_append:
            break

    if rows_to_append:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PRODUCT_CSV_COLUMNS)
            writer.writerows(rows_to_append)

    print(
        "[CSV_APPEND] "
        f"path={csv_path} existing={len(existing_keys)} "
        f"considered={considered} appended={len(rows_to_append)}"
    )

    return {
        "csv_path": str(csv_path),
        "num_seen_existing": len(existing_keys),
        "num_candidates_considered": considered,
        "num_appended": len(rows_to_append),
        "appended_urls": appended_urls,
    }
