from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, urlparse


VARIANT_WORDS = {
    # colors
    "black", "white", "cream", "ivory", "beige", "tan", "brown", "natural",
    "gray", "grey", "charcoal", "green", "sage", "blue", "navy", "pink",
    "blush", "red", "orange", "yellow", "gold", "brass", "bronze",
    "silver",

    # common finish/material variants
    "oak", "walnut", "wood", "wooden", "metal", "glass", "rattan",
    "boucle", "linen", "velvet", "leather", "fabric",

    # sizes / packs
    "small", "medium", "large", "xl", "xs", "set", "pack",
}


URL_TITLE_PATTERN = re.compile(r"^https?://", flags=re.IGNORECASE)


def _is_url_like(value: Any) -> bool:
    return bool(URL_TITLE_PATTERN.match(str(value or "").strip()))


def _slug_to_title(slug: str) -> str:
    slug = unquote(slug or "")
    slug = re.sub(r"[-_]+", " ", slug)
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug.title()


def _product_slug_from_url(url: str) -> str:
    try:
        parsed = urlparse(str(url))
        parts = [p for p in parsed.path.split("/") if p]

        if not parts:
            return ""

        # Shopify / common ecommerce:
        # /products/cohen-bar-stool-natural-brown
        if "products" in parts:
            idx = parts.index("products")
            if idx + 1 < len(parts):
                return parts[idx + 1]

        return parts[-1]
    except Exception:
        return ""


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = unquote(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_trailing_variant_words(text: str) -> str:
    """
    Turns:
      cohen bar stool natural brown -> cohen bar stool

    But keeps:
      natural wood coffee table -> natural wood coffee table
    because the trailing token is not just a variant suffix.
    """
    tokens = _normalize_text(text).split()

    while len(tokens) > 2 and tokens[-1] in VARIANT_WORDS:
        tokens.pop()

    return " ".join(tokens)


def clean_product_title(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fixes bad product title extraction where title is actually product_url.
    """
    p = copy.deepcopy(product)

    raw_title = str(p.get("title") or "").strip()
    product_url = str(p.get("product_url") or "").strip()

    if raw_title:
        p.setdefault("raw_title", raw_title)

    if not raw_title or _is_url_like(raw_title):
        slug = _product_slug_from_url(product_url or raw_title)

        if slug:
            p["title"] = _slug_to_title(slug)
        elif product_url:
            p["title"] = _slug_to_title(_product_slug_from_url(product_url))
        else:
            p["title"] = "Untitled Product"

    return p


def product_family_key(product: Dict[str, Any]) -> str:
    """
    Creates a stable family key so color/material variants collapse.

    Example:
      Cohen Bar Stool Natural Brown
      Cohen Bar Stool Black
      Cohen Bar Stool Light Oak

    all become something close to:
      nathan james|cohen bar stool
    """
    site = _normalize_text(
        product.get("site_key")
        or product.get("source_site")
        or ""
    )

    product_url = str(product.get("product_url") or "")
    slug = _product_slug_from_url(product_url)

    if slug:
        base = _remove_trailing_variant_words(slug)
    else:
        title = str(product.get("title") or "")
        base = _remove_trailing_variant_words(title)

    if not base:
        base = _normalize_text(product_url)

    return f"{site}|{base}"


def title_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ta = _normalize_text(str(a.get("title") or ""))
    tb = _normalize_text(str(b.get("title") or ""))

    if not ta or not tb:
        return 0.0

    return SequenceMatcher(None, ta, tb).ratio()


def dedupe_exact_products(products: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Removes exact duplicate product URLs and fixes bad titles.
    Does NOT remove color variants yet.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    duplicates = 0

    for product in products or []:
        p = clean_product_title(product)

        url = str(p.get("product_url") or "").strip()
        parsed = urlparse(url)
        canonical_url = f"{parsed.netloc}{parsed.path}".lower()

        key = canonical_url or product_family_key(p)

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        p["product_family_key"] = product_family_key(p)
        out.append(p)

    return out, {
        "input_count": len(products or []),
        "output_count": len(out),
        "exact_duplicates_removed": duplicates,
    }


def _extract_products_container(candidate_products_json: Any) -> Tuple[Dict[str, Any], str]:
    """
    Supports several possible web discovery output shapes.
    """
    if isinstance(candidate_products_json, list):
        return {"products": candidate_products_json}, "products"

    if not isinstance(candidate_products_json, dict):
        return {"products": []}, "products"

    for key in [
        "products",
        "candidate_products",
        "product_cards",
        "results",
        "items",
    ]:
        if isinstance(candidate_products_json.get(key), list):
            return copy.deepcopy(candidate_products_json), key

    return copy.deepcopy(candidate_products_json), ""


def preprocess_candidate_products_json(candidate_products_json: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run before cross-encoder reranking.
    Cleans titles and removes exact duplicate URLs.
    """
    data, key = _extract_products_container(candidate_products_json)

    if not key:
        return data, {
            "input_count": 0,
            "output_count": 0,
            "exact_duplicates_removed": 0,
            "warning": "No product list key found.",
        }

    cleaned, summary = dedupe_exact_products(data.get(key, []))
    data[key] = cleaned

    return data, summary


def diversify_ranked_products(
    ranked_products: List[Dict[str, Any]],
    max_results: int = 20,
    max_per_family: int = 1,
    max_per_site: int = 5,
    near_duplicate_title_threshold: float = 0.92,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run after cross-encoder scoring.

    Keeps the highest-ranked product from each product family and filters
    near-identical variants such as same stool in different colors.
    """
    selected: List[Dict[str, Any]] = []
    family_counts: Dict[str, int] = {}
    site_counts: Dict[str, int] = {}

    skipped_same_family = 0
    skipped_same_site = 0
    skipped_near_title = 0

    for product in ranked_products or []:
        p = clean_product_title(product)
        family = p.get("product_family_key") or product_family_key(p)
        p["product_family_key"] = family

        site = _normalize_text(
            p.get("site_key")
            or p.get("source_site")
            or ""
        )

        if family_counts.get(family, 0) >= max_per_family:
            skipped_same_family += 1
            continue

        if site and site_counts.get(site, 0) >= max_per_site:
            skipped_same_site += 1
            continue

        near_duplicate = False
        for chosen in selected:
            if title_similarity(p, chosen) >= near_duplicate_title_threshold:
                near_duplicate = True
                break

        if near_duplicate:
            skipped_near_title += 1
            continue

        selected.append(p)
        family_counts[family] = family_counts.get(family, 0) + 1

        if site:
            site_counts[site] = site_counts.get(site, 0) + 1

        if len(selected) >= max_results:
            break

    return selected, {
        "input_count": len(ranked_products or []),
        "output_count": len(selected),
        "skipped_same_family": skipped_same_family,
        "skipped_same_site": skipped_same_site,
        "skipped_near_title": skipped_near_title,
        "max_per_family": max_per_family,
        "max_per_site": max_per_site,
    }