from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


BAD_URL_TERMS = [
    "/search",
    "search?",
    "/cart",
    "/checkout",
    "/login",
    "/signin",
    "/account",
    "/blog",
    "/ideas",
    "/inspiration",
    "/help",
    "/support",
    "/privacy",
    "/terms",
]

PRODUCT_URL_HINTS = [
    "/product",
    "/products",
    "/p/",
    "/dp/",
    "/ip/",
    "/item",
    "/listing",
    "/pd/",
    "sku=",
    "pid=",
    "productid",
]

PRICE_REGEX = re.compile(
    r"(?P<currency>₹|Rs\.?|INR|\$|USD|£|GBP|€|EUR)\s?"
    r"(?P<amount>[\d,]+(?:\.\d{1,2})?)",
    flags=re.IGNORECASE,
)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value

    if value is None:
        return {}

    text = str(value).strip()
    if not text:
        return {}

    text = _strip_json_fence(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_products(value: Any) -> List[Dict[str, Any]]:
    data = _safe_json_loads(value)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if not isinstance(data, dict):
        return []

    for key in [
        "products",
        "recommended_products",
        "verified_products",
        "ranked_products",
        "final_products",
    ]:
        products = data.get(key)
        if isinstance(products, list):
            return [x for x in products if isinstance(x, dict)]

    nested_keys = [
        "tool_result",
        "result",
        "web_discovery_output",
        "bundle_result",
        "validated_recommendation",
    ]

    for key in nested_keys:
        nested = data.get(key)
        if nested:
            products = _extract_products(nested)
            if products:
                return products

    return []


def _normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""

    url = str(url).strip()

    if not url.startswith(("http://", "https://", "data:")):
        return ""

    return url


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _looks_like_product_url(url: str) -> bool:
    url_l = url.lower()

    if any(term in url_l for term in BAD_URL_TERMS):
        return False

    return any(hint in url_l for hint in PRODUCT_URL_HINTS)


def _parse_price_amount(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None

    match = PRICE_REGEX.search(str(price_text))
    if not match:
        return None

    amount = match.group("amount").replace(",", "")

    try:
        return float(amount)
    except ValueError:
        return None


def _as_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(k) or "")
        for k in [
            "title",
            "description",
            "category",
            "why_it_matches",
            "source_site",
            "price_text",
        ]
    ).lower()


def verify_product_cards(
    product_candidates_json: str,
    require_image: bool = True,
) -> Dict[str, Any]:
    """
    Deterministically verify product cards produced by Browserbase.

    This tool does not invent or enrich products. It only filters and annotates.
    """
    products = _extract_products(product_candidates_json)

    verified: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in products:
        issues: List[str] = []

        product_url = _normalize_url(
            item.get("product_url") or item.get("url")
        )
        image_url = _normalize_url(item.get("image_url"))

        if not product_url:
            issues.append("missing_or_invalid_product_url")

        if product_url and product_url in seen_urls:
            issues.append("duplicate_product_url")

        if product_url:
            seen_urls.add(product_url)

        if product_url and not _looks_like_product_url(product_url):
            issues.append("url_does_not_look_like_product_page")

        if require_image and not image_url:
            issues.append("missing_or_invalid_image_url")

        if not str(item.get("title") or "").strip():
            issues.append("missing_title")

        if product_url and not _host(product_url):
            issues.append("missing_host")

        normalized = {
            "title": str(item.get("title") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "price_text": item.get("price_text"),
            "product_url": product_url,
            "image_url": image_url or None,
            "source_site": item.get("source_site") or _host(product_url),
            "category": item.get("category"),
            "relevance_score": int(item.get("relevance_score") or 0),
            "why_it_matches": str(item.get("why_it_matches") or "").strip(),
            "is_valid": len(issues) == 0,
            "verification_issues": issues,
        }

        if issues:
            rejected.append(normalized)
        else:
            verified.append(normalized)

    return {
        "verified_products": verified,
        "rejected_products": rejected,
        "num_input_products": len(products),
        "num_verified_products": len(verified),
        "num_rejected_products": len(rejected),
    }


def rerank_product_cards(
    planner_json: str,
    verified_products_json: str,
    max_results: int = 30,
) -> Dict[str, Any]:
    """
    Transparent rule-based reranker.

    Later this can be replaced by a Vertex AI ranking model, contextual bandit,
    or learned scoring model.
    """
    planner = _safe_json_loads(planner_json)
    products = _extract_products(verified_products_json)

    profile = planner.get("user_profile", {}) if isinstance(planner, dict) else {}
    constraints = planner.get("constraints", {}) if isinstance(planner, dict) else {}

    style_terms = [str(x).lower() for x in profile.get("styles", [])]
    color_terms = [str(x).lower() for x in profile.get("colors", [])]
    material_terms = [str(x).lower() for x in profile.get("materials", [])]
    brand_terms = [str(x).lower() for x in profile.get("brands", [])]
    avoid_terms = [str(x).lower() for x in profile.get("avoid", [])]
    required_categories = [
        str(x).lower() for x in constraints.get("required_categories", [])
    ]
    budget = constraints.get("total_budget")

    ranked: List[Dict[str, Any]] = []

    for product in products:
        text = _as_text(product)
        score = float(product.get("relevance_score") or 0)
        reasons = [f"base_relevance={score:g}"]

        for term in style_terms:
            if term and term in text:
                score += 10
                reasons.append(f"style_match:{term}")

        for term in color_terms:
            if term and term in text:
                score += 5
                reasons.append(f"color_match:{term}")

        for term in material_terms:
            if term and term in text:
                score += 5
                reasons.append(f"material_match:{term}")

        for term in brand_terms:
            if term and term in text:
                score += 6
                reasons.append(f"brand_match:{term}")

        for category in required_categories:
            if category and category in text:
                score += 12
                reasons.append(f"category_match:{category}")

        for term in avoid_terms:
            if term and term in text:
                score -= 20
                reasons.append(f"avoid_penalty:{term}")

        price_amount = _parse_price_amount(product.get("price_text"))
        if budget is not None and price_amount is not None:
            try:
                if price_amount <= float(budget):
                    score += 4
                    reasons.append("within_budget_item")
            except (TypeError, ValueError):
                pass

        if product.get("image_url"):
            score += 3
            reasons.append("has_image")

        if product.get("price_text"):
            score += 3
            reasons.append("has_price")

        ranked_item = dict(product)
        ranked_item["final_score"] = round(score, 3)
        ranked_item["score_reasons"] = reasons
        ranked.append(ranked_item)

    ranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return {
        "ranked_products": ranked[:max_results],
        "ranking_policy": "rule_based_v1",
    }


def build_recommendation_bundle(
    planner_json: str,
    ranked_products_json: str,
    max_products: int = 8,
) -> Dict[str, Any]:
    """
    Build a coherent bundle.

    If required categories are present, select one best product per category.
    Otherwise return the top-ranked products.
    """
    planner = _safe_json_loads(planner_json)
    ranked = _extract_products(ranked_products_json)

    constraints = planner.get("constraints", {}) if isinstance(planner, dict) else {}
    required_categories = constraints.get("required_categories", []) or []
    required_categories_l = [str(x).lower() for x in required_categories]

    if not required_categories_l:
        return {
            "products": ranked[:max_products],
            "missing_categories": [],
            "notes": ["No required categories were specified; returned top ranked products."],
        }

    selected: List[Dict[str, Any]] = []
    used_urls: set[str] = set()
    matched_categories: set[str] = set()

    for category_raw, category_l in zip(required_categories, required_categories_l):
        category_matches = []

        for product in ranked:
            if product.get("product_url") in used_urls:
                continue

            text = _as_text(product)
            category_field = str(product.get("category") or "").lower()

            if category_l in text or category_l in category_field:
                category_matches.append(product)

        if category_matches:
            chosen = category_matches[0]
            chosen = dict(chosen)
            chosen["selected_for_category"] = category_raw
            selected.append(chosen)
            used_urls.add(chosen.get("product_url"))
            matched_categories.add(category_l)

    missing = [
        original
        for original, normalized in zip(required_categories, required_categories_l)
        if normalized not in matched_categories
    ]

    return {
        "products": selected[:max_products],
        "missing_categories": missing,
        "notes": [
            "Bundle built with one product per requested category where available."
        ],
    }


def critique_recommendation(
    planner_json: str,
    bundle_json: str,
) -> Dict[str, Any]:
    """
    Final deterministic guardrail.
    """
    planner = _safe_json_loads(planner_json)
    bundle = _safe_json_loads(bundle_json)
    products = _extract_products(bundle)

    constraints = planner.get("constraints", {}) if isinstance(planner, dict) else {}
    required_categories = constraints.get("required_categories", []) or []
    missing_categories = bundle.get("missing_categories", []) if isinstance(bundle, dict) else []

    issues: List[str] = []
    seen_urls: set[str] = set()

    for product in products:
        title = product.get("title") or "unknown_product"
        url = product.get("product_url")
        image_url = product.get("image_url")

        if not url:
            issues.append(f"missing_product_url:{title}")

        if not image_url:
            issues.append(f"missing_image_url:{title}")

        if url and url in seen_urls:
            issues.append(f"duplicate_product_url:{url}")

        if url:
            seen_urls.add(url)

    for category in missing_categories:
        issues.append(f"missing_required_category:{category}")

    if required_categories and not products:
        issues.append("no_products_selected_for_required_categories")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "final_products": products,
        "missing_categories": missing_categories,
        "notes": [
            "Final critic is deterministic and checks URL, image, duplicate, and category coverage."
        ],
    }