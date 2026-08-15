from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)


PRICE_REGEX = re.compile(
    r"(?P<currency>₹|Rs\.?|INR|\$|USD|£|GBP|€|EUR)\s?"
    r"(?P<amount>[\d,]+(?:\.\d{1,2})?)",
    flags=re.IGNORECASE,
)


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


def as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if value is None:
        return []

    text = str(value).strip()
    return [text] if text else []


def extract_products(value: Any) -> List[Dict[str, Any]]:
    """
    Robustly extracts products from ADK state/tool wrappers.

    Handles:
    {"products": [...]}
    {"web_discovery_output": {"products": [...]}}
    {"verified_products": [...]}
    {"verification_output": {"verified_products": [...]}}
    """
    data = safe_json_loads(value)

    product_keys = {
        "products",
        "verified_products",
        "ranked_products",
        "recommended_products",
        "final_products",
    }

    def looks_like_product_list(items: List[Any]) -> bool:
        dict_items = [x for x in items if isinstance(x, dict)]
        if not dict_items:
            return False

        return any(
            "product_url" in item or "url" in item or "title" in item
            for item in dict_items
        )

    def walk(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            if looks_like_product_list(obj):
                return [x for x in obj if isinstance(x, dict)]
            return []

        if not isinstance(obj, dict):
            return []

        for key in product_keys:
            maybe = obj.get(key)
            if isinstance(maybe, list) and looks_like_product_list(maybe):
                return [x for x in maybe if isinstance(x, dict)]

        for nested in obj.values():
            found = walk(nested)
            if found:
                return found

        return []

    products = walk(data)

    deduped: List[Dict[str, Any]] = []
    seen_urls = set()

    for product in products:
        url = product.get("product_url") or product.get("url")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        deduped.append(product)

    return deduped


def build_cross_encoder_query(planner_json: Any) -> str:
    planner = safe_json_loads(planner_json)

    if not isinstance(planner, dict):
        return "Find the best matching ecommerce products."

    profile = planner.get("user_profile", {}) or {}
    constraints = planner.get("constraints", {}) or {}

    parts = [
        planner.get("interpreted_need", ""),
        planner.get("browser_query", ""),
        profile.get("room_or_use_case", ""),
        "styles: " + ", ".join(as_list(profile.get("styles"))),
        "colors: " + ", ".join(as_list(profile.get("colors"))),
        "materials: " + ", ".join(as_list(profile.get("materials"))),
        "brands: " + ", ".join(as_list(profile.get("brands"))),
        "avoid: " + ", ".join(as_list(profile.get("avoid"))),
        "required categories: " + ", ".join(
            as_list(constraints.get("required_categories"))
        ),
        "must have: " + ", ".join(as_list(constraints.get("must_have"))),
        "nice to have: " + ", ".join(as_list(constraints.get("nice_to_have"))),
    ]

    total_budget = constraints.get("total_budget")
    currency = constraints.get("currency")

    if total_budget:
        parts.append(f"budget under {total_budget} {currency or ''}")

    query = " ".join(str(p) for p in parts if p).strip()

    return query or "Find the best matching ecommerce products."


def product_to_text(product: Dict[str, Any]) -> str:
    return " ".join(
        [
            f"title: {product.get('title', '')}",
            f"category: {product.get('category', '')}",
            f"description: {product.get('description', '')}",
            f"price: {product.get('price_text', '')}",
            f"source: {product.get('source_site', '')}",
            f"why: {product.get('why_it_matches', '')}",
            f"url: {product.get('product_url', '')}",
        ]
    ).strip()


def parse_price_amount(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None

    match = PRICE_REGEX.search(str(price_text))

    if not match:
        return None

    try:
        return float(match.group("amount").replace(",", ""))
    except ValueError:
        return None


def extract_budget(planner_json: Any) -> Optional[float]:
    planner = safe_json_loads(planner_json)

    if not isinstance(planner, dict):
        return None

    constraints = planner.get("constraints", {}) or {}
    budget = constraints.get("total_budget")

    if budget is None:
        return None

    try:
        return float(budget)
    except (TypeError, ValueError):
        return None


def minmax(values: List[float]) -> List[float]:
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)

    if math.isclose(v_min, v_max):
        return [0.5 for _ in values]

    return [(v - v_min) / (v_max - v_min) for v in values]


def lexical_fallback_scores(query: str, products: List[Dict[str, Any]]) -> List[float]:
    query_terms = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) >= 3
    ]

    scores: List[float] = []

    for product in products:
        text = product_to_text(product).lower()
        overlap = sum(1 for term in query_terms if term in text)

        has_image = 1.0 if product.get("image_url") else 0.0
        has_price = 1.0 if product.get("price_text") else 0.0

        scores.append(float(overlap) + 0.25 * has_image + 0.25 * has_price)

    return scores


@lru_cache(maxsize=1)
def get_cross_encoder_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)


def constraint_bonus(
    planner_json: Any,
    product: Dict[str, Any],
) -> Tuple[float, List[str]]:
    planner = safe_json_loads(planner_json)

    if not isinstance(planner, dict):
        return 0.0, []

    profile = planner.get("user_profile", {}) or {}
    constraints = planner.get("constraints", {}) or {}

    text = product_to_text(product).lower()
    bonus = 0.0
    reasons: List[str] = []

    required_categories = [
        x.lower() for x in as_list(constraints.get("required_categories"))
    ]
    must_have = [x.lower() for x in as_list(constraints.get("must_have"))]
    nice_to_have = [x.lower() for x in as_list(constraints.get("nice_to_have"))]
    styles = [x.lower() for x in as_list(profile.get("styles"))]
    colors = [x.lower() for x in as_list(profile.get("colors"))]
    materials = [x.lower() for x in as_list(profile.get("materials"))]
    avoid = [x.lower() for x in as_list(profile.get("avoid"))]

    for term in required_categories:
        if term and term in text:
            bonus += 0.12
            reasons.append(f"category_match:{term}")

    for term in must_have:
        if term and term in text:
            bonus += 0.08
            reasons.append(f"must_have_match:{term}")

    for term in nice_to_have:
        if term and term in text:
            bonus += 0.04
            reasons.append(f"nice_to_have_match:{term}")

    for term in styles:
        if term and term in text:
            bonus += 0.06
            reasons.append(f"style_match:{term}")

    for term in colors:
        if term and term in text:
            bonus += 0.03
            reasons.append(f"color_match:{term}")

    for term in materials:
        if term and term in text:
            bonus += 0.04
            reasons.append(f"material_match:{term}")

    for term in avoid:
        if term and term in text:
            bonus -= 0.20
            reasons.append(f"avoid_penalty:{term}")

    if product.get("image_url"):
        bonus += 0.03
        reasons.append("has_image")

    if product.get("price_text"):
        bonus += 0.03
        reasons.append("has_price")

    budget = extract_budget(planner_json)
    price = parse_price_amount(product.get("price_text"))

    if budget is not None and price is not None:
        if price <= budget:
            bonus += 0.05
            reasons.append("price_within_budget")
        else:
            bonus -= 0.08
            reasons.append("price_above_budget")

    return round(bonus, 4), reasons


def rerank_product_cards_cross_encoder(
    planner_json: Any,
    candidate_products_json: Any,
    max_results: int = 20,
    cross_encoder_weight: float = 0.82,
    constraint_weight: float = 0.18,
) -> Dict[str, Any]:
    query = build_cross_encoder_query(planner_json)
    products = extract_products(candidate_products_json)

    if not products:
        return {
            "ranked_products": [],
            "ranking_policy": "cross_encoder_v1",
            "cross_encoder_model": DEFAULT_CROSS_ENCODER_MODEL,
            "query_used_for_reranking": query,
            "notes": [
                "No products available for reranking.",
                f"candidate_products_json_type={type(candidate_products_json).__name__}",
            ],
        }

    product_texts = [product_to_text(product) for product in products]
    pairs = [[query, text] for text in product_texts]

    ranking_policy = "cross_encoder_v1"
    model_name = DEFAULT_CROSS_ENCODER_MODEL
    notes: List[str] = []

    try:
        model = get_cross_encoder_model()
        raw_scores = model.predict(pairs)
        raw_scores = [float(score) for score in raw_scores]
    except Exception as exc:
        ranking_policy = "lexical_fallback_after_cross_encoder_error"
        model_name = None
        raw_scores = lexical_fallback_scores(query, products)
        notes.append(f"Cross-encoder fallback used: {type(exc).__name__}: {exc}")

    normalized_scores = minmax(raw_scores)

    ranked: List[Dict[str, Any]] = []

    for product, raw_score, normalized_score in zip(
        products,
        raw_scores,
        normalized_scores,
    ):
        bonus, reasons = constraint_bonus(planner_json, product)

        final_score = (
            cross_encoder_weight * normalized_score
            + constraint_weight * bonus
        )

        item = dict(product)
        item["cross_encoder_raw_score"] = round(float(raw_score), 4)
        item["cross_encoder_score"] = round(float(normalized_score), 4)
        item["constraint_bonus"] = bonus
        item["final_score"] = round(float(final_score), 4)
        item["score_reasons"] = [
            f"cross_encoder_score={round(float(normalized_score), 4)}",
            f"constraint_bonus={bonus}",
            *reasons,
        ]

        ranked.append(item)

    ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return {
        "ranked_products": ranked[:max_results],
        "num_input_products": len(products),
        "num_ranked_products": min(len(ranked), max_results),
        "ranking_policy": ranking_policy,
        "cross_encoder_model": model_name,
        "query_used_for_reranking": query,
        "notes": notes,
    }