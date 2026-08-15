# shopping_agent/debug_utils.py
# ============================================================
# Debug helpers for ADK shopping agent pipeline.
#
# Goal:
#   Print and save full state snapshots after each pipeline step:
#   - planner_output
#   - web_discovery_output
#   - reranking_output
#   - final_output
#
# Output directory default:
#   debug_runs/<ADK_SESSION_ID or local_debug>/
#
# Env vars:
#   SHOPPING_AGENT_DEBUG=1
#   SHOPPING_AGENT_DEBUG_DIR=debug_runs
#   SHOPPING_AGENT_DEBUG_PREVIEW_PRODUCTS=10
# ============================================================

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def debug_enabled() -> bool:
    return os.environ.get("SHOPPING_AGENT_DEBUG", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
            return value

    return value


def json_default(value: Any) -> str:
    return str(value)


def to_pretty_json(value: Any) -> str:
    parsed = safe_json_loads(value)
    return json.dumps(
        parsed,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )


def get_session_id(ctx: Any) -> str:
    session = getattr(ctx, "session", None)

    for attr in ["id", "session_id", "name"]:
        value = getattr(session, attr, None)
        if value:
            return str(value).replace("/", "_").replace("\\", "_")

    return "local_debug"


def get_debug_dir(ctx: Any) -> Path:
    root = Path(os.environ.get("SHOPPING_AGENT_DEBUG_DIR", "debug_runs"))
    session_id = get_session_id(ctx)
    path = root / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_snapshot(
    ctx: Any,
    filename: str,
    payload: Any,
) -> Path:
    out_dir = get_debug_dir(ctx)
    path = out_dir / filename

    parsed = safe_json_loads(payload)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            parsed,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    return path


def extract_products(value: Any) -> List[Dict[str, Any]]:
    data = safe_json_loads(value)

    product_keys = [
        "products",
        "verified_products",
        "ranked_products",
        "recommendations",
        "final_products",
    ]

    def looks_like_products(items: Any) -> bool:
        if not isinstance(items, list):
            return False

        dict_items = [x for x in items if isinstance(x, dict)]

        if not dict_items:
            return False

        return any(
            "product_url" in item
            or "url" in item
            or "title" in item
            for item in dict_items
        )

    def walk(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            if looks_like_products(obj):
                return [x for x in obj if isinstance(x, dict)]
            return []

        if not isinstance(obj, dict):
            return []

        for key in product_keys:
            value = obj.get(key)
            if looks_like_products(value):
                return [x for x in value if isinstance(x, dict)]

        for nested in obj.values():
            found = walk(nested)
            if found:
                return found

        return []

    return walk(data)


def compact_product_row(product: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "rank": idx,
        "title": product.get("title", ""),
        "price_text": product.get("price_text"),
        "source_site": product.get("source_site"),
        "category": product.get("category"),
        "final_score": product.get("final_score"),
        "cross_encoder_score": product.get("cross_encoder_score"),
        "constraint_bonus": product.get("constraint_bonus"),
        "retrieval_source": product.get("retrieval_source"),
        "product_url": product.get("product_url") or product.get("url"),
    }


def print_product_preview(
    label: str,
    value: Any,
) -> None:
    products = extract_products(value)
    limit = int(os.environ.get("SHOPPING_AGENT_DEBUG_PREVIEW_PRODUCTS", "10"))

    print(f"\n[{label}] product_count={len(products)}")

    for idx, product in enumerate(products[:limit], start=1):
        row = compact_product_row(product, idx)
        print(
            f"  {idx:02d}. "
            f"title={row['title']!r} | "
            f"price={row['price_text']!r} | "
            f"source={row['source_site']!r} | "
            f"score={row['final_score']!r} | "
            f"url={row['product_url']!r}"
        )

    if len(products) > limit:
        print(f"  ... {len(products) - limit} more products not shown in preview")


def print_state_snapshot(
    label: str,
    key: str,
    value: Any,
    snapshot_path: Path,
    print_full_json: bool = True,
) -> None:
    print("\n" + "=" * 100)
    print(f"[DEBUG] {label}")
    print(f"[DEBUG] state_key={key}")
    print(f"[DEBUG] snapshot_path={snapshot_path}")

    parsed = safe_json_loads(value)

    if isinstance(parsed, dict):
        print(f"[DEBUG] top_level_keys={list(parsed.keys())}")
    elif isinstance(parsed, list):
        print(f"[DEBUG] list_len={len(parsed)}")
    else:
        print(f"[DEBUG] value_type={type(parsed).__name__}")

    print_product_preview(label=f"{label}:{key}", value=parsed)

    if print_full_json:
        print(f"\n[DEBUG] full_json_for={key}")
        print(to_pretty_json(parsed))

    print("=" * 100 + "\n")


def build_debug_event_payload(
    label: str,
    saved_files: List[str],
) -> Dict[str, Any]:
    return {
        "debug_status": "snapshot_written",
        "label": label,
        "saved_files": saved_files,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
