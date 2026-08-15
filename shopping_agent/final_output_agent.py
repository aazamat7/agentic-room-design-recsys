from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types


logger = logging.getLogger(__name__)


def _safe_json_loads(value: Any) -> Any:
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
        return {}


def _extract_ranked_products(reranking_output: Any) -> List[Dict[str, Any]]:
    data = _safe_json_loads(reranking_output)

    if isinstance(data, dict):
        products = data.get("ranked_products", [])
        if isinstance(products, list):
            return [p for p in products if isinstance(p, dict)]

    return []


class FinalOutputAgent(BaseAgent):
    """
    Deterministic final output formatter.

    Reads:
      - planner_output
      - reranking_output

    Writes:
      - final_output

    It does not invent products or URLs.
    """

    top_k: int = 8

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        planner_output = ctx.session.state.get("planner_output", {})
        reranking_output = ctx.session.state.get("reranking_output", {})

        ranked_products = _extract_ranked_products(reranking_output)
        top_products = ranked_products[: self.top_k]

        recommendations: List[Dict[str, Any]] = []

        for idx, product in enumerate(top_products, start=1):
            recommendations.append(
                {
                    "rank": idx,
                    "title": product.get("title", ""),
                    "description": product.get("description", ""),
                    "price_text": product.get("price_text"),
                    "product_url": product.get("product_url"),
                    "image_url": product.get("image_url"),
                    "source_site": product.get("source_site"),
                    "category": product.get("category"),
                    "final_score": product.get("final_score"),
                    "cross_encoder_score": product.get("cross_encoder_score"),
                    "constraint_bonus": product.get("constraint_bonus"),
                    "score_reasons": product.get("score_reasons", []),
                }
            )

        final_output = {
            "status": "completed" if recommendations else "no_recommendations",
            "num_recommendations": len(recommendations),
            "recommendations": recommendations,
            "notes": [],
        }

        if not recommendations:
            final_output["notes"].append(
                "No ranked products were available. Check web_discovery_output and reranking_output."
            )

        logger.info("[%s] Produced %d recommendations.", self.name, len(recommendations))

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "final_output": final_output,
                }
            ),
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            final_output,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                ],
            ),
        )


final_output_agent = FinalOutputAgent(
    name="FinalOutputAgent",
    description=(
        "Formats the top cross-encoder-ranked ecommerce products into final JSON. "
        "Makes no LLM calls and does not invent products."
    ),
)