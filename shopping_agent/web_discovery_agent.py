# import os

# from dotenv import load_dotenv

# # Load .env before ADK/model initialization.
# load_dotenv()

# # Force Vertex AI / ADC mode.
# # This prevents ADK from trying to use the Gemini Developer API key path.
# os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

# PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
# LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# if not PROJECT_ID:
#     raise RuntimeError(
#         "Missing GOOGLE_CLOUD_PROJECT. "
#         "Set it in .env or export it before running ADK."
#     )

# os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

# from google.adk.agents.llm_agent import LlmAgent

# from shopping_agent.tools.browserbase_product_browser_V1 import browse_products_with_browserbase


# MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


# web_discovery_agent = LlmAgent(
#     name="WebDiscoveryAgent",
#     model=MODEL,
#     description="Uses Browserbase to browse ecommerce sites and retrieve live product cards.",
#     instruction="""
# You are the Web Discovery Agent.

# Input planner JSON:
# {planner_output}

# Your task:
# 1. Read browser_query from the planner JSON.
# 2. Read country from constraints.country.
# 3. Call browse_products_with_browserbase.
# 4. Use max_results=20 and max_product_pages=24.
# 5. Do not invent product URLs, images, prices, titles, or descriptions.
# 6. Return only JSON with this schema:
# {
#   "products": [
#     {
#       "title": "...",
#       "description": "...",
#       "price_text": "...",
#       "product_url": "...",
#       "image_url": "...",
#       "source_site": "...",
#       "category": "...",
#       "relevance_score": 0,
#       "why_it_matches": "..."
#     }
#   ],
#   "notes": []
# }

# Important:
# - Products must come from the tool result.
# - If the tool returns no products, return products=[] and include the tool notes.
# """,
#     tools=[browse_products_with_browserbase],
#     output_key="web_discovery_output",
# )

# from __future__ import annotations

# import json
# import logging
# import os
# from typing import Any, AsyncGenerator, Dict

# from dotenv import load_dotenv
# from typing_extensions import override

# load_dotenv()
# os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

# from google.adk.agents import BaseAgent  # noqa: E402
# from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
# from google.adk.events import Event, EventActions  # noqa: E402
# from google.genai import types  # noqa: E402

# from shopping_agent.tools.browserbase_product_browser_V2 import (  # noqa: E402
#     browse_products_with_browserbase,
# )


# logger = logging.getLogger(__name__)


# def _safe_json_loads(value: Any) -> Any:
#     if isinstance(value, (dict, list)):
#         return value

#     if value is None:
#         return {}

#     text = str(value).strip()
#     if not text:
#         return {}

#     if text.startswith("```"):
#         text = text.replace("```json", "").replace("```", "").strip()

#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         return {}


# def _extract_planner_payload(planner_output: Any) -> Dict[str, Any]:
#     data = _safe_json_loads(planner_output)

#     if isinstance(data, dict):
#         if "browser_query" in data or "constraints" in data:
#             return data

#         for key in ["planner_output", "output", "result"]:
#             nested = data.get(key)
#             nested_data = _safe_json_loads(nested)
#             if isinstance(nested_data, dict) and (
#                 "browser_query" in nested_data or "constraints" in nested_data
#             ):
#                 return nested_data

#     return {}


# def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
#     cur: Any = data
#     for key in keys:
#         if not isinstance(cur, dict):
#             return default
#         cur = cur.get(key)
#     return cur if cur is not None else default


# class WebDiscoveryAgent(BaseAgent):
#     """
#     Non-LLM high-recall retrieval agent.

#     Reads planner_output from ADK session state, calls Browserbase/CSV retrieval,
#     and writes exact tool result to web_discovery_output.
#     """

#     @override
#     async def _run_async_impl(
#         self,
#         ctx: InvocationContext,
#     ) -> AsyncGenerator[Event, None]:
#         planner_output = ctx.session.state.get("planner_output", {})
#         planner = _extract_planner_payload(planner_output)

#         browser_query = (
#             planner.get("browser_query")
#             or planner.get("interpreted_need")
#             or "home decor furniture accessories"
#         )
#         country = _get_nested(planner, "constraints", "country", default="US")

#         logger.info("[%s] browser_query=%s", self.name, browser_query)
#         logger.info("[%s] country=%s", self.name, country)

#         tool_result = await browse_products_with_browserbase(
#             query=browser_query,
#             country=country,
#             max_results=50,
#             sites=None,
#             use_proxy=False,
#             use_csv_cache=True,
#             max_csv_candidates=150,
#         )

#         products = tool_result.get("products", [])
#         notes = tool_result.get("notes", [])

#         yield Event(
#             author=self.name,
#             actions=EventActions(
#                 state_delta={
#                     "web_discovery_output": tool_result,
#                 }
#             ),
#             content=types.Content(
#                 role="model",
#                 parts=[
#                     types.Part(
#                         text=json.dumps(
#                             {
#                                 "status": "web_discovery_completed",
#                                 "browser_query": browser_query,
#                                 "country": country,
#                                 "num_products": len(products),
#                                 "notes": notes,
#                             },
#                             ensure_ascii=False,
#                         )
#                     )
#                 ],
#             ),
#         )


# web_discovery_agent = WebDiscoveryAgent(
#     name="WebDiscoveryAgent",
#     description=(
#         "Deterministically retrieves high-recall ecommerce candidates using "
#         "Browserbase search pages plus CSV cache. Makes no LLM calls."
#     ),
# )

# shopping_agent/web_discovery_agent.py
# Deterministic WebDiscoveryAgent for Browserbase V3.
# Replaces the LlmAgent version so the candidate pool is not filtered by an LLM.

# shopping_agent/web_discovery_agent.py
# ============================================================
# Deterministic Web Discovery Agent for Browserbase V4 FINAL.
#
# Important:
#   This is NOT an LlmAgent.
#   It directly calls the Browserbase tool and stores the exact tool result.
#   This prevents the LLM from filtering/dropping retrieved candidates.
# ============================================================

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from typing_extensions import override

load_dotenv()
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

from google.adk.agents import BaseAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.events import Event, EventActions  # noqa: E402
from google.genai import types  # noqa: E402

from shopping_agent.tools.browserbase_product_browser import (  # noqa: E402
    browse_products_with_browserbase,
)


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


def _extract_planner_payload(planner_output: Any) -> Dict[str, Any]:
    data = _safe_json_loads(planner_output)

    if isinstance(data, dict):
        if "browser_query" in data or "constraints" in data:
            return data

        for key in ["planner_output", "output", "result"]:
            nested = data.get(key)
            nested_data = _safe_json_loads(nested)

            if isinstance(nested_data, dict) and (
                "browser_query" in nested_data or "constraints" in nested_data
            ):
                return nested_data

    return {}


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data

    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)

    return cur if cur is not None else default


class WebDiscoveryAgent(BaseAgent):
    """
    Non-LLM high-recall retrieval agent.

    Reads:
      - planner_output

    Writes:
      - web_discovery_output

    Does:
      - calls Browserbase tool
      - includes CSV fallback and CSV append
      - returns exact tool result

    Does NOT:
      - filter products
      - rank products
      - summarize products
      - rewrite product URLs
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        planner_output = ctx.session.state.get("planner_output", {})
        planner = _extract_planner_payload(planner_output)

        browser_query = (
            planner.get("browser_query")
            or planner.get("interpreted_need")
            or "home decor furniture accessories"
        )
        country = _get_nested(planner, "constraints", "country", default="US")

        logger.info("[%s] browser_query=%s", self.name, browser_query)
        logger.info("[%s] country=%s", self.name, country)

        tool_result = await browse_products_with_browserbase(
            query=browser_query,
            country=country,
            max_results=50,
            sites=None,
            use_proxy=False,
            use_csv_cache=True,
            max_csv_candidates=150,
            max_product_pages=24,
            visit_product_pages=True,
            product_page_concurrency=2,
            update_csv_cache=True,
            max_csv_append=100,
        )

        products = tool_result.get("products", [])
        notes = tool_result.get("notes", [])
        csv_append_result = tool_result.get("csv_append_result", {})

        logger.info("[%s] retrieved_products=%d", self.name, len(products))
        logger.info("[%s] csv_append_result=%s", self.name, csv_append_result)
        logger.info("[%s] notes=%s", self.name, notes)

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "web_discovery_output": tool_result,
                }
            ),
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            {
                                "status": "web_discovery_completed",
                                "browser_query": browser_query,
                                "country": country,
                                "num_products": len(products),
                                "csv_append_result": csv_append_result,
                                "notes": notes,
                            },
                            ensure_ascii=False,
                        )
                    )
                ],
            ),
        )


web_discovery_agent = WebDiscoveryAgent(
    name="WebDiscoveryAgent",
    description=(
        "Deterministically retrieves ecommerce product candidates using "
        "Browserbase strict product validation plus CSV cache. Makes no LLM calls."
    ),
)
