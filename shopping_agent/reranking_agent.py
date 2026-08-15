# from __future__ import annotations

# import json
# import logging
# from typing import Any, AsyncGenerator, Dict

# from typing_extensions import override

# from google.adk.agents import BaseAgent
# from google.adk.agents.invocation_context import InvocationContext
# from google.adk.events import Event, EventActions
# from google.genai import types

# from shopping_agent.tools.cross_encoder_reranker import (
#     rerank_product_cards_cross_encoder,
# )


# logger = logging.getLogger(__name__)


# class CrossEncoderRerankingAgent(BaseAgent):
#     """
#     Non-LLM reranking agent.

#     Uses a cross-encoder model to score query-product pairs.
#     This is similar to RAG reranking, but applied to ecommerce product cards.
#     """

#     @override
#     async def _run_async_impl(
#         self,
#         ctx: InvocationContext,
#     ) -> AsyncGenerator[Event, None]:
#         logger.info("[%s] Starting cross-encoder reranking.", self.name)

#         planner_output = ctx.session.state.get("planner_output", {})
#         verification_output = ctx.session.state.get("verification_output", {})

#         reranking_output: Dict[str, Any] = rerank_product_cards_cross_encoder(
#             planner_json=planner_output,
#             verified_products_json=verification_output,
#             max_results=30,
#             cross_encoder_weight=0.75,
#             browser_relevance_weight=0.15,
#             constraint_weight=0.10,
#         )

#         num_ranked = len(reranking_output.get("ranked_products", []))

#         logger.info("[%s] Reranked %d products.", self.name, num_ranked)

#         yield Event(
#             author=self.name,
#             actions=EventActions(
#                 state_delta={
#                     "reranking_output": reranking_output,
#                 }
#             ),
#             content=types.Content(
#                 role="model",
#                 parts=[
#                     types.Part(
#                         text=json.dumps(
#                             {
#                                 "status": "cross_encoder_reranking_completed",
#                                 "ranking_policy": reranking_output.get(
#                                     "ranking_policy",
#                                     "cross_encoder_v1",
#                                 ),
#                                 "cross_encoder_model": reranking_output.get(
#                                     "cross_encoder_model"
#                                 ),
#                                 "num_ranked_products": num_ranked,
#                             },
#                             ensure_ascii=False,
#                         )
#                     )
#                 ],
#             ),
#         )


# reranking_agent = CrossEncoderRerankingAgent(
#     name="CrossEncoderRerankingAgent",
#     description=(
#         "Deterministically reranks verified products using a cross-encoder "
#         "semantic relevance model plus constraint-aware business scoring. "
#         "This agent makes no LLM calls."
#     ),
# )



# from __future__ import annotations

# import json
# import logging
# from typing import Any, AsyncGenerator, Dict

# from typing_extensions import override

# from google.adk.agents import BaseAgent
# from google.adk.agents.invocation_context import InvocationContext
# from google.adk.events import Event, EventActions
# from google.genai import types

# from shopping_agent.tools.cross_encoder_reranker import (
#     rerank_product_cards_cross_encoder,
# )


# logger = logging.getLogger(__name__)


# class CrossEncoderRerankingAgent(BaseAgent):
#     """
#     Non-LLM reranking agent.

#     Reads:
#       - planner_output
#       - web_discovery_output

#     Writes:
#       - reranking_output

#     This agent performs semantic query-product reranking using a cross-encoder.
#     """

#     @override
#     async def _run_async_impl(
#         self,
#         ctx: InvocationContext,
#     ) -> AsyncGenerator[Event, None]:
#         planner_output = ctx.session.state.get("planner_output", {})
#         web_discovery_output = ctx.session.state.get("web_discovery_output", {})

#         logger.info("[%s] Starting cross-encoder reranking.", self.name)

#         reranking_output: Dict[str, Any] = rerank_product_cards_cross_encoder(
#             planner_json=planner_output,
#             candidate_products_json=web_discovery_output,
#             max_results=20,
#             cross_encoder_weight=0.82,
#             constraint_weight=0.18,
#         )

#         num_ranked = len(reranking_output.get("ranked_products", []))

#         logger.info("[%s] Reranked %d products.", self.name, num_ranked)

#         yield Event(
#             author=self.name,
#             actions=EventActions(
#                 state_delta={
#                     "reranking_output": reranking_output,
#                 }
#             ),
#             content=types.Content(
#                 role="model",
#                 parts=[
#                     types.Part(
#                         text=json.dumps(
#                             {
#                                 "status": "cross_encoder_reranking_completed",
#                                 "num_ranked_products": num_ranked,
#                                 "ranking_policy": reranking_output.get(
#                                     "ranking_policy"
#                                 ),
#                                 "cross_encoder_model": reranking_output.get(
#                                     "cross_encoder_model"
#                                 ),
#                                 "notes": reranking_output.get("notes", []),
#                             },
#                             ensure_ascii=False,
#                         )
#                     )
#                 ],
#             ),
#         )


# reranking_agent = CrossEncoderRerankingAgent(
#     name="CrossEncoderRerankingAgent",
#     description=(
#         "Reranks high-recall ecommerce candidates using a cross-encoder. "
#         "This agent makes no LLM calls."
#     ),
# )

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict

from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from shopping_agent.tools.cross_encoder_reranker import (
    rerank_product_cards_cross_encoder,
)
from shopping_agent.tools.product_dedupe import (
    diversify_ranked_products,
    preprocess_candidate_products_json,
)


logger = logging.getLogger(__name__)


class CrossEncoderRerankingAgent(BaseAgent):
    """
    Non-LLM reranking agent.

    Reads:
      - planner_output
      - web_discovery_output

    Writes:
      - reranking_output

    This agent performs:
      1. candidate cleanup
      2. exact URL dedupe
      3. cross-encoder semantic reranking
      4. product-family diversity filtering
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        planner_output = ctx.session.state.get("planner_output", {})
        web_discovery_output = ctx.session.state.get("web_discovery_output", {})

        logger.info("[%s] Starting cross-encoder reranking.", self.name)

        cleaned_web_output, dedupe_summary = preprocess_candidate_products_json(
            web_discovery_output
        )

        # Ask the cross-encoder for more than the final output size.
        # Otherwise duplicate variants may occupy all top-20 slots.
        raw_reranking_output: Dict[str, Any] = rerank_product_cards_cross_encoder(
            planner_json=planner_output,
            candidate_products_json=cleaned_web_output,
            max_results=80,
            cross_encoder_weight=0.82,
            constraint_weight=0.18,
        )

        raw_ranked = raw_reranking_output.get("ranked_products", [])

        diverse_ranked, diversity_summary = diversify_ranked_products(
            raw_ranked,
            max_results=20,
            max_per_family=1,
            max_per_site=5,
            near_duplicate_title_threshold=0.92,
        )

        reranking_output = dict(raw_reranking_output)
        reranking_output["ranked_products"] = diverse_ranked
        reranking_output["num_ranked_products_before_diversity"] = len(raw_ranked)
        reranking_output["num_ranked_products"] = len(diverse_ranked)
        reranking_output["dedupe_summary"] = dedupe_summary
        reranking_output["diversity_summary"] = diversity_summary
        reranking_output["ranking_policy"] = (
            str(reranking_output.get("ranking_policy") or "")
            + " + exact_url_dedupe + product_family_diversity_filter"
        ).strip(" +")

        reranking_output.setdefault("notes", [])
        reranking_output["notes"].append(
            "Applied product-family dedupe so near-identical color/material variants do not dominate the final ranked list."
        )

        num_ranked = len(diverse_ranked)

        logger.info(
            "[%s] Reranked %d raw products into %d diverse products.",
            self.name,
            len(raw_ranked),
            num_ranked,
        )

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "reranking_output": reranking_output,
                }
            ),
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            {
                                "status": "cross_encoder_reranking_completed",
                                "num_ranked_products_before_diversity": len(raw_ranked),
                                "num_ranked_products": num_ranked,
                                "dedupe_summary": dedupe_summary,
                                "diversity_summary": diversity_summary,
                                "ranking_policy": reranking_output.get(
                                    "ranking_policy"
                                ),
                                "cross_encoder_model": reranking_output.get(
                                    "cross_encoder_model"
                                ),
                                "notes": reranking_output.get("notes", []),
                            },
                            ensure_ascii=False,
                        )
                    )
                ],
            ),
        )


reranking_agent = CrossEncoderRerankingAgent(
    name="CrossEncoderRerankingAgent",
    description=(
        "Reranks high-recall ecommerce candidates using a cross-encoder, then "
        "deduplicates product-family variants. This agent makes no LLM calls."
    ),
)