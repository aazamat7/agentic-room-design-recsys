# from __future__ import annotations

# import json
# import re
# from typing import Any, AsyncGenerator, Dict

# from typing_extensions import override

# from google.adk.agents import BaseAgent
# from google.adk.agents.invocation_context import InvocationContext
# from google.adk.events import Event

# from shopping_agent.visual_preference_agent import visual_preference_agent
# from shopping_agent.tools.planner_agent import planner_agent
# from shopping_agent.web_discovery_agent import web_discovery_agent
# from shopping_agent.reranking_agent import reranking_agent
# from shopping_agent.final_output_agent import final_output_agent
# from shopping_agent.debug_state_agent import debug_after_planner_agent


# def _strip_code_fences(text: str) -> str:
#     text = str(text or "").strip()

#     if text.startswith("```"):
#         text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"```$", "", text).strip()

#     return text


# def _parse_json_maybe(value: Any) -> Dict[str, Any]:
#     if isinstance(value, dict):
#         return value

#     if value is None:
#         return {}

#     text = _strip_code_fences(str(value).strip())

#     if not text:
#         return {}

#     try:
#         parsed = json.loads(text)
#         return parsed if isinstance(parsed, dict) else {}
#     except Exception:
#         pass

#     match = re.search(r"\{.*\}", text, flags=re.DOTALL)

#     if not match:
#         return {}

#     try:
#         parsed = json.loads(match.group(0))
#         return parsed if isinstance(parsed, dict) else {}
#     except Exception:
#         return {}


# def _boolish(value: Any) -> bool:
#     if isinstance(value, bool):
#         return value

#     if value is None:
#         return False

#     return str(value).strip().lower() in {"true", "1", "yes", "y"}


# def _event_state_delta(event: Event) -> Dict[str, Any]:
#     actions = getattr(event, "actions", None)

#     if actions is None:
#         return {}

#     state_delta = getattr(actions, "state_delta", None)

#     if not state_delta:
#         return {}

#     return dict(state_delta)


# def _selected_image_option_exists(state: Dict[str, Any]) -> bool:
#     selected = _parse_json_maybe(state.get("selected_image_iteration_option"))

#     if selected.get("selected_image_path"):
#         return True

#     if state.get("selected_image_iteration_image_path"):
#         return True

#     if state.get("active_design_reference_image_path"):
#         return True

#     return False


# def _image_iteration_blocks_browser(state: Dict[str, Any]) -> bool:
#     """
#     Browserbase must not run immediately after A/B/D image generation.

#     Flow:
#       image uploaded / modify requested
#         -> VisualPreferenceAgent generates A/B/D
#         -> Planner asks user to choose A/B/D
#         -> Root pipeline must stop before Browserbase

#     Once the user selects A/B/D, selected_image_iteration_option or
#     active_design_reference_image_path is set, and browsing can proceed
#     only if PlannerAgent produces a valid browser_query.
#     """
#     current_iter = _parse_json_maybe(
#         state.get("current_turn_image_iteration_output")
#     )

#     if (
#         current_iter.get("requested") is True
#         and current_iter.get("ok") is True
#         and current_iter.get("pending_selection") is True
#         and not _selected_image_option_exists(state)
#     ):
#         return True

#     if (
#         _boolish(state.get("image_iteration_pending_selection"))
#         and not _selected_image_option_exists(state)
#     ):
#         return True

#     return False


# def _planner_allows_continue(state: Dict[str, Any]) -> bool:
#     """
#     Hard gate before Browserbase.

#     This is intentionally defensive. Even if the LLM planner accidentally
#     emits browser_query, Browserbase cannot run when A/B/D image options
#     are pending selection.
#     """
#     if _image_iteration_blocks_browser(state):
#         return False

#     if state.get("planner_should_continue") is False:
#         return False

#     planner_output = _parse_json_maybe(state.get("planner_output"))

#     if not planner_output:
#         return False

#     if planner_output.get("task_type") == "needs_clarification":
#         return False

#     if not str(planner_output.get("browser_query") or "").strip():
#         return False

#     confidence = planner_output.get("confidence_score", 0)

#     try:
#         confidence = int(confidence)
#     except Exception:
#         confidence = 0

#     if confidence < 50:
#         return False

#     return True


# class ShoppingRecommendationPipeline(BaseAgent):
#     """
#     Conditional shopping recommendation pipeline.

#     Replaces SequentialAgent because SequentialAgent would keep running
#     downstream agents even after PlannerAgent decided the turn should stop.
#     """

#     def __init__(self) -> None:
#         super().__init__(
#             name="ShoppingRecommendationPipeline",
#             description=(
#                 "Conditional shopping recommendation pipeline. Runs visual extraction "
#                 "and image iteration, then planner, then stops before Browserbase "
#                 "unless the planner produced an actionable browser_query."
#             ),
#         )

#     async def _run_child_and_capture_state(
#         self,
#         child_agent: BaseAgent,
#         ctx: InvocationContext,
#         local_state: Dict[str, Any],
#     ) -> AsyncGenerator[Event, None]:
#         async for event in child_agent.run_async(ctx):
#             delta = _event_state_delta(event)

#             if delta:
#                 local_state.update(delta)

#             yield event

#     @override
#     async def _run_async_impl(
#         self,
#         ctx: InvocationContext,
#     ) -> AsyncGenerator[Event, None]:
#         local_state: Dict[str, Any] = dict(
#             getattr(getattr(ctx, "session", None), "state", {}) or {}
#         )

#         # 1. Visual extraction + optional A/B/D image generation.
#         async for event in self._run_child_and_capture_state(
#             visual_preference_agent,
#             ctx,
#             local_state,
#         ):
#             yield event

#         # 2. Planner decides:
#         #    - ask user to choose A/B/D
#         #    - ask clarification
#         #    - or produce browser_query
#         async for event in self._run_child_and_capture_state(
#             planner_agent,
#             ctx,
#             local_state,
#         ):
#             yield event

#         # 3. Hard stop before Browserbase.
#         if not _planner_allows_continue(local_state):
#             return

#         # 4. Optional debug after planner.
#         async for event in self._run_child_and_capture_state(
#             debug_after_planner_agent,
#             ctx,
#             local_state,
#         ):
#             yield event

#         # 5. Browserbase / web discovery.
#         async for event in self._run_child_and_capture_state(
#             web_discovery_agent,
#             ctx,
#             local_state,
#         ):
#             yield event

#         # 6. Reranking.
#         async for event in self._run_child_and_capture_state(
#             reranking_agent,
#             ctx,
#             local_state,
#         ):
#             yield event

#         # 7. Final response.
#         async for event in self._run_child_and_capture_state(
#             final_output_agent,
#             ctx,
#             local_state,
#         ):
#             yield event


# root_agent = ShoppingRecommendationPipeline()

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict

from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from shopping_agent.visual_preference_agent import visual_preference_agent
from shopping_agent.tools.planner_agent import planner_agent
from shopping_agent.web_discovery_agent import web_discovery_agent
from shopping_agent.reranking_agent import reranking_agent
from shopping_agent.final_output_agent import final_output_agent
from shopping_agent.debug_state_agent import debug_after_planner_agent


def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def _parse_json_maybe(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    text = _strip_code_fences(str(value).strip())

    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _event_state_delta(event: Event) -> Dict[str, Any]:
    actions = getattr(event, "actions", None)

    if actions is None:
        return {}

    state_delta = getattr(actions, "state_delta", None)

    if not state_delta:
        return {}

    return dict(state_delta)


def _selected_image_option_exists(state: Dict[str, Any]) -> bool:
    selected = _parse_json_maybe(state.get("selected_image_iteration_option"))

    return bool(
        selected.get("selected_image_path")
        or state.get("selected_image_iteration_image_path")
        or state.get("active_design_reference_image_path")
    )


def _image_iteration_blocks_browser(state: Dict[str, Any]) -> bool:
    """
    Browserbase must not run while design variants are pending selection.

    Verified edits do not set pending_selection=True, but PlannerAgent still stops
    the turn after an edit so the user can inspect the generated image first.
    """
    current_iter = _parse_json_maybe(
        state.get("current_turn_image_iteration_output")
        or state.get("image_iteration_output")
    )

    if (
        current_iter.get("requested") is True
        and current_iter.get("ok") is True
        and current_iter.get("pending_selection") is True
        and not _selected_image_option_exists(state)
    ):
        return True

    if (
        _boolish(state.get("image_iteration_pending_selection"))
        and not _selected_image_option_exists(state)
    ):
        return True

    return False


def _planner_allows_continue(state: Dict[str, Any]) -> bool:
    """
    Hard gate before Browserbase.

    Even if the LLM accidentally emits browser_query, Browserbase cannot run when
    image options are waiting for selection or when the planner controller stopped.
    """
    if _image_iteration_blocks_browser(state):
        return False

    if state.get("planner_should_continue") is False:
        return False

    planner_output = _parse_json_maybe(state.get("planner_output"))

    if not planner_output:
        return False

    if planner_output.get("task_type") == "needs_clarification":
        return False

    if not str(planner_output.get("browser_query") or "").strip():
        return False

    try:
        confidence = int(planner_output.get("confidence_score", 0))
    except Exception:
        confidence = 0

    return confidence >= 50


class ShoppingRecommendationPipeline(BaseAgent):
    """
    Conditional shopping recommendation pipeline.

    Flow:
      1. VisualPreferenceAgent extracts visual context and may run image iteration.
      2. PlannerAgent decides whether to stop, ask user, or browse.
      3. Browserbase runs only when planner_should_continue=True and browser_query exists.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ShoppingRecommendationPipeline",
            description=(
                "Conditional shopping recommendation pipeline. Runs visual extraction "
                "and verified image iteration, then planner, then stops before "
                "Browserbase unless the planner produced an actionable browser_query."
            ),
        )

    async def _run_child_and_capture_state(
        self,
        child_agent: BaseAgent,
        ctx: InvocationContext,
        local_state: Dict[str, Any],
    ) -> AsyncGenerator[Event, None]:
        async for event in child_agent.run_async(ctx):
            delta = _event_state_delta(event)

            if delta:
                local_state.update(delta)

            yield event

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        local_state: Dict[str, Any] = dict(
            getattr(getattr(ctx, "session", None), "state", {}) or {}
        )

        async for event in self._run_child_and_capture_state(
            visual_preference_agent,
            ctx,
            local_state,
        ):
            yield event

        async for event in self._run_child_and_capture_state(
            planner_agent,
            ctx,
            local_state,
        ):
            yield event

        if not _planner_allows_continue(local_state):
            return

        async for event in self._run_child_and_capture_state(
            debug_after_planner_agent,
            ctx,
            local_state,
        ):
            yield event

        async for event in self._run_child_and_capture_state(
            web_discovery_agent,
            ctx,
            local_state,
        ):
            yield event

        async for event in self._run_child_and_capture_state(
            reranking_agent,
            ctx,
            local_state,
        ):
            yield event

        async for event in self._run_child_and_capture_state(
            final_output_agent,
            ctx,
            local_state,
        ):
            yield event


root_agent = ShoppingRecommendationPipeline()
