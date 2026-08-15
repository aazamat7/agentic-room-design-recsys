# shopping_agent/visual_preference_agent.py
# ============================================================
# VisualPreferenceAgent
#
# Runs before PlannerAgent.
#
# Reads:
#   - image from ADK user content, or session state:
#     input_image_path / image_path / uploaded_image_path
#     input_image_gcs_uri / image_gcs_uri
#     input_image_base64
#
# Writes:
#   - visual_preference_output
#
# This agent makes no Gemini text-generation call.
# It uses Gemini Embedding 2 for image-to-taxonomy preference extraction.
# ============================================================

# shopping_agent/visual_preference_agent.py
# ============================================================
# VisualPreferenceAgent
#
# Runs before PlannerAgent.
#
# Reads:
#   - image from ADK user content, or session state:
#     input_image_path / image_path / uploaded_image_path
#     input_image_gcs_uri / image_gcs_uri
#     input_image_base64
#
# Writes:
#   - visual_preference_output
#   - visual_preference_output_json
#   - current_turn_visual_preference_output
#   - current_turn_visual_preference_output_json
#   - last_visual_preference_output
#   - last_visual_preference_output_json
#
# Important:
#   - This agent is SILENT. It does not emit user-visible text.
#   - If no new image is provided, it reuses the previous successful
#     visual preferences for conversational continuity.
#   - If a new image is provided but extraction fails, it does NOT reuse
#     old visual context as the active visual_preference_output.
# ============================================================

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from shopping_agent.tools.visual_preference_extractor import (
    GeminiEmbedding2VisualPreferenceExtractor,
    empty_visual_preference_output,
    extract_text_from_content,
    find_image_input,
    safe_json_loads,
)


logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> Dict[str, Any]:
    """
    Safely convert state values into dicts.

    ADK state may contain either:
      - a Python dict
      - a JSON string
      - None
    """
    if isinstance(value, dict):
        return value

    parsed = safe_json_loads(value)

    if isinstance(parsed, dict):
        return parsed

    return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _has_successful_visual(value: Any) -> bool:
    data = _as_dict(value)
    return bool(data.get("has_image") is True)


def _reuse_previous_visual(previous_visual: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reuse previous successful visual preferences without repeatedly growing notes.
    """
    output = dict(previous_visual)

    existing_notes = list(output.get("notes") or [])
    reuse_note = (
        "No new image provided; reused previous visual preferences "
        "for conversational continuity."
    )

    if reuse_note not in existing_notes:
        existing_notes.append(reuse_note)

    output["notes"] = existing_notes
    output["reused_from_previous_turn"] = True
    output["image_available_this_turn"] = False
    output["visual_extraction_failed_this_turn"] = False

    return output


class VisualPreferenceAgent(BaseAgent):
    """
    Silent deterministic visual preference extraction agent.

    Behavior:
      1. If a new image is found:
         - run GeminiEmbedding2VisualPreferenceExtractor
         - save output as active visual_preference_output
         - save it as last_visual_preference_output

      2. If no new image is found but previous visual context exists:
         - reuse previous visual context
         - mark reused_from_previous_turn=True

      3. If no new image and no previous visual context:
         - write empty visual output

      4. If new image exists but extraction fails:
         - write empty visual output with error reason
         - do NOT reuse previous visual as active output
           because user likely intended to provide a new visual reference
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        logger.info("[%s] Starting visual preference extraction.", self.name)

        state = getattr(getattr(ctx, "session", None), "state", {}) or {}

        previous_visual = _as_dict(
            state.get("last_visual_preference_output")
            or state.get("visual_preference_output")
        )

        image_input = find_image_input(ctx)
        user_query = extract_text_from_content(getattr(ctx, "user_content", None))

        current_turn_output: Dict[str, Any]
        effective_output: Dict[str, Any]
        last_visual_output: Optional[Dict[str, Any]] = (
            previous_visual if _has_successful_visual(previous_visual) else None
        )

        # ------------------------------------------------------------
        # Case 1: no new image this turn
        # ------------------------------------------------------------
        if image_input is None:
            current_turn_output = empty_visual_preference_output(
                reason="No image input provided this turn."
            )
            current_turn_output["image_available_this_turn"] = False
            current_turn_output["reused_from_previous_turn"] = False
            current_turn_output["visual_extraction_failed_this_turn"] = False

            if _has_successful_visual(previous_visual):
                effective_output = _reuse_previous_visual(previous_visual)
                last_visual_output = previous_visual
            else:
                effective_output = empty_visual_preference_output(
                    reason="No image input provided."
                )
                effective_output["image_available_this_turn"] = False
                effective_output["reused_from_previous_turn"] = False
                effective_output["visual_extraction_failed_this_turn"] = False
                last_visual_output = None

        # ------------------------------------------------------------
        # Case 2: new image provided this turn
        # ------------------------------------------------------------
        else:
            try:
                extractor = GeminiEmbedding2VisualPreferenceExtractor()
                extracted = extractor.extract_preferences(
                    image_input=image_input,
                    user_query=user_query,
                )

                extracted["image_available_this_turn"] = True
                extracted["reused_from_previous_turn"] = False
                extracted["visual_extraction_failed_this_turn"] = False

                current_turn_output = extracted
                effective_output = extracted
                last_visual_output = extracted

            except Exception as exc:
                reason = (
                    f"Visual preference extraction failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                logger.exception("[%s] Visual extraction failed.", self.name)

                failed_output = empty_visual_preference_output(reason=reason)
                failed_output["image_available_this_turn"] = True
                failed_output["reused_from_previous_turn"] = False
                failed_output["visual_extraction_failed_this_turn"] = True

                current_turn_output = failed_output
                effective_output = failed_output

                # Preserve previous successful visual in last_visual_preference_output,
                # but do not use it as active visual_preference_output.
                if _has_successful_visual(previous_visual):
                    last_visual_output = previous_visual
                else:
                    last_visual_output = None

        logger.info(
            "[%s] active_visual_has_image=%s reused=%s failed=%s room=%s styles=%s",
            self.name,
            effective_output.get("has_image"),
            effective_output.get("reused_from_previous_turn"),
            effective_output.get("visual_extraction_failed_this_turn"),
            effective_output.get("room_or_use_case"),
            effective_output.get("styles"),
        )

        state_delta: Dict[str, Any] = {
            "visual_preference_output": effective_output,
            "visual_preference_output_json": _json_dumps(effective_output),
            "current_turn_visual_preference_output": current_turn_output,
            "current_turn_visual_preference_output_json": _json_dumps(
                current_turn_output
            ),
        }

        if last_visual_output is not None:
            state_delta["last_visual_preference_output"] = last_visual_output
            state_delta["last_visual_preference_output_json"] = _json_dumps(
                last_visual_output
            )

        # Silent event: state update only, no content.
        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta=state_delta,
            ),
        )


visual_preference_agent = VisualPreferenceAgent(
    name="VisualPreferenceAgent",
    description=(
        "Silently extracts or reuses visual interior-design preferences from "
        "optional room/product images using Gemini Embedding 2."
    ),
)