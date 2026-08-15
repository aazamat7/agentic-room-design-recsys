# shopping_agent/visual_preference_agent.py
# ============================================================
# VisualPreferenceAgent
#
# Silent multimodal state agent.
#
# Responsibilities:
#   1. Extract visual preferences from optional image input.
#   2. Preserve previous visual context across conversational turns.
#   3. Phase 1: when a user uploads a room image, generate selectable
#      design variants through design_session via visual_iteration_ops.
#   4. Phase 2: when a user asks to edit/iterate the selected design,
#      route the request to self-contained visual_iteration_ops
#      instead of rerunning broad A/B/E generation.
#   5. For multi-reference/product placement, use Gemini scene-level merge
#      through visual_iteration_ops.place_product behind local apply_reliable.
#
# Important:
#   This agent is silent. It yields state_delta only.
# ============================================================

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from PIL import Image
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from shopping_agent.tools import visual_iteration_ops as vio
from shopping_agent.tools.visual_preference_extractor import (
    GeminiEmbedding2VisualPreferenceExtractor,
    ImageInput,
    empty_visual_preference_output,
    extract_text_from_content,
    find_image_input,
    safe_json_loads,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Runtime config
# ============================================================

ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE = (
    os.getenv("ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE", "true").lower()
    in {"1", "true", "yes", "y"}
)

IMAGE_ITERATION_OUTPUT_DIR = Path(
    os.getenv(
        "IMAGE_ITERATION_OUTPUT_DIR",
        "shopping_agent/data/generated/image_iterations",
    )
)


# ============================================================
# 2. User intent detection
# ============================================================

MODIFY_IMAGE_PATTERNS = [
    r"\bmodify\b",
    r"\biterate\b",
    r"\biteration\b",
    r"\bvariation\b",
    r"\bvariations\b",
    r"\bredo\b",
    r"\bremake\b",
    r"\bretry\b",
    r"\bchange\b",
    r"\bedit\b",
    r"\brevise\b",
    r"\bmake it\b",
    r"\bmake this\b",
    r"\bmake the room\b",
    r"\bdecorate\b",
    r"\bfurnish\b",
    r"\bredesign\b",
    r"\brestyle\b",
    r"\brenovate\b",
    r"\badd\b",
    r"\bremove\b",
    r"\breplace\b",
    r"\bswap\b",
    r"\bmove\b",
    r"\brearrange\b",
    r"\brelight\b",
    r"\blighting\b",
    r"\bplace this\b",
    r"\bput this\b",
    r"\bmerge\b",
    r"\bcombine\b",
    r"\bcozier\b",
    r"\bmore cozy\b",
    r"\bpet friendly\b",
    r"\bpet-friendly\b",
    r"\bbrighter\b",
    r"\bdarker\b",
    r"\bwarmer\b",
    r"\bmore minimal\b",
    r"\bmore modern\b",
]


def _is_modify_image_intent(user_query: str) -> bool:
    text = (user_query or "").lower()
    return any(re.search(pattern, text) for pattern in MODIFY_IMAGE_PATTERNS)


def _make_generation_intent(user_query: str) -> str:
    text = (user_query or "").strip()

    if text:
        return text

    return (
        "Furnish this room as a comfortable residential living room with sofa, "
        "coffee table, chairs, lighting, textiles, and home decor. Preserve the "
        "original architecture exactly."
    )


# ============================================================
# 3. Generic helpers
# ============================================================

def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    parsed = safe_json_loads(value)

    if isinstance(parsed, dict):
        return parsed

    return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _has_successful_visual(value: Any) -> bool:
    data = _as_dict(value)
    return bool(data.get("has_image") is True)


def _clean_path(value: Any) -> str:
    text = str(value or "").strip()

    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()

    text = os.path.expandvars(text)
    text = os.path.expanduser(text)
    return text


def _reuse_previous_visual(previous_visual: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(previous_visual)
    notes = list(output.get("notes") or [])
    note = "No new image provided; reused previous visual preferences for conversational continuity."

    if note not in notes:
        notes.append(note)

    output["notes"] = notes
    output["reused_from_previous_turn"] = True
    output["image_available_this_turn"] = False
    output["visual_extraction_failed_this_turn"] = False
    return output


def _save_pil_image(img: Image.Image, run_id: str, name: str) -> str:
    IMAGE_ITERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:160] or "image"
    path = IMAGE_ITERATION_OUTPUT_DIR / f"{run_id}_{safe}.png"
    img.convert("RGB").save(path)
    return str(path)


def _load_pil_from_path(path: str) -> Image.Image:
    return Image.open(_clean_path(path)).convert("RGB")


def _decode_possible_base64(value: Any) -> Optional[bytes]:
    if not value:
        return None

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    text = str(value).strip()

    if not text:
        return None

    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1].strip()

    try:
        return base64.b64decode(text)
    except Exception:
        return None


def _part_inline_bytes(part: Any) -> Optional[bytes]:
    inline_data = getattr(part, "inline_data", None)

    if inline_data is None:
        return None

    data = getattr(inline_data, "data", None)
    return _decode_possible_base64(data)


def _get_image_input_attr(image_input: ImageInput, *names: str) -> Any:
    for name in names:
        value = getattr(image_input, name, None)
        if value:
            return value
    return None


def _image_input_to_pil_and_path(
    image_input: ImageInput,
    run_id: str,
) -> Tuple[Image.Image, Optional[str]]:
    """
    Converts ImageInput to PIL image.

    Local paths are preserved. Inline/base64 images are saved locally because
    image_ops/reliable_ops/design_session expect PIL/local image handling.
    """
    image_path = _get_image_input_attr(
        image_input,
        "image_path",
        "path",
        "local_path",
        "input_image_path",
    )

    if image_path:
        path = _clean_path(image_path)
        return _load_pil_from_path(path), path

    image_bytes = _get_image_input_attr(
        image_input,
        "image_bytes",
        "bytes",
        "data",
    )

    if image_bytes:
        raw = _decode_possible_base64(image_bytes)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    image_base64 = _get_image_input_attr(
        image_input,
        "image_base64",
        "base64",
        "input_image_base64",
    )

    if image_base64:
        raw = _decode_possible_base64(image_base64)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    provided_part = _get_image_input_attr(
        image_input,
        "provided_part",
        "part",
        "image_part",
    )

    if provided_part is not None:
        raw = _part_inline_bytes(provided_part)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    gcs_uri = _get_image_input_attr(
        image_input,
        "gcs_uri",
        "image_gcs_uri",
        "input_image_gcs_uri",
        "uri",
    )

    if gcs_uri:
        raise ValueError(
            "Image iteration requires a local or inline image. "
            f"Got GCS URI: {gcs_uri}. Download it locally or pass image_path / input_image_base64."
        )

    raise ValueError("Could not convert ImageInput to PIL image for image iteration.")


def _empty_image_iteration_output(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "requested": False,
        "mode": "none",
        "options": [],
        "pending_selection": False,
        "notes": [reason],
    }


# ============================================================
# 4. Selection / previous image helpers
# ============================================================

def _get_last_iteration_output(state: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(state.get("last_image_iteration_output") or state.get("image_iteration_output"))


def _get_option_by_letter(iteration_output: Dict[str, Any], option_letter: str) -> Optional[Dict[str, Any]]:
    wanted = option_letter.upper()

    for option in iteration_output.get("options", []) or []:
        if str(option.get("option_id", "")).upper() == wanted:
            return option

    return None


def _detect_selected_option(user_query: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detects user selection of A/B/E without treating article "a" as option A.
    """
    raw = (user_query or "").strip()
    lower = raw.lower()
    option_letter: Optional[str] = None

    full = re.fullmatch(r"(?:option\s+|pipeline\s+)?([abe])(?:\s+please)?[.!]?", lower)

    if full:
        option_letter = full.group(1).upper()

    if option_letter is None:
        explicit = re.search(
            r"\b(?:choose|pick|select|use|go\s+with)\s+(?:option\s+|pipeline\s+)?([abe])\b",
            lower,
        )
        if explicit:
            option_letter = explicit.group(1).upper()

    if option_letter is None:
        explicit_option = re.search(r"\b(?:option|pipeline)\s+([abe])\b", lower)
        if explicit_option:
            option_letter = explicit_option.group(1).upper()

    if option_letter is None:
        uppercase = re.search(r"\b([ABE])\b", raw)
        if uppercase:
            option_letter = uppercase.group(1).upper()

    if option_letter is None:
        return None

    last_iter = _get_last_iteration_output(state)
    option = _get_option_by_letter(last_iter, option_letter)

    if not option:
        return None

    return {
        "selected_option_id": option_letter,
        "selected_pipeline": option.get("pipeline"),
        "selected_image_path": option.get("image_path"),
        "selected_option": option,
    }


def _resolve_previous_iteration_base_image(
    user_query: str,
    state: Dict[str, Any],
) -> Tuple[Optional[Image.Image], Optional[str], Optional[Dict[str, Any]]]:
    """
    Base image priority for edits:
      1. Explicit selected option A/B/E in the current user text.
      2. Last selected/generated active design.
      3. Last image-iteration source.
      4. Last successful visual input path.
    """
    selected = _detect_selected_option(user_query, state)

    if selected and selected.get("selected_image_path"):
        path = str(selected["selected_image_path"])
        if Path(path).exists():
            return _load_pil_from_path(path), path, selected

    for key in (
        "selected_image_iteration_image_path",
        "active_design_reference_image_path",
    ):
        selected_path = state.get(key)
        if selected_path and Path(str(selected_path)).exists():
            path = str(selected_path)
            return _load_pil_from_path(path), path, None

    last_iter = _get_last_iteration_output(state)
    for key in ("selected_output_image_path", "source_image_path"):
        source_path = last_iter.get(key)
        if source_path and Path(str(source_path)).exists():
            path = str(source_path)
            return _load_pil_from_path(path), path, None

    last_input_path = state.get("last_image_iteration_input_path")
    if last_input_path and Path(str(last_input_path)).exists():
        path = str(last_input_path)
        return _load_pil_from_path(path), path, None

    previous_visual = _as_dict(state.get("last_visual_preference_output") or state.get("visual_preference_output"))
    visual_path = previous_visual.get("input_image_local_path")

    if visual_path and Path(str(visual_path)).exists():
        path = str(visual_path)
        return _load_pil_from_path(path), path, None

    return None, None, selected


# ============================================================
# 5. Visual preference extraction runner
# ============================================================

async def _extract_visual_preferences_async(
    image_input: ImageInput,
    user_query: str,
) -> Dict[str, Any]:
    def _run() -> Dict[str, Any]:
        extractor = GeminiEmbedding2VisualPreferenceExtractor()
        return extractor.extract_preferences(
            image_input=image_input,
            user_query=user_query,
        )

    return await asyncio.to_thread(_run)


# ============================================================
# 6. VisualPreferenceAgent
# ============================================================

class VisualPreferenceAgent(BaseAgent):
    """
    Silent visual context + image iteration agent.

    It writes:
      - visual_preference_output for PlannerAgent
      - image_iteration_output for design variants or reliable edits
      - selected_image_iteration_* state when user chooses a variant or an edit completes
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        logger.info("[%s] Starting visual preference / image iteration.", self.name)

        state = dict(getattr(getattr(ctx, "session", None), "state", {}) or {})
        previous_visual = _as_dict(state.get("last_visual_preference_output") or state.get("visual_preference_output"))

        image_input = find_image_input(ctx)
        user_query = extract_text_from_content(getattr(ctx, "user_content", None))
        user_intent = _make_generation_intent(user_query)

        current_turn_output: Dict[str, Any]
        effective_output: Dict[str, Any]
        last_visual_output: Optional[Dict[str, Any]] = previous_visual if _has_successful_visual(previous_visual) else None

        image_iteration_requested_this_turn = False
        image_iteration_output = _empty_image_iteration_output("No image iteration requested this turn.")
        selected_iteration_option = _detect_selected_option(user_query, state)

        source_image_for_generation: Optional[Image.Image] = None
        source_image_path: Optional[str] = None
        image_generation_trigger: Optional[str] = None

        # ------------------------------------------------------------
        # Case 1: new image provided this turn.
        # Usually this starts Phase 1 variant generation. If the user is
        # explicitly merging/placing a reference product into the previous
        # design, the new image is treated as product_image instead.
        # ------------------------------------------------------------
        if image_input is not None:
            conversion_error: Optional[str] = None
            uploaded_pil: Optional[Image.Image] = None
            uploaded_path: Optional[str] = None

            try:
                uploaded_pil, uploaded_path = _image_input_to_pil_and_path(
                    image_input=image_input,
                    run_id=uuid.uuid4().hex[:12],
                )
                source_image_for_generation = uploaded_pil
                source_image_path = uploaded_path
            except Exception as exc:  # noqa: BLE001
                conversion_error = (
                    "Image was usable for visual extraction but not for image iteration: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.warning(conversion_error)

            extraction_task = asyncio.create_task(
                _extract_visual_preferences_async(
                    image_input=image_input,
                    user_query=user_query,
                )
            )

            image_iteration_task: Optional[asyncio.Task[Dict[str, Any]]] = None

            is_merge_with_reference = (
                uploaded_pil is not None
                and _is_modify_image_intent(user_query)
                and vio.is_merge_or_product_placement(user_query)
            )

            if is_merge_with_reference:
                base_img, base_path, selected_for_edit = _resolve_previous_iteration_base_image(user_query, state)
                if base_img is not None:
                    selected_iteration_option = selected_iteration_option or selected_for_edit
                    source_image_for_generation = base_img
                    source_image_path = base_path
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "merge_reference_image"
                    image_iteration_task = asyncio.create_task(
                        asyncio.to_thread(
                            vio.run_reliable_edit,
                            base_img,
                            base_path,
                            user_query,
                            uploaded_pil,
                        )
                    )
                else:
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "merge_reference_image_missing_base"
                    image_iteration_output = {
                        "ok": False,
                        "requested": True,
                        "mode": "reliable_edit",
                        "pending_selection": False,
                        "options": [],
                        "errors": [
                            "The user uploaded a reference/product image to merge, but no active room/design image was available."
                        ],
                        "notes": [
                            "Ask the user to first upload a room image or choose a generated design option."
                        ],
                    }

            elif ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE and source_image_for_generation is not None:
                image_iteration_requested_this_turn = True
                image_generation_trigger = "new_image"
                image_iteration_task = asyncio.create_task(
                    asyncio.to_thread(
                        vio.run_design_variants,
                        source_image_for_generation,
                        source_image_path,
                        user_intent,
                        image_generation_trigger,
                    )
                )

            try:
                extracted = await extraction_task
                extracted["image_available_this_turn"] = True
                extracted["reused_from_previous_turn"] = False
                extracted["visual_extraction_failed_this_turn"] = False

                if uploaded_path:
                    extracted["input_image_local_path"] = uploaded_path

                if conversion_error:
                    extracted.setdefault("notes", []).append(conversion_error)

                if is_merge_with_reference:
                    extracted.setdefault("notes", []).append(
                        "This uploaded image was interpreted as a reference/product image for an edit."
                    )

                current_turn_output = extracted
                effective_output = extracted
                last_visual_output = extracted

            except Exception as exc:  # noqa: BLE001
                reason = f"Visual preference extraction failed: {type(exc).__name__}: {exc}"
                logger.exception("[%s] Visual extraction failed.", self.name)

                failed_output = empty_visual_preference_output(reason=reason)
                failed_output["image_available_this_turn"] = True
                failed_output["reused_from_previous_turn"] = False
                failed_output["visual_extraction_failed_this_turn"] = True

                if uploaded_path:
                    failed_output["input_image_local_path"] = uploaded_path

                if conversion_error:
                    failed_output.setdefault("notes", []).append(conversion_error)

                current_turn_output = failed_output
                effective_output = failed_output
                last_visual_output = previous_visual if _has_successful_visual(previous_visual) else None

            if image_iteration_task is not None:
                image_iteration_output = await image_iteration_task

        # ------------------------------------------------------------
        # Case 2: no new image this turn.
        # This is where selected-option edits become reliable_ops edits.
        # ------------------------------------------------------------
        else:
            current_turn_output = empty_visual_preference_output(reason="No image input provided this turn.")
            current_turn_output["image_available_this_turn"] = False
            current_turn_output["reused_from_previous_turn"] = False
            current_turn_output["visual_extraction_failed_this_turn"] = False

            if _has_successful_visual(previous_visual):
                effective_output = _reuse_previous_visual(previous_visual)
                last_visual_output = previous_visual
            else:
                effective_output = empty_visual_preference_output(reason="No image input provided.")
                effective_output["image_available_this_turn"] = False
                effective_output["reused_from_previous_turn"] = False
                effective_output["visual_extraction_failed_this_turn"] = False
                last_visual_output = None

            if _is_modify_image_intent(user_query):
                prev_img, prev_path, selected_for_edit = _resolve_previous_iteration_base_image(user_query, state)

                if prev_img is not None:
                    selected_iteration_option = selected_iteration_option or selected_for_edit
                    source_image_for_generation = prev_img
                    source_image_path = prev_path
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "modify_selected_option" if selected_for_edit else "modify_previous_image"

                    image_iteration_output = await asyncio.to_thread(
                        vio.run_reliable_edit,
                        source_image_for_generation,
                        source_image_path,
                        user_query,
                        None,
                    )
                else:
                    image_iteration_output = {
                        "ok": False,
                        "requested": True,
                        "mode": "reliable_edit",
                        "pending_selection": False,
                        "options": [],
                        "errors": [
                            "User asked to modify/iterate an image, but no previous local image or generated option was available."
                        ],
                        "notes": ["Ask the user to upload an image or choose an existing option first."],
                    }

        edited_path = image_iteration_output.get("selected_output_image_path")
        if edited_path:
            selected_iteration_option = {
                "selected_option_id": "EDIT",
                "selected_pipeline": image_iteration_output.get("mode"),
                "selected_image_path": edited_path,
                "selected_option": (image_iteration_output.get("options") or [{}])[0],
            }

        logger.info(
            "[%s] visual_has_image=%s reused=%s image_iter_requested=%s mode=%s options=%s selected=%s",
            self.name,
            effective_output.get("has_image"),
            effective_output.get("reused_from_previous_turn"),
            image_iteration_requested_this_turn,
            image_iteration_output.get("mode"),
            len(image_iteration_output.get("options", []) or []),
            bool(selected_iteration_option),
        )

        # ------------------------------------------------------------
        # State update
        # ------------------------------------------------------------
        state_delta: Dict[str, Any] = {
            "visual_preference_output": effective_output,
            "visual_preference_output_json": _json_dumps(effective_output),
            "current_turn_visual_preference_output": current_turn_output,
            "current_turn_visual_preference_output_json": _json_dumps(current_turn_output),
            "image_iteration_requested_this_turn": image_iteration_requested_this_turn,
            "current_turn_image_iteration_output": image_iteration_output,
            "current_turn_image_iteration_output_json": _json_dumps(image_iteration_output),
            "selected_image_iteration_option": selected_iteration_option or {},
            "selected_image_iteration_option_json": _json_dumps(selected_iteration_option or {}),
        }

        if last_visual_output is not None:
            state_delta["last_visual_preference_output"] = last_visual_output
            state_delta["last_visual_preference_output_json"] = _json_dumps(last_visual_output)

        if image_iteration_output.get("requested") is True:
            state_delta["image_iteration_output"] = image_iteration_output
            state_delta["image_iteration_output_json"] = _json_dumps(image_iteration_output)

            if image_iteration_output.get("ok") is True:
                state_delta["last_image_iteration_output"] = image_iteration_output
                state_delta["last_image_iteration_output_json"] = _json_dumps(image_iteration_output)
                state_delta["image_iteration_pending_selection"] = bool(image_iteration_output.get("pending_selection"))

                if source_image_path:
                    state_delta["last_image_iteration_input_path"] = source_image_path

                if edited_path:
                    state_delta["selected_image_iteration_image_path"] = edited_path
                    state_delta["active_design_reference_image_path"] = edited_path
                    state_delta["last_reliable_image_edit_output"] = image_iteration_output
                    state_delta["last_reliable_image_edit_output_json"] = _json_dumps(image_iteration_output)
                    state_delta["image_iteration_pending_selection"] = False

        if selected_iteration_option:
            selected_path = selected_iteration_option.get("selected_image_path")
            selected_pipeline = selected_iteration_option.get("selected_pipeline")

            if selected_path:
                state_delta["selected_image_iteration_image_path"] = selected_path
                state_delta["selected_image_iteration_pipeline"] = selected_pipeline
                state_delta["active_design_reference_image_path"] = selected_path
                state_delta["image_iteration_pending_selection"] = False

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


visual_preference_agent = VisualPreferenceAgent(
    name="VisualPreferenceAgent",
    description=(
        "Silently extracts/reuses visual preferences, generates initial design variants, "
        "and applies verified Imagen/Gemini image edits through self-contained visual_iteration_ops."
    ),
)
