from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Tuple

from dotenv import load_dotenv
from typing_extensions import override

load_dotenv()

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GEMINI_TEXT_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))

if not PROJECT_ID:
    raise RuntimeError(
        "Missing GOOGLE_CLOUD_PROJECT. "
        "Set it in .env or export it before running ADK."
    )

# Gemini-only build: keep Vertex Gemini calls on the explicit text location.
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

from google.adk.agents import BaseAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.agents.llm_agent import LlmAgent  # noqa: E402
from google.adk.events import Event, EventActions  # noqa: E402
from google.genai import types  # noqa: E402


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


# ============================================================
# 1. Parsing helpers
# ============================================================

def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
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
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _session_state(ctx: InvocationContext) -> Dict[str, Any]:
    return dict(getattr(getattr(ctx, "session", None), "state", {}) or {})


# ============================================================
# 2. Image iteration state helpers
# ============================================================

def _selected_image_option_exists(state: Dict[str, Any]) -> bool:
    selected = _parse_json_maybe(state.get("selected_image_iteration_option"))

    if selected.get("selected_image_path"):
        return True

    if state.get("selected_image_iteration_image_path"):
        return True

    if state.get("active_design_reference_image_path"):
        return True

    return False


def _image_iteration_needs_selection(state: Dict[str, Any]) -> bool:
    current_iter = _parse_json_maybe(
        state.get("current_turn_image_iteration_output")
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


def _image_iteration_failed_this_turn(state: Dict[str, Any]) -> bool:
    current_iter = _parse_json_maybe(
        state.get("current_turn_image_iteration_output")
    )

    return (
        current_iter.get("requested") is True
        and current_iter.get("ok") is False
    )


def _current_image_iteration_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the image-iteration output produced in this ADK turn.

    Do not fall back to image_iteration_output or last_reliable_image_edit_output here.
    Those are persistent cross-turn context keys. Treating them as current-turn
    results causes stale responses like repeating the previous "remove chair"
    output when the user asks for a new edit such as "make the bench brown".
    """
    return _parse_json_maybe(state.get("current_turn_image_iteration_output"))


def _candidate_image_iteration_outputs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return image-iteration outputs from most-current to oldest known state."""
    candidates: List[Dict[str, Any]] = []

    for key in (
        "current_turn_image_iteration_output",
        "last_reliable_image_edit_output",
        "image_iteration_output",
    ):
        parsed = _parse_json_maybe(state.get(key))
        if parsed:
            candidates.append(parsed)

    return candidates


def _latest_reliable_edit_output(state: Dict[str, Any]) -> Dict[str, Any]:
    for output in _candidate_image_iteration_outputs(state):
        if output.get("mode") == "reliable_edit":
            return output
    return {}


def _current_successful_reliable_edit_output(state: Dict[str, Any]) -> Dict[str, Any]:
    output = _current_image_iteration_output(state)
    if output.get("mode") == "reliable_edit" and output.get("ok") is True:
        return output
    return {}


def _current_failed_reliable_edit_output(state: Dict[str, Any]) -> Dict[str, Any]:
    output = _current_image_iteration_output(state)
    if output.get("mode") == "reliable_edit" and output.get("ok") is False:
        return output
    return {}


_ASYNC_WAIT_RE = re.compile(
    r"\b(still working|in progress|being generated|generating that|generating the|"
    r"i['’]?ll let you know|let you know as soon|as soon as it['’]?s ready|"
    r"come back|check back|wait|processing)\b",
    flags=re.IGNORECASE,
)


def _planner_output_claims_background_work(plan: Dict[str, Any]) -> bool:
    response = str(plan.get("response_to_user") or "")
    reasoning = str(plan.get("confidence_reasoning") or "")
    return bool(_ASYNC_WAIT_RE.search(response) or _ASYNC_WAIT_RE.search(reasoning))


def _render_missing_edit_output_message(state: Dict[str, Any]) -> str:
    active_path = str(state.get("active_design_reference_image_path") or "").strip()

    if active_path:
        return (
            "I do not see a completed Gemini edit output for this turn. "
            "This is not a background job, so I will not say it is still running. "
            f"The current active design image is still: {active_path}. "
            "Please retry the edit on this active design image, or choose A, B, or E again if you want to reset the design reference."
        )

    return (
        "I do not see a completed Gemini edit output for this turn. "
        "This is not a background job, so I will not say it is still running. "
        "Please upload an image or select A, B, or E before asking for another image edit."
    )


def _forced_missing_edit_output_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_type": "needs_clarification",
        "confidence_score": 30,
        "confidence_reasoning": (
            "The planner LLM attempted to describe an asynchronous image generation state, "
            "but image editing must complete synchronously in VisualPreferenceAgent before PlannerAgent runs."
        ),
        "response_to_user": _render_missing_edit_output_message(state),
        "input_modalities_used": ["text", "image"] if _selected_image_option_exists(state) else ["text"],
        "clarifying_questions": [],
        "partial_understanding": {
            "what_user_said": "The user referred to an image edit or updated image.",
            "what_we_inferred": (
                "No completed Gemini edit output was present in planner state for this turn; "
                "the planner must not claim background work is happening."
            ),
        },
    }


def _sanitize_planner_output(plan: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove impossible async/background claims from LLM planner output.

    The ADK turn is synchronous from the user's perspective: VisualPreferenceAgent either
    produced an image-iteration output before PlannerAgent runs, or it did not. The
    planner must never say "I'm still working" or "I'll let you know" because there
    is no background job created here.
    """
    if not plan:
        return plan

    if not _planner_output_claims_background_work(plan):
        return plan

    successful_edit = _current_successful_reliable_edit_output(state)
    if successful_edit:
        return _forced_reliable_edit_completed_plan_for_output(successful_edit)

    failed_edit = _current_failed_reliable_edit_output(state)
    if failed_edit:
        return _forced_image_iteration_failure_plan_for_output(failed_edit)

    return _forced_missing_edit_output_plan(state)


def _render_image_iteration_options_message(iteration_output: Dict[str, Any]) -> str:
    options = iteration_output.get("options") or []
    ok_options = [opt for opt in options if opt.get("ok")]

    lines = [
        "I generated three design directions from your image:",
        "",
    ]

    labels = {
        "A": "A — naive preservation baseline",
        "B": "B — vision-described baseline",
        "E": "E — Gemini Embedding 2 facet-diverse fanout",
    }

    seen = set()

    for opt in ok_options:
        option_id = str(opt.get("option_id") or "").upper()

        if option_id in labels and option_id not in seen:
            seconds = opt.get("sec")
            suffix = f" ({seconds}s)" if seconds is not None else ""
            lines.append(f"{labels[option_id]}{suffix}")
            seen.add(option_id)

    for option_id in ["A", "B", "E"]:
        if option_id not in seen:
            failed = next(
                (
                    opt for opt in options
                    if str(opt.get("option_id") or "").upper() == option_id
                ),
                None,
            )

            if failed:
                lines.append(f"{labels[option_id]} — failed")
            else:
                lines.append(labels[option_id])

    lines.extend(
        [
            "",
            "Choose A, B, or E. After you choose, I can either browse matching products for that version or iterate on it further.",
        ]
    )

    return "\n".join(lines)


def _render_image_iteration_failure_message(iteration_output: Dict[str, Any]) -> str:
    errors = iteration_output.get("errors") or []

    if errors:
        return (
            "I tried to run the Gemini visual-iteration step, but it failed. "
            "The main issue was:\n"
            f"{errors[0]}\n\n"
            "Try uploading the image again, or ask for a simpler modification."
        )

    return (
        "I tried to run the Gemini visual-iteration step, but it failed. "
        "Try uploading the image again, or ask for a simpler modification."
    )


def _forced_image_iteration_selection_plan(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    iteration_output = _current_image_iteration_output(state)

    return {
        "task_type": "needs_clarification",
        "confidence_score": 25,
        "confidence_reasoning": (
            "A/B/E image design options were generated this turn, so the user "
            "must choose one before Browserbase product retrieval can run."
        ),
        "response_to_user": _render_image_iteration_options_message(
            iteration_output
        ),
        "input_modalities_used": ["text", "image"],
        "clarifying_questions": [],
        "partial_understanding": {
            "what_user_said": "The user provided or modified an image.",
            "what_we_inferred": (
                "The system generated design options A, B, and E and is waiting "
                "for the user to select one."
            ),
        },
    }


def _forced_image_iteration_failure_plan(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    return _forced_image_iteration_failure_plan_for_output(
        _current_image_iteration_output(state)
    )


def _forced_image_iteration_failure_plan_for_output(
    iteration_output: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task_type": "needs_clarification",
        "confidence_score": 20,
        "confidence_reasoning": (
            "The user requested image iteration, but the Gemini image-generation/editing step failed."
        ),
        "response_to_user": _render_image_iteration_failure_message(
            iteration_output
        ),
        "input_modalities_used": ["text", "image"],
        "clarifying_questions": [],
        "partial_understanding": {
            "what_user_said": "The user provided or modified an image.",
            "what_we_inferred": "Gemini visual iteration was attempted but failed.",
        },
    }


def _image_reliable_edit_completed_this_turn(state: Dict[str, Any]) -> bool:
    output = _current_image_iteration_output(state)
    return (
        output.get("requested") is True
        and output.get("mode") == "reliable_edit"
        and output.get("ok") is True
        and output.get("pending_selection") is not True
    )


def _render_reliable_edit_message(iteration_output: Dict[str, Any]) -> str:
    options = iteration_output.get("options") or []
    option = options[0] if options else {}
    edit_plan = option.get("edit_plan") or {}
    op = edit_plan.get("op") or "image edit"
    verified = bool(option.get("verified"))
    score = option.get("score")
    out_path = iteration_output.get("selected_output_image_path") or option.get("image_path")

    if verified:
        line = f"Done — I applied the Gemini {op} visual edit and verified it."
    else:
        line = (
            f"I generated an updated image with Gemini for the {op} edit, but the verifier did not fully confirm the change. "
            "Treat it as a best-effort result."
        )

    if score is not None:
        line += f" Verification score: {score}/10."

    if out_path:
        line += f" Active design image: {out_path}"

    line += " Do you want me to keep iterating on this image or browse matching products for it?"
    return line


def _forced_reliable_edit_completed_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    iteration_output = _current_image_iteration_output(state)
    return _forced_reliable_edit_completed_plan_for_output(iteration_output)


def _forced_reliable_edit_completed_plan_for_output(iteration_output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_type": "needs_clarification",
        "confidence_score": 35,
        "confidence_reasoning": (
            "A Gemini-native verified/best-effort visual edit was produced. The pipeline should stop "
            "so the user can inspect it before Browserbase product retrieval runs."
        ),
        "response_to_user": _render_reliable_edit_message(iteration_output),
        "input_modalities_used": ["text", "image"],
        "clarifying_questions": [],
        "partial_understanding": {
            "what_user_said": "The user asked to iterate on the active design image.",
            "what_we_inferred": "The active design reference was updated with the generated edit output.",
        },
    }


# ============================================================
# 3. Stop / continue logic
# ============================================================

def _planner_should_stop(plan: Dict[str, Any]) -> Tuple[bool, str]:
    if not plan:
        return True, "Planner did not produce parseable JSON."

    task_type = str(plan.get("task_type", "")).strip()
    confidence_score = _safe_int(plan.get("confidence_score"), default=0)
    browser_query = str(plan.get("browser_query") or "").strip()

    if task_type == "needs_clarification":
        return True, (
            "Planner judged that the request needs clarification, direct "
            "conversation handling, or should not trigger browsing."
        )

    if confidence_score < 50:
        return True, f"Planner confidence is below threshold: {confidence_score}."

    if not browser_query:
        return True, "Planner did not produce browser_query."

    return False, ""


def _render_stop_message(plan: Dict[str, Any], stop_reason: str) -> str:
    response_to_user = str(plan.get("response_to_user") or "").strip()

    if response_to_user:
        return response_to_user

    questions = plan.get("clarifying_questions") or []
    clean_questions: List[str] = [
        str(q).strip()
        for q in questions
        if str(q).strip()
    ]

    if clean_questions:
        lines = ["I need a bit more information before I start browsing:"]
        for idx, question in enumerate(clean_questions, start=1):
            lines.append(f"{idx}. {question}")
        return "\n".join(lines)

    confidence_reasoning = str(plan.get("confidence_reasoning") or "").strip()

    if confidence_reasoning:
        return confidence_reasoning

    return stop_reason or "I need a bit more information before I can continue."


# ============================================================
# 4. Planner instruction
# ============================================================

PLANNER_INSTRUCTION = """
You are the Planner Agent for a personalized ecommerce recommendation system
focused on US home furniture and home decor. Scope is US-only.

You are both:
1. A conversational front door for the shopping agent.
2. A strict planning module when the user has an actionable shopping request.

You must return strict JSON only.

# Runtime State From Previous Agents / Previous Turns

visual_preference_output:
{visual_preference_output_json?}

last_visual_preference_output:
{last_visual_preference_output_json?}

last_actionable_planner_output:
{last_actionable_planner_output_json?}

current_turn_image_iteration_output:
{current_turn_image_iteration_output_json?}

image_iteration_output:
{image_iteration_output_json?}

image_iteration_pending_selection:
{image_iteration_pending_selection?}

selected_image_iteration_option:
{selected_image_iteration_option_json?}

active_design_reference_image_path:
{active_design_reference_image_path?}

pipeline_status:
{pipeline_status?}

planner_stop_reason:
{planner_stop_reason?}

last_reliable_image_edit_output:
{last_reliable_image_edit_output_json?}

# Image Iteration Handling

The VisualPreferenceAgent may generate image design options before browsing.
All image editing and iterative refinement are Gemini-native instruction-based edits, not Imagen semantic-mask edits.
It can produce:
- A = naive preservation baseline
- B = vision-described baseline
- E = facet-diverse fanout, if available

If current_turn_image_iteration_output.requested is true
AND current_turn_image_iteration_output.ok is true:
- Do NOT create browser_query.
- Do NOT start Browserbase.
- Return FORMAT B.
- confidence_score between 20 and 35.
- response_to_user should say that options A, B, and E were generated.
- Ask the user to choose A, B, or E depending on which options exist.
- Mention:
  A = naive preservation baseline
  B = vision-described baseline
  E = Gemini Embedding 2 facet fan-out

If image_iteration_pending_selection is true and selected_image_iteration_option is empty:
- Do NOT create browser_query.
- Return FORMAT B asking the user to choose A, B, or E.

If selected_image_iteration_option.selected_image_path exists:
- Treat that image as the active design reference.
- If the user only selected the option, return FORMAT B asking:
  "Do you want me to browse matching products for this version or keep iterating?"
- If the user selected an option AND explicitly asks to browse/shop/find products,
  then return FORMAT A.
- In FORMAT A, include a design_reference object:
  {
    "selected_option_id": "...",
    "selected_pipeline": "...",
    "image_path": "..."
  }

Examples:
User: "choose E"
Return FORMAT B:
{
  "task_type": "needs_clarification",
  "confidence_score": 35,
  "confidence_reasoning": "The user selected option E but did not yet ask to browse products or iterate further.",
  "response_to_user": "Great — I’ll use option E as the design reference. Do you want me to browse matching products for this version, or keep iterating on the image?",
  "input_modalities_used": ["text", "image"],
  "clarifying_questions": [],
  "partial_understanding": {
    "what_user_said": "The user selected option E.",
    "what_we_inferred": "Option E should be treated as the active design reference."
  }
}

User: "choose E and browse matching products under $1000"
Return FORMAT A with browser_query.

# Verified Gemini Image Edit Handling

If current_turn_image_iteration_output.mode is "reliable_edit" and ok is true, the edit came from Gemini image editing in THIS TURN:
- Do NOT create browser_query in the same turn.
- Return FORMAT B.
- Tell the user the Gemini-edited image is now the active design reference.
- If verified is true, say it was verified.
- If verified is false, say it was a best-effort edit and should be inspected.
- Ask whether to keep iterating or browse matching products.

Critical state rule:
- current_turn_image_iteration_output is the only key that may be treated as a newly completed edit.
- last_reliable_image_edit_output and image_iteration_output are historical/context keys only.
- Never repeat an old reliable edit response from historical keys when the user asks for a new edit.
- Example: after "E, remove the chair", if the next user says "make the bench brown", do not repeat the remove-chair output unless current_turn_image_iteration_output contains a new remove-chair result from this same turn.

# Visual Context Interpretation

- visual_preference_output is the active visual context for this turn.
- If visual_preference_output.has_image is true, image context is available.
- If visual_preference_output.reused_from_previous_turn is true, it came from a previous image.
- Reused visual context is valid for follow-ups like:
  "make it cheaper", "make it pet friendly", "similar but smaller",
  "for the same room", "more like this", "continue".
- Do NOT let reused image context turn a generic greeting like "hello"
  into a shopping search.
- current text constraints override image-derived preferences.
- last_actionable_planner_output helps resolve follow-ups like:
  "make it cheaper", "add a rug", "show more", "continue", "same room".

# Conversation / Meta / Status Handling

Critical: Never claim background/asynchronous work. Do not say "I am still working",
"I will let you know", "it is in progress", or "come back later". This system does
not create background image jobs. If a Gemini visual edit completed, use the completed
image state. If no completed edit state exists, say that no completed edit output is present.

If the user asks a conversational or status question such as:
- "hello"
- "hi"
- "hwllo"
- "thanks"
- "ok"
- "is it done"
- "did it finish"
- "are the results ready"
- "what happened"
- "why did it fail"
- "what now"

Do NOT create browser_query.
Do NOT start a new product search.

Return FORMAT B with:
- task_type = "needs_clarification"
- confidence_score between 20 and 35
- clarifying_questions = []
- response_to_user = a natural conversational response
- confidence_reasoning = short explanation

# Critical Reference Word Check

Before producing a normal search plan, scan the user text for these reference words:
- "matching"
- "match"
- "similar"
- "like this"
- "like these"
- "goes with"
- "complements"
- "coordinates with"
- "same as"
- "more of these"
- "to match my"
- "fits my"
- "same room"
- "this room"

IF the user text contains any of these words AND there is no active or previous visual context
AND the user does not describe what to match in text:
  - Return FORMAT B.
  - confidence_score must be between 40 and 49.
  - Do NOT produce browser_query.
  - Ask the user to upload a photo or describe what to match.

EXCEPTION:
If the user explicitly describes what to match in the same text,
such as "matching modern beige furniture", proceed with FORMAT A.

# Follow-up Handling

If the user gives a follow-up like:
- "make it cheaper"
- "make it pet friendly"
- "add a rug"
- "remove leather"
- "same room but under $500"
- "continue"
- "show more like that"

Then:
- Use last_actionable_planner_output if available.
- Use visual_preference_output / last_visual_preference_output if available.
- Use active_design_reference_image_path if a generated image option was selected.
- If enough context exists, produce FORMAT A with an updated browser_query.
- If not enough context exists, produce FORMAT B.

# Main Task

Your output will be passed to Browserbase only when you produce FORMAT A.

Your job in FORMAT A:
Create a broad, high-recall browser_query that retrieves many possible ecommerce
product candidates from US home furniture and decor retailers.

The browser_query is NOT the final recommendation query.
The browser_query is only for candidate retrieval.
Downstream reranking handles precision.

# Scope

- Country: US only.
- Currency: USD only.
- Product categories: home furniture and home decor only.
- Non-US requests -> FORMAT B.
- Out-of-scope products -> FORMAT B.

# Output Formats

FORMAT A — Normal actionable plan.
Use only when confidence_score >= 50 and browser_query exists.

{
  "interpreted_need": "string",
  "task_type": "single_product_search | bundle_recommendation | similar_product_search | gift_recommendation | style_based_recommendation",
  "confidence_score": 50-100,
  "confidence_reasoning": "string explaining why this score",
  "input_modalities_used": ["text", "image"] | ["text"] | ["image"],
  "design_reference": {
    "selected_option_id": null,
    "selected_pipeline": null,
    "image_path": null
  },
  "user_profile": {
    "styles": [],
    "colors": [],
    "materials": [],
    "brands": [],
    "avoid": [],
    "room_or_use_case": ""
  },
  "constraints": {
    "country": "US",
    "currency": "USD",
    "total_budget": null,
    "required_categories": [],
    "must_have": [],
    "nice_to_have": []
  },
  "browser_query": "broad US ecommerce retrieval query",
  "search_strategy": "web_only"
}

FORMAT B — Clarification, direct answer, status response, or out-of-scope.
Use when confidence_score < 50 or there is no actionable shopping search.

{
  "task_type": "needs_clarification",
  "confidence_score": 0-49,
  "confidence_reasoning": "string explaining what information is missing, why out of scope, or why no browsing should run",
  "response_to_user": "natural user-facing response",
  "input_modalities_used": ["text", "image"] | ["text"] | ["image"],
  "clarifying_questions": [],
  "partial_understanding": {
    "what_user_said": "summary of what user provided",
    "what_we_inferred": "what can be reasonably inferred from text, image, and prior context"
  }
}

# Planning Rules

1. FORMAT A must always include browser_query.
2. FORMAT B must never include browser_query.
3. If confidence_score < 50, use FORMAT B.
4. Do not invent a budget.
5. If user says under $1000, set total_budget = 1000.
6. Country is always "US".
7. Currency is always "USD".
8. Required categories should only include explicit product categories from the user.
9. Image-derived product_clues should usually go into nice_to_have, not required_categories.
10. Text constraints override image-derived preferences.
11. If visual context is available, use image-derived style/color/material/room to improve the plan.
12. If visual context was reused from a previous turn, use it only when the user appears to refer to prior context.
13. browser_query must maximize recall.
14. browser_query should include:
    - room/use case when available
    - product category terms
    - style/color/material terms when available
    - budget when available
    - synonyms and related product types
15. Do not make browser_query overly narrow.
16. Do not use abstract phrases that ecommerce sites may not index well.
17. Return strict JSON only.
18. No markdown.
19. No comments.
"""


# ============================================================
# 5. Inner LLM planner
# ============================================================

_planner_llm = LlmAgent(
    name="PlannerLLM",
    model=MODEL,
    description=(
        "Creates a structured shopping retrieval plan or a clarification/direct-answer decision."
    ),
    instruction=PLANNER_INSTRUCTION,
    output_key="planner_output_raw",
)


# ============================================================
# 6. Planner controller
# ============================================================

class PlannerControllerAgent(BaseAgent):
    """
    Runs the inner LLM planner and controls whether the rest of the pipeline
    should proceed.
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state = _session_state(ctx)

        # ------------------------------------------------------------
        # Deterministic guard 1:
        # A/B/E image options were generated. Ask user to pick one.
        # This avoids paying for the LLM and prevents Browserbase drift.
        # ------------------------------------------------------------
        if _image_iteration_needs_selection(state):
            planner_output = _forced_image_iteration_selection_plan(state)
            stop_reason = (
                "Image iteration options are pending selection; browsing is blocked."
            )

            state_delta = {
                "planner_output": planner_output,
                "planner_output_json": _json_dumps(planner_output),
                "planner_output_raw": _json_dumps(planner_output),
                "planner_should_continue": False,
                "planner_stop_reason": stop_reason,
                "pipeline_status": "stopped_after_planner",
            }

            try:
                ctx.end_invocation = True
            except Exception:
                pass

            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=planner_output["response_to_user"])],
                ),
            )
            return

        # ------------------------------------------------------------
        # Deterministic guard 2:
        # User asked for image iteration but A/B/E failed.
        # ------------------------------------------------------------
        if _image_iteration_failed_this_turn(state):
            planner_output = _forced_image_iteration_failure_plan(state)
            stop_reason = "Image iteration failed this turn."

            state_delta = {
                "planner_output": planner_output,
                "planner_output_json": _json_dumps(planner_output),
                "planner_output_raw": _json_dumps(planner_output),
                "planner_should_continue": False,
                "planner_stop_reason": stop_reason,
                "pipeline_status": "stopped_after_planner",
            }

            try:
                ctx.end_invocation = True
            except Exception:
                pass

            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=planner_output["response_to_user"])],
                ),
            )
            return

        # ------------------------------------------------------------
        # Deterministic guard 3:
        # A reliable Gemini-native visual edit completed. Stop so user can inspect.
        # ------------------------------------------------------------
        if _image_reliable_edit_completed_this_turn(state):
            planner_output = _forced_reliable_edit_completed_plan(state)
            stop_reason = "Reliable image edit completed this turn."

            state_delta = {
                "planner_output": planner_output,
                "planner_output_json": _json_dumps(planner_output),
                "planner_output_raw": _json_dumps(planner_output),
                "planner_should_continue": False,
                "planner_stop_reason": stop_reason,
                "pipeline_status": "stopped_after_planner",
            }

            try:
                ctx.end_invocation = True
            except Exception:
                pass

            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=planner_output["response_to_user"])],
                ),
            )
            return

        # ------------------------------------------------------------
        # Normal LLM planning.
        # ------------------------------------------------------------
        planner_text_chunks: List[str] = []
        non_partial_chunks: List[str] = []

        async for event in _planner_llm.run_async(ctx):
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                text = getattr(part, "text", None)

                if not text:
                    continue

                planner_text_chunks.append(str(text))

                if not getattr(event, "partial", False):
                    non_partial_chunks.append(str(text))

        planner_raw_text = "\n".join(
            non_partial_chunks or planner_text_chunks
        ).strip()

        planner_output = _parse_json_maybe(planner_raw_text)
        planner_output = _sanitize_planner_output(planner_output, state)

        should_stop, stop_reason = _planner_should_stop(planner_output)

        state_delta: Dict[str, Any] = {
            "planner_output": planner_output,
            "planner_output_json": _json_dumps(planner_output),
            "planner_output_raw": planner_raw_text,
            "planner_should_continue": not should_stop,
            "planner_stop_reason": stop_reason,
        }

        if should_stop:
            state_delta["pipeline_status"] = "stopped_after_planner"

            try:
                ctx.end_invocation = True
            except Exception:
                pass

            user_message = _render_stop_message(planner_output, stop_reason)

            yield Event(
                author=self.name,
                actions=EventActions(state_delta=state_delta),
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=user_message)],
                ),
            )
            return

        state_delta["pipeline_status"] = "planner_completed_continue"
        state_delta["last_actionable_planner_output"] = planner_output
        state_delta["last_actionable_planner_output_json"] = _json_dumps(
            planner_output
        )

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


planner_agent = PlannerControllerAgent(
    name="PlannerAgent",
    description=(
        "Runs the conversational shopping planner, handles image-iteration selection, "
        "answers or asks clarification when browsing should not run, and allows the "
        "pipeline to continue only when an actionable browser_query exists."
    ),
)