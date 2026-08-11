from __future__ import annotations

import json

from google import genai
from google.genai import types

from renovation_agent.config import Settings
from renovation_agent.schemas import RenovationPlan
from renovation_agent.services.image_io import ImagePayload


class GeminiRenovationReasoner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.gemini_location,
        )

    def create_plan(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload,
        user_goal: str,
        user_notes: str,
        room_type: str,
        prior_history: list[dict],
    ) -> RenovationPlan:
        prompt = f"""
You are the visual design reasoning stage of a room-renovation system.

Image 1 is the user's current room. Image 2 is the renovated reference selected by the user.
Create a precise edit plan for an image-edit model. Transfer the reference's style language, not its
architecture. The user's room geometry must remain unchanged.

Room type: {room_type or 'living_room'}
Original user goal: {user_goal or 'Renovate this room in the selected reference style.'}
Current user notes: {user_notes or 'No additional notes.'}
Prior design/edit history: {json.dumps(prior_history[-5:], ensure_ascii=False)}

Hard constraints:
- Preserve camera position, perspective, walls, windows, doors, ceiling, floor boundaries, and room scale.
- Preserve all architecture unless the user explicitly asks for a structural change.
- Use the reference only for palette, furniture language, textiles, decor density, materials, and mood.
- Avoid blur, warped geometry, duplicated furniture, impossible shadows, text, logos, people, and watermarks.
- The qwen_prompt must be directly usable as an image-editing prompt and must clearly separate what to
  preserve from what to change.
""".strip()

        response = self.client.models.generate_content(
            model=self.settings.reasoning_model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=source.data, mime_type=source.mime_type
                        ),
                        types.Part.from_bytes(
                            data=reference.data, mime_type=reference.mime_type
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_json_schema=RenovationPlan.model_json_schema(),
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini reasoning returned an empty response")
        return RenovationPlan.model_validate_json(_strip_json_fence(response.text))


def compose_initial_qwen_prompt(
    settings: Settings,
    plan: RenovationPlan,
    user_goal: str,
) -> str:
    preserve = "; ".join(plan.preserve)
    changes = "; ".join(plan.change)
    negatives = "; ".join(plan.negative_constraints)
    return (
        f"{settings.fixed_renovation_prompt}\n\n"
        f"USER GOAL: {user_goal or plan.design_summary}\n"
        f"REFERENCE STYLE: {plan.reference_style}.\n"
        f"PALETTE: {', '.join(plan.color_palette)}.\n"
        f"MATERIALS: {', '.join(plan.materials)}.\n"
        f"PRESERVE EXACTLY: {preserve}.\n"
        f"MAKE THESE CHANGES: {changes}.\n"
        f"DETAILED EDIT: {plan.qwen_prompt}\n"
        f"AVOID: {negatives}."
    )


def compose_iteration_prompt(
    settings: Settings,
    plan: RenovationPlan | None,
    edit_request: str,
) -> str:
    plan_context = ""
    if plan:
        plan_context = (
            f"Maintain the established {plan.reference_style} design, palette "
            f"({', '.join(plan.color_palette)}), and materials ({', '.join(plan.materials)}). "
        )
    return (
        "Edit the current generated room image, not the original reference. "
        "Apply only the requested delta and keep all unmentioned elements unchanged. "
        "Preserve the exact camera, room architecture, perspective, geometry, object placement, "
        "lighting direction, and visual identity already present. "
        f"{plan_context}"
        f"REQUESTED DELTA: {edit_request}. "
        "Do not redesign the whole room. Avoid blur, warping, duplicated objects, text, logos, "
        "people, watermarks, and unintended color shifts."
    )


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()
