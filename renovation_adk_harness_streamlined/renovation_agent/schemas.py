from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReferenceCandidate(BaseModel):
    rank: int
    reference_id: str
    distance: float | None = None
    image_uri: str | None = None
    preview_url: str | None = None
    style: str | None = None
    room_type: str | None = None
    caption: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class RenovationPlan(BaseModel):
    design_summary: str = Field(
        description="A concise description of the proposed renovated room."
    )
    reference_style: str = Field(
        description="The primary style inferred from the selected reference."
    )
    color_palette: list[str] = Field(
        description="Dominant and accent colors to carry into the renovation."
    )
    materials: list[str] = Field(
        description="Materials and finishes to use in furniture, textiles, flooring, and decor."
    )
    preserve: list[str] = Field(
        description="Elements of the source room that must remain geometrically unchanged."
    )
    change: list[str] = Field(
        description="Furniture, decor, palette, and styling changes to make."
    )
    negative_constraints: list[str] = Field(
        description="Failures and unwanted changes that the image model must avoid."
    )
    qwen_prompt: str = Field(
        description="A detailed image-editing prompt ready for the Qwen image-edit model."
    )


class GenerationRecord(BaseModel):
    step: str
    prompt: str
    output_uri: str
    preview_url: str | None = None
    seed: int | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
