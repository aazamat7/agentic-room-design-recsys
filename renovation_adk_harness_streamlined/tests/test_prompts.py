from renovation_agent.config import Settings
from renovation_agent.schemas import RenovationPlan
from renovation_agent.services.gemini_reasoner import (
    compose_initial_qwen_prompt,
    compose_iteration_prompt,
)


def sample_plan() -> RenovationPlan:
    return RenovationPlan(
        design_summary="Warm minimalist living room",
        reference_style="warm minimalist",
        color_palette=["cream", "oak"],
        materials=["linen", "light oak"],
        preserve=["windows", "camera"],
        change=["replace sofa", "reduce clutter"],
        negative_constraints=["no warped walls"],
        qwen_prompt="Use a cream sofa and oak furniture.",
    )


def test_initial_prompt_contains_preservation_and_changes():
    prompt = compose_initial_qwen_prompt(Settings(), sample_plan(), "Modernize it")
    assert "PRESERVE EXACTLY" in prompt
    assert "replace sofa" in prompt
    assert "Modernize it" in prompt


def test_iteration_prompt_is_delta_only():
    prompt = compose_iteration_prompt(Settings(), sample_plan(), "Make sofa beige")
    assert "Apply only the requested delta" in prompt
    assert "Make sofa beige" in prompt
