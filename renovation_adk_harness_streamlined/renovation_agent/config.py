from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # Google Cloud / Gemini
    project_id: str = Field(default="adsp-s26-reccys", alias="GOOGLE_CLOUD_PROJECT")
    vector_location: str = Field(default="us-central1", alias="VECTOR_LOCATION")
    gemini_location: str = Field(default="global", alias="GEMINI_LOCATION")
    orchestration_model: str = Field(
        default="gemini-3.5-flash", alias="ORCHESTRATION_MODEL"
    )
    reasoning_model: str = Field(
        default="gemini-3.1-pro-preview", alias="GEMINI_REASONING_MODEL"
    )
    embedding_model: str = Field(
        default="gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )

    embedding_location: str = Field(
    default="us",
    alias="EMBEDDING_LOCATION",
    )

    # Existing Vector Search assets
    vector_data_prefix: str = Field(
        default="gs://adsp-s26-reccys-bucket/living-room-renovation-index",
        alias="VECTOR_DATA_PREFIX",
    )
    vector_index_display_name: str = Field(
        default="living-room-renovation-index", alias="VECTOR_INDEX_DISPLAY_NAME"
    )
    vector_index_name: str | None = Field(default=None, alias="VECTOR_INDEX_NAME")
    vector_index_endpoint_name: str | None = Field(
        default=None, alias="VECTOR_INDEX_ENDPOINT_NAME"
    )
    deployed_index_id: str | None = Field(default=None, alias="DEPLOYED_INDEX_ID")
    vector_metadata_uri: str | None = Field(default=None, alias="VECTOR_METADATA_URI")
    embedding_dimension: int | None = Field(default=768, alias="EMBEDDING_DIMENSION")
    default_num_neighbors: int = Field(default=3, alias="DEFAULT_NUM_NEIGHBORS")

    # Durable images and previews
    output_bucket: str = Field(
        default="adsp-s26-reccys-bucket", alias="OUTPUT_BUCKET"
    )
    output_prefix: str = Field(
        default="renovation-agent-outputs", alias="OUTPUT_PREFIX"
    )
    signed_url_ttl_minutes: int = Field(default=120, alias="SIGNED_URL_TTL_MINUTES")
    max_image_bytes: int = Field(default=25_000_000, alias="MAX_IMAGE_BYTES")

    # Initial first-result backend. The first renovated image normally uses Qwen+LoRA,
    # but gemini_flash_image is supported as a testing substitute when the LoRA model
    # is not yet available.
    qwen_lora_backend: Literal[
        "http", "vertex_endpoint", "replicate_model", "gemini_flash_image"
    ] = Field(default="http", alias="QWEN_LORA_BACKEND")
    qwen_lora_api_url: str | None = Field(
        default="http://127.0.0.1:8001/edit", alias="QWEN_LORA_API_URL"
    )
    qwen_lora_http_bearer_token: str | None = Field(
        default=None, alias="QWEN_LORA_HTTP_BEARER_TOKEN"
    )
    qwen_lora_health_url: str | None = Field(
        default="http://127.0.0.1:8001/health", alias="QWEN_LORA_HEALTH_URL"
    )
    qwen_lora_healthcheck_before_generation: bool = Field(
        default=True, alias="QWEN_LORA_HEALTHCHECK_BEFORE_GENERATION"
    )
    qwen_lora_send_reference_image: bool = Field(
        default=False, alias="QWEN_LORA_SEND_REFERENCE_IMAGE"
    )
    qwen_lora_vertex_endpoint: str | None = Field(
        default=None, alias="QWEN_LORA_VERTEX_ENDPOINT"
    )
    qwen_lora_vertex_location: str = Field(
        default="us-central1", alias="QWEN_LORA_VERTEX_LOCATION"
    )
    qwen_lora_replicate_model: str | None = Field(
        default=None, alias="QWEN_LORA_REPLICATE_MODEL"
    )
    qwen_lora_source_field: str = Field(default="image", alias="QWEN_LORA_SOURCE_FIELD")
    qwen_lora_reference_field: str = Field(
        default="reference_image", alias="QWEN_LORA_REFERENCE_FIELD"
    )
    qwen_lora_prompt_field: str = Field(default="prompt", alias="QWEN_LORA_PROMPT_FIELD")
    qwen_lora_extra_input_json: str = Field(
        default="{}", alias="QWEN_LORA_EXTRA_INPUT_JSON"
    )

    # Optional Gemini Flash Image backend for a first-result A/B comparison.
    initial_gemini_image_model: str = Field(
        default="gemini-3.1-flash-image", alias="INITIAL_GEMINI_IMAGE_MODEL"
    )
    generate_initial_gemini_comparison: bool = Field(
        default=True, alias="GENERATE_INITIAL_GEMINI_COMPARISON"
    )

    # Qwen Replicate API for follow-up iterations
    replicate_api_token: str | None = Field(
        default=None, alias="REPLICATE_API_TOKEN"
    )
    replicate_iteration_model: str = Field(
        default="qwen/qwen-image-edit", alias="REPLICATE_ITERATION_MODEL"
    )
    replicate_iteration_image_field: str = Field(
        default="image", alias="REPLICATE_ITERATION_IMAGE_FIELD"
    )
    replicate_iteration_reference_field: str | None = Field(
        default=None, alias="REPLICATE_ITERATION_REFERENCE_FIELD"
    )
    replicate_iteration_prompt_field: str = Field(
        default="prompt", alias="REPLICATE_ITERATION_PROMPT_FIELD"
    )
    replicate_iteration_extra_input_json: str = Field(
        default="{}", alias="REPLICATE_ITERATION_EXTRA_INPUT_JSON"
    )

    # Optional fallback when the initial Qwen+LoRA backend is unavailable
    enable_initial_gemini_fallback: bool = Field(
        default=True, alias="ENABLE_INITIAL_GEMINI_FALLBACK"
    )
    gemini_image_fallback_model: str = Field(
        default="gemini-3.1-flash-image", alias="GEMINI_IMAGE_FALLBACK_MODEL"
    )

    # Shared generation settings
    generation_timeout_seconds: int = Field(
        default=900, alias="GENERATION_TIMEOUT_SECONDS"
    )
    num_inference_steps: int = Field(default=28, alias="NUM_INFERENCE_STEPS")
    guidance_scale: float = Field(default=4.0, alias="GUIDANCE_SCALE")
    default_seed: int = Field(default=-1, alias="DEFAULT_SEED")

    fixed_renovation_prompt: str = Field(
        default=(
            "Create a photorealistic renovated version of the input room. Preserve the exact "
            "room architecture, camera viewpoint, perspective, window and door positions, wall "
            "geometry, ceiling height, floor boundaries, and lighting direction. Apply the selected "
            "reference's design language through furniture, decor, colors, materials, and styling. "
            "Do not add structural openings, move walls, change the camera, distort straight lines, "
            "duplicate objects, or introduce text, people, watermarks, blur, or surreal artifacts."
        ),
        alias="FIXED_RENOVATION_PROMPT",
    )

    @field_validator("vector_data_prefix", mode="before")
    @classmethod
    def normalize_vector_prefix(cls, value: Any) -> Any:
        if isinstance(value, str) and value and not value.startswith("gs://"):
            return f"gs://{value.lstrip('/')}"
        return value

    @field_validator("output_bucket", mode="before")
    @classmethod
    def normalize_bucket(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.removeprefix("gs://").strip("/")
        return value

    def qwen_lora_extra_input(self) -> dict[str, Any]:
        return _parse_json_object(self.qwen_lora_extra_input_json, "QWEN_LORA_EXTRA_INPUT_JSON")

    def replicate_iteration_extra_input(self) -> dict[str, Any]:
        return _parse_json_object(
            self.replicate_iteration_extra_input_json,
            "REPLICATE_ITERATION_EXTRA_INPUT_JSON",
        )


def _parse_json_object(raw: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
