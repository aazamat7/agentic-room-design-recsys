from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from google import genai
from google.cloud import aiplatform
from google.genai import types

from renovation_agent.config import Settings
from renovation_agent.services.image_io import ImagePayload, extension_for_mime


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes | None = None
    mime_type: str | None = None
    uri: str | None = None
    model: str | None = None
    raw_metadata: dict[str, Any] | None = None


class ImageEditBackend(Protocol):
    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage: ...


class HttpLoraBackend:
    """Calls a custom Qwen+LoRA HTTP service using a multipart request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.qwen_lora_api_url:
            raise ValueError("QWEN_LORA_API_URL is required for QWEN_LORA_BACKEND=http")

    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage:
        files: dict[str, tuple[str, bytes, str]] = {
            self.settings.qwen_lora_source_field: (
                source.filename,
                source.data,
                source.mime_type,
            )
        }
        reference_field = _optional_field(self.settings.qwen_lora_reference_field)
        if (
            reference
            and reference_field
            and self.settings.qwen_lora_send_reference_image
        ):
            files[reference_field] = (
                reference.filename,
                reference.data,
                reference.mime_type,
            )

        extra_input = self.settings.qwen_lora_extra_input()
        form: dict[str, str] = {
            self.settings.qwen_lora_prompt_field: prompt,
            "seed": str(seed),
            "num_inference_steps": str(self.settings.num_inference_steps),
            "guidance_scale": str(self.settings.guidance_scale),
            "extra_input_json": json.dumps(extra_input),
        }
        for key, value in extra_input.items():
            form.setdefault(
                key,
                json.dumps(value) if isinstance(value, (dict, list)) else str(value),
            )
        headers: dict[str, str] = {}
        if self.settings.qwen_lora_http_bearer_token:
            headers["Authorization"] = (
                f"Bearer {self.settings.qwen_lora_http_bearer_token}"
            )

        if self.settings.qwen_lora_healthcheck_before_generation:
            self._assert_ready(headers=headers)

        try:
            response = requests.post(
                self.settings.qwen_lora_api_url,
                files=files,
                data=form,
                headers=headers,
                timeout=self.settings.generation_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = _response_error_detail(getattr(exc, "response", None))
            suffix = f" Server response: {detail}" if detail else ""
            raise RuntimeError(
                "The local Qwen+LoRA inference endpoint request failed. "
                f"URL={self.settings.qwen_lora_api_url}.{suffix}"
            ) from exc
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type.startswith("image/"):
            metadata = {
                "backend": "qwen_lora_http",
                "endpoint": self.settings.qwen_lora_api_url,
                "seed": response.headers.get("X-Seed", str(seed)),
                "width": response.headers.get("X-Width"),
                "height": response.headers.get("X-Height"),
                "endpoint_model": response.headers.get("X-Model"),
            }
            return GeneratedImage(
                data=response.content,
                mime_type=content_type,
                model=response.headers.get("X-Model", "qwen-lora-http"),
                raw_metadata={k: v for k, v in metadata.items() if v is not None},
            )
        payload = response.json()
        return _generated_from_value(payload, model="qwen-lora-http")

    def _assert_ready(self, *, headers: dict[str, str]) -> None:
        health_url = self.settings.qwen_lora_health_url
        if not health_url:
            return
        try:
            response = requests.get(health_url, headers=headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "The local Qwen+LoRA endpoint is not reachable or did not return "
                f"valid health JSON. Health URL={health_url}."
            ) from exc
        status = str(payload.get("status", "")).lower()
        if status != "ready":
            raise RuntimeError(
                "The local Qwen+LoRA endpoint is not ready. "
                f"Health response={payload!r}"
            )


class VertexEndpointLoraBackend:
    """Calls a custom Vertex AI endpoint with a documented base64 JSON contract."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.qwen_lora_vertex_endpoint:
            raise ValueError(
                "QWEN_LORA_VERTEX_ENDPOINT is required for QWEN_LORA_BACKEND=vertex_endpoint"
            )
        aiplatform.init(
            project=settings.project_id, location=settings.qwen_lora_vertex_location
        )
        self.endpoint = aiplatform.Endpoint(settings.qwen_lora_vertex_endpoint)

    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage:
        instance: dict[str, Any] = {
            "source_image_base64": base64.b64encode(source.data).decode("ascii"),
            "source_mime_type": source.mime_type,
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": self.settings.num_inference_steps,
            "guidance_scale": self.settings.guidance_scale,
            **self.settings.qwen_lora_extra_input(),
        }
        if reference:
            instance.update(
                {
                    "reference_image_base64": base64.b64encode(reference.data).decode(
                        "ascii"
                    ),
                    "reference_mime_type": reference.mime_type,
                }
            )
        prediction = self.endpoint.predict(instances=[instance])
        values = getattr(prediction, "predictions", None) or []
        if not values:
            raise RuntimeError("Vertex Qwen+LoRA endpoint returned no predictions")
        return _generated_from_value(values[0], model="qwen-lora-vertex-endpoint")


class ReplicateModelBackend:
    """Runs either the initial LoRA model or the follow-up iteration model."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: str,
        source_field: str,
        prompt_field: str,
        reference_field: str | None,
        extra_input: dict[str, Any],
        model_label: str,
    ):
        self.settings = settings
        self.model = model
        self.source_field = source_field
        self.prompt_field = prompt_field
        self.reference_field = reference_field
        self.extra_input = extra_input
        self.model_label = model_label

    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage:
        try:
            import replicate
        except ImportError as exc:
            raise RuntimeError("Install the 'replicate' package to use Replicate") from exc
        if not self.settings.replicate_api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is required for follow-up Replicate iterations"
            )
        client = replicate.Client(api_token=self.settings.replicate_api_token)

        with tempfile.TemporaryDirectory(prefix="renovation-qwen-") as tmp:
            source_path = _write_temp_payload(source, Path(tmp), "source")
            reference_path = (
                _write_temp_payload(reference, Path(tmp), "reference")
                if reference
                else None
            )
            with source_path.open("rb") as source_file:
                model_input: dict[str, Any] = {
                    self.source_field: source_file,
                    self.prompt_field: prompt,
                    **self.extra_input,
                }
                if seed >= 0 and "seed" not in model_input:
                    model_input["seed"] = seed
                if (
                    reference_path
                    and self.reference_field
                    and self.reference_field not in model_input
                ):
                    with reference_path.open("rb") as reference_file:
                        model_input[self.reference_field] = reference_file
                        output = client.run(self.model, input=model_input)
                else:
                    output = client.run(self.model, input=model_input)
        return _generated_from_value(output, model=self.model_label)


class GeminiFlashImageBackend:
    """Uses Gemini Flash Image for the first renovated result.

    This is useful as a testing substitute when the Qwen+LoRA model is not yet
    available. It consumes the source room, the selected reference image, and
    the already-composed initial edit prompt.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.gemini_location,
        )

    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage:
        instruction = _compose_gemini_image_edit_prompt(prompt=prompt)
        parts: list[types.Part] = [
            types.Part.from_text(text=instruction),
            types.Part.from_bytes(data=source.data, mime_type=source.mime_type),
        ]
        if reference is not None:
            parts.append(
                types.Part.from_bytes(
                    data=reference.data,
                    mime_type=reference.mime_type,
                )
            )

        response = self.client.models.generate_content(
            model=self.settings.initial_gemini_image_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        image = _extract_generated_image_from_gemini_response(
            response,
            model=self.settings.initial_gemini_image_model,
        )
        metadata = {
            "backend": "gemini_flash_image",
            "seed": seed,
        }
        text_response = getattr(response, "text", None)
        if text_response:
            metadata["text"] = text_response

        return GeneratedImage(
            data=image.data,
            mime_type=image.mime_type,
            uri=image.uri,
            model=self.settings.initial_gemini_image_model,
            raw_metadata=metadata,
        )


class GeminiFlashImageFallbackBackend:
    """Fallback backend used only when the preferred Qwen+LoRA first-result path fails."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.gemini_location,
        )

    def generate(
        self,
        *,
        source: ImagePayload,
        reference: ImagePayload | None,
        prompt: str,
        seed: int,
    ) -> GeneratedImage:
        instruction = _compose_gemini_image_edit_prompt(prompt=prompt)
        parts: list[types.Part] = [
            types.Part.from_text(text=instruction),
            types.Part.from_bytes(data=source.data, mime_type=source.mime_type),
        ]
        if reference is not None:
            parts.append(
                types.Part.from_bytes(
                    data=reference.data,
                    mime_type=reference.mime_type,
                )
            )

        response = self.client.models.generate_content(
            model=self.settings.gemini_image_fallback_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        image = _extract_generated_image_from_gemini_response(
            response,
            model=self.settings.gemini_image_fallback_model,
        )
        metadata = {
            "backend": "gemini_flash_image_fallback",
            "fallback_used": True,
            "seed": seed,
        }
        text_response = getattr(response, "text", None)
        if text_response:
            metadata["text"] = text_response

        return GeneratedImage(
            data=image.data,
            mime_type=image.mime_type,
            uri=image.uri,
            model=self.settings.gemini_image_fallback_model,
            raw_metadata=metadata,
        )


def build_initial_backend(settings: Settings) -> ImageEditBackend:
    if settings.qwen_lora_backend == "http":
        return HttpLoraBackend(settings)
    if settings.qwen_lora_backend == "vertex_endpoint":
        return VertexEndpointLoraBackend(settings)
    if settings.qwen_lora_backend == "replicate_model":
        if not settings.qwen_lora_replicate_model:
            raise ValueError(
                "QWEN_LORA_REPLICATE_MODEL is required for QWEN_LORA_BACKEND=replicate_model"
            )
        return ReplicateModelBackend(
            settings=settings,
            model=settings.qwen_lora_replicate_model,
            source_field=settings.qwen_lora_source_field,
            prompt_field=settings.qwen_lora_prompt_field,
            reference_field=_optional_field(settings.qwen_lora_reference_field),
            extra_input=settings.qwen_lora_extra_input(),
            model_label="qwen-lora-replicate",
        )
    if settings.qwen_lora_backend == "gemini_flash_image":
        return GeminiFlashImageBackend(settings)
    raise ValueError(f"Unsupported QWEN_LORA_BACKEND={settings.qwen_lora_backend}")


def build_initial_fallback_backend(settings: Settings) -> ImageEditBackend | None:
    if not settings.enable_initial_gemini_fallback:
        return None
    return GeminiFlashImageFallbackBackend(settings)


def build_initial_comparison_backend(settings: Settings) -> ImageEditBackend | None:
    if not settings.generate_initial_gemini_comparison:
        return None
    return GeminiFlashImageBackend(settings)


def build_iteration_backend(settings: Settings) -> ImageEditBackend:
    return ReplicateModelBackend(
        settings=settings,
        model=settings.replicate_iteration_model,
        source_field=settings.replicate_iteration_image_field,
        prompt_field=settings.replicate_iteration_prompt_field,
        reference_field=_optional_field(settings.replicate_iteration_reference_field),
        extra_input=settings.replicate_iteration_extra_input(),
        model_label=settings.replicate_iteration_model,
    )


def _compose_gemini_image_edit_prompt(*, prompt: str) -> str:
    return (
        "Image 1 is the user's original room. "
        "Image 2, when present, is the style reference selected by the user. "
        "Generate the first renovated-room result. Preserve the source room's architecture, "
        "camera viewpoint, geometry, windows, doors, wall positions, ceiling, floor boundaries, "
        "and perspective. Use the style reference for palette, furniture language, materials, "
        "decor density, and mood. Return a single photorealistic edited room image. "
        "Do not add text, people, watermarks, warped geometry, duplicated objects, or blur.\n\n"
        f"EDIT BRIEF:\n{prompt}"
    )


@dataclass(frozen=True)
class _InlineImage:
    data: bytes
    mime_type: str
    uri: str | None = None


def _extract_generated_image_from_gemini_response(
    response: Any,
    *,
    model: str,
) -> _InlineImage:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            mime_type = getattr(inline_data, "mime_type", None) or "image/png"
            if data:
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return _InlineImage(data=data, mime_type=mime_type)
    raise RuntimeError(
        f"{model} did not return an inline image. Check that the configured Gemini image model supports image generation/editing."
    )


def _write_temp_payload(
    payload: ImagePayload | None,
    directory: Path,
    stem: str,
) -> Path:
    if payload is None:
        raise ValueError("Image payload is required")
    path = directory / f"{stem}{extension_for_mime(payload.mime_type)}"
    path.write_bytes(payload.data)
    return path


def _response_error_detail(response: Any) -> str | None:
    if response is None:
        return None
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error") or payload.get("message")
            if detail:
                return str(detail)
    except Exception:
        pass
    text = getattr(response, "text", "")
    return text[:500] if text else None


def _generated_from_value(value: Any, *, model: str) -> GeneratedImage:
    """Normalize common Replicate/custom endpoint output shapes."""
    if value is None:
        raise RuntimeError(f"{model} returned no output")

    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError(f"{model} returned an empty output list")
        return _generated_from_value(value[0], model=model)

    if isinstance(value, bytes):
        return GeneratedImage(data=value, mime_type="image/png", model=model)

    if isinstance(value, str):
        if value.startswith("data:image/"):
            header, encoded = value.split(",", 1)
            mime = header.split(";", 1)[0].removeprefix("data:")
            return GeneratedImage(
                data=base64.b64decode(encoded), mime_type=mime, model=model
            )
        return GeneratedImage(uri=value, model=model)

    if isinstance(value, dict):
        for key in (
            "output",
            "output_url",
            "image_url",
            "url",
            "image",
            "generated_image",
            "gcs_uri",
        ):
            if key in value and value[key] is not None:
                generated = _generated_from_value(value[key], model=model)
                return GeneratedImage(
                    data=generated.data,
                    mime_type=generated.mime_type,
                    uri=generated.uri,
                    model=model,
                    raw_metadata=value,
                )
        for key in ("output_base64", "image_base64", "generated_image_base64"):
            if key in value and value[key]:
                return GeneratedImage(
                    data=base64.b64decode(value[key]),
                    mime_type=value.get("mime_type", "image/png"),
                    model=model,
                    raw_metadata=value,
                )

    url = getattr(value, "url", None)
    if callable(url):
        url = url()
    if url:
        return GeneratedImage(uri=str(url), model=model)

    read = getattr(value, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return GeneratedImage(data=data, mime_type="image/png", model=model)

    raise RuntimeError(f"Could not interpret output from {model}: {type(value).__name__}")


def _optional_field(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"", "none", "null", "disabled"}:
        return None
    return cleaned
