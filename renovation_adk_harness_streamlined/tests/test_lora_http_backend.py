from __future__ import annotations

from types import SimpleNamespace

from renovation_agent.config import Settings
from renovation_agent.services.image_io import ImagePayload
from renovation_agent.services.qwen_backends import (
    HttpLoraBackend,
    build_initial_comparison_backend,
)


def _image(name: str = "room.png") -> ImagePayload:
    return ImagePayload(
        data=b"fake-image",
        mime_type="image/png",
        filename=name,
        source="test",
    )


def test_http_lora_uses_expected_contract(monkeypatch):
    settings = Settings(
        QWEN_LORA_BACKEND="http",
        QWEN_LORA_API_URL="http://127.0.0.1:8001/edit",
        QWEN_LORA_HEALTH_URL="http://127.0.0.1:8001/health",
        QWEN_LORA_HEALTHCHECK_BEFORE_GENERATION=True,
        QWEN_LORA_SEND_REFERENCE_IMAGE=False,
        NUM_INFERENCE_STEPS=30,
        GUIDANCE_SCALE=4.0,
    )
    calls = {}

    class Response:
        headers = {
            "content-type": "image/png",
            "X-Model": "qwen-image-edit-2511-lora",
            "X-Seed": "42",
        }
        content = b"generated"
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ready"}

    def fake_get(url, **kwargs):
        calls["health"] = (url, kwargs)
        return Response()

    def fake_post(url, **kwargs):
        calls["post"] = (url, kwargs)
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)

    output = HttpLoraBackend(settings).generate(
        source=_image(),
        reference=_image("reference.png"),
        prompt="renovate",
        seed=42,
    )

    assert output.data == b"generated"
    assert output.model == "qwen-image-edit-2511-lora"
    assert calls["post"][1]["data"]["num_inference_steps"] == "30"
    assert "image" in calls["post"][1]["files"]
    assert "reference_image" not in calls["post"][1]["files"]


def test_gemini_comparison_can_be_disabled():
    settings = Settings(GENERATE_INITIAL_GEMINI_COMPARISON=False)
    assert build_initial_comparison_backend(settings) is None
