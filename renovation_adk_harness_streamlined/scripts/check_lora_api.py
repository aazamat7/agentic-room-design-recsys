from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
from urllib.parse import urljoin

import requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the local Qwen+LoRA FastAPI service and optionally generate one image."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8001/edit")
    parser.add_argument("--health-url", default="http://127.0.0.1:8001/health")
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/generated/lora_api_test.png"),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Create a photorealistic modern furnished renovation. Preserve the exact "
            "architecture, camera viewpoint, perspective, windows, doors, walls, ceiling "
            "and floor boundaries. Use warm neutral colors, light oak furniture, cream "
            "upholstery and restrained decor."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    health = requests.get(args.health_url, timeout=10)
    health.raise_for_status()
    payload = health.json()
    print("Health:", payload)
    if str(payload.get("status", "")).lower() != "ready":
        raise SystemExit("LoRA endpoint is not ready")

    if args.image is None:
        return
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    mime_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    with args.image.open("rb") as image_file:
        response = requests.post(
            args.api_url,
            files={"image": (args.image.name, image_file, mime_type)},
            data={
                "prompt": args.prompt,
                "seed": str(args.seed),
                "num_inference_steps": str(args.steps),
                "guidance_scale": str(args.guidance_scale),
                "extra_input_json": "{}",
            },
            timeout=args.timeout,
        )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(
            f"Expected an image response, got {content_type}: {response.text[:500]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print("Saved:", args.output.resolve())
    print("Response headers:", dict(response.headers))


if __name__ == "__main__":
    main()
