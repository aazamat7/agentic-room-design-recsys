"""Plumbing-only mock. It echoes the source image; it is not a Qwen model."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

app = FastAPI(title="Mock Qwen LoRA Server")


@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    reference_image: UploadFile | None = File(default=None),
    seed: int = Form(default=-1),
    num_inference_steps: int = Form(default=28),
    guidance_scale: float = Form(default=4.0),
    extra_input_json: str = Form(default="{}"),
) -> Response:
    del prompt, reference_image, seed, num_inference_steps, guidance_scale, extra_input_json
    data = await image.read()
    return Response(content=data, media_type=image.content_type or "image/png")
