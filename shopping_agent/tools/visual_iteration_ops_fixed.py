"""
shopping_agent/tools/visual_iteration_ops.py

Self-contained image-iteration backend for VisualPreferenceAgent.

This file intentionally contains everything that was previously split across:
- facet_pipeline.py       -> Gemini image generation helper + variant pipelines
- evaluation_pipeline.py  -> pipeline registry for Phase-1 variants
- image_ops.py            -> Imagen/Gemini operation registry
- reliable_ops.py         -> verify/retry/fallback layer
- design_session.py       -> two-phase session orchestration

External project imports are avoided. The only runtime dependencies are SDKs/services:
- PIL
- google-genai for Gemini image generation / image editing
- vertexai.preview.vision_models for Imagen semantic-mask inpainting
- anthropic.AnthropicVertex for required Claude verification only

Main public entry points used by visual_preference_agent.py:
- run_design_variants(...)
- run_reliable_edit(...)
- is_merge_or_product_placement(...)
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import textwrap
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ============================================================
# 0. Runtime configuration
# ============================================================

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

PROJECT_ID = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GOOGLE_PROJECT_ID")
    or os.getenv("PROJECT_ID")
)
# LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_REGION", "us-central1"))
# os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

# GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
# GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
# IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "imagen-3.0-capability-001")

# EMBED_LOCATION = (
#     os.getenv("GOOGLE_CLOUD_EMBED_LOCATION")
#     or os.getenv("GOOGLE_CLOUD_EMBEDDING_LOCATION")
#     or os.getenv("GOOGLE_CLOUD_LOCATION")
#     or "global"
# )

LOCATION = os.getenv("GOOGLE_CLOUD_TEXT_LOCATION") or "global"
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# Image generation and Imagen editing are forced to global for these publisher models.
# Do not inherit GOOGLE_CLOUD_LOCATION / GOOGLE_CLOUD_REGION here.
GEMINI_IMAGE_LOCATION = "global"
# IMAGEN_LOCATION = "global"
os.environ["GEMINI_IMAGE_LOCATION"] = GEMINI_IMAGE_LOCATION
# os.environ["IMAGEN_LOCATION"] = IMAGEN_LOCATION

GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
# IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "imagen-3.0-capability-001")
IMAGEN_LOCATION =  "us-central1"
IMAGEN_GEN_MODEL = "imagen-3.0-generate-002"
IMAGEN_MODEL = "imagen-3.0-capability-001"

# Embeddings must stay global and must not inherit GOOGLE_CLOUD_LOCATION / GOOGLE_CLOUD_REGION.
EMBED_LOCATION = "global"
os.environ["GOOGLE_CLOUD_EMBED_LOCATION"] = EMBED_LOCATION

EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
TOP_K = int(os.getenv("TOP_K", "5"))
STYLE_BRANCHES = int(os.getenv("STYLE_BRANCHES", "4"))
COLOR_TOP_N = int(os.getenv("COLOR_TOP_N", "3"))
MATERIAL_TOP_N = int(os.getenv("MATERIAL_TOP_N", "3"))
MAX_CONCURRENT_IMAGE_GEN = int(os.getenv("MAX_CONCURRENT_IMAGE_GEN", "3"))

IMAGE_ITERATION_OUTPUT_DIR = Path(
    os.getenv(
        "IMAGE_ITERATION_OUTPUT_DIR",
        "shopping_agent/data/generated/image_iterations",
    )
)

RELIABLE_IMAGE_EDIT_TRIES = int(os.getenv("RELIABLE_IMAGE_EDIT_TRIES", "3"))
RELIABLE_IMAGE_EDIT_MIN_SCORE = float(os.getenv("RELIABLE_IMAGE_EDIT_MIN_SCORE", "6.0"))
RELIABLE_IMAGE_EDIT_VERIFY_RUNS = int(os.getenv("RELIABLE_IMAGE_EDIT_VERIFY_RUNS", "1"))
SCENE_EDIT_BEST_OF_N = int(os.getenv("SCENE_EDIT_BEST_OF_N", "3"))
DESIGN_VARIANT_BEST_OF_N = int(os.getenv("DESIGN_VARIANT_BEST_OF_N", "2"))

_JUDGE_MODEL = os.getenv("IMAGE_JUDGE_MODEL", "claude-sonnet-4-6")
_JUDGE_REGION = os.getenv("IMAGE_JUDGE_REGION", "global")


# ============================================================
# 1. Generic image / JSON helpers
# ============================================================

def _require_project_id() -> str:
    if not PROJECT_ID:
        raise RuntimeError(
            "Missing GOOGLE_CLOUD_PROJECT / GOOGLE_PROJECT_ID / PROJECT_ID. "
            "Set it before running image generation or judging."
        )
    return PROJECT_ID


def save_pil_image(
    img: Image.Image,
    run_id: str,
    name: str,
    output_dir: Path = IMAGE_ITERATION_OUTPUT_DIR,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:160] or "image"
    path = output_dir / f"{run_id}_{safe}.png"
    img.convert("RGB").save(path)
    return str(path)


def load_pil(path: str) -> Image.Image:
    cleaned = os.path.expanduser(os.path.expandvars(str(path)))
    return Image.open(cleaned).convert("RGB")


def _pil_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        end = text.rfind("}")
        if end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            return None


def _first_text_from_response(response: Any) -> str:
    chunks: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
    direct = getattr(response, "text", None)
    if direct:
        chunks.append(str(direct))
    return "\n".join(chunks).strip()


def _extract_first_image_from_response(response: Any) -> Image.Image:
    """Extract first image-like inline part from a google-genai response."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                inline_data = getattr(part, "inlineData", None)
            if inline_data is None:
                continue

            data = getattr(inline_data, "data", None)
            if data is None:
                continue
            if isinstance(data, str):
                raw = base64.b64decode(data)
            else:
                raw = bytes(data)
            return Image.open(io.BytesIO(raw)).convert("RGB")

    raise RuntimeError("Gemini response did not contain an inline image part.")


def _side_by_side_png(left: Image.Image, right: Image.Image, max_long_edge: int = 1536) -> bytes:
    h = max(left.height, right.height)
    wl = int(left.width * h / left.height)
    wr = int(right.width * h / right.height)
    combined = Image.new("RGB", (wl + wr + 20, h), (255, 255, 255))
    combined.paste(left.convert("RGB").resize((wl, h)), (0, 0))
    combined.paste(right.convert("RGB").resize((wr, h)), (wl + 20, 0))
    if max(combined.size) > max_long_edge:
        s = max_long_edge / max(combined.size)
        combined = combined.resize((int(combined.width * s), int(combined.height * s)))
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


def _mean_abs_diff(a: Image.Image, b: Image.Image, size: int = 256) -> float:
    import numpy as np

    x = np.asarray(a.convert("RGB").resize((size, size)), np.float32)
    y = np.asarray(b.convert("RGB").resize((size, size)), np.float32)
    return float(abs(x - y).mean())


# ============================================================
# 2. Gemini image helper, replacing facet_pipeline.py
# ============================================================

_genai_client = None
_embedding_client = None
_ANCHOR_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        project_id = _require_project_id()
        from google import genai

        # Gemini image/text generation uses the google-genai client. Keep this global
        # for gemini-3.1-flash-image / gemini image endpoints. Do NOT reuse this
        # for Imagen semantic-mask edits; Imagen uses vertexai.preview + IMAGEN_LOCATION.
        _genai_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=GEMINI_IMAGE_LOCATION,
        )
    return _genai_client


def _pil_to_genai_part(img: Image.Image):
    from google.genai import types

    return types.Part.from_bytes(data=_pil_png_bytes(img), mime_type="image/png")


def _get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        _require_project_id()
        from google import genai

        _embedding_client = genai.Client(vertexai=True, project=PROJECT_ID, location=EMBED_LOCATION)
    return _embedding_client


def generate_text_from_image(prompt: str, image: Image.Image, model: Optional[str] = None) -> str:
    """Small local replacement for the text/vision part of facet_pipeline."""
    client = _get_genai_client()
    response = client.models.generate_content(
        model=model or GEMINI_TEXT_MODEL,
        contents=[prompt, _pil_to_genai_part(image)],
    )
    return _first_text_from_response(response)


def generate_image_from_prompt(
    prompt: str,
    image: Image.Image | List[Image.Image],
    aspect_ratio: str = "4:3",
    model: Optional[str] = None,
) -> Image.Image:
    """
    Gemini image edit/generation helper.

    Supports both:
    - one input room image for scene edits
    - [room_image, product_image] for product/reference-image merge
    """
    client = _get_genai_client()
    from google.genai import types

    images = image if isinstance(image, list) else [image]
    contents: List[Any] = [prompt]
    contents.extend(_pil_to_genai_part(img) for img in images)

    config_kwargs: Dict[str, Any] = {
        "response_modalities": ["TEXT", "IMAGE"],
    }

    # Not all google-genai versions expose aspect_ratio for this config, so try it
    # first and gracefully retry without it if the installed SDK rejects the field.
    try:
        config = types.GenerateContentConfig(**config_kwargs, image_config={"aspect_ratio": aspect_ratio})
    except Exception:  # noqa: BLE001
        config = types.GenerateContentConfig(**config_kwargs)

    try:
        response = client.models.generate_content(
            model=model or GEMINI_IMAGE_MODEL,
            contents=contents,
            config=config,
        )
    except TypeError:
        # SDK compatibility fallback.
        response = client.models.generate_content(
            model=model or GEMINI_IMAGE_MODEL,
            contents=contents,
        )

    return _extract_first_image_from_response(response)


# ============================================================
# 3. Design-variant pipelines + Gemini Embedding 2 facet pipeline
# ============================================================

# Phase-1 exposes only A, B, and E.
VARIANT_PIPELINES = [
    "A_naive",
    "B_vision_described",
    "E_facet_diverse_fanout",
]

DESIGN_VARIANT_OPTION_IDS = {
    "A_naive": "A",
    "B_vision_described": "B",
    "E_facet_diverse_fanout": "E",
}

FACETS = ["style", "color", "material"]

TAXONOMY: Dict[str, List[Dict[str, str]]] = {
    "style": [
        {"label": "Japandi", "description": "Japanese minimalism with Scandinavian warmth, light wood, natural textures, low clutter, calm neutral palette."},
        {"label": "Scandinavian", "description": "Bright functional cozy style, pale woods, soft textiles, simple forms, warm minimalism, natural light."},
        {"label": "Modern Organic", "description": "Contemporary clean forms with earthy materials, warm neutrals, curved lines, wood, stone, linen, tactile natural surfaces."},
        {"label": "Minimalist", "description": "Sparse uncluttered clean geometry, restrained palette, functional furniture, low visual noise."},
        {"label": "Contemporary", "description": "Current refined design, clean but comfortable, polished surfaces, balanced neutrals, simple modern furniture."},
        {"label": "Mid-Century Modern", "description": "Tapered legs, walnut or teak wood, retro-modern silhouettes, simple lines, warm wood tones."},
        {"label": "Bohemian", "description": "Eclectic relaxed style, layered textiles, woven textures, plants, global patterns, artistic mood."},
        {"label": "Industrial", "description": "Black metal, raw wood, exposed materials, warehouse influence, darker palette, utilitarian forms."},
        {"label": "Coastal", "description": "Airy beach-inspired palette, whites, creams, light blues, sandy neutrals, linen, rattan."},
        {"label": "Traditional", "description": "Classic formal furniture, ornate details, balanced symmetry, rich finishes, timeless room composition."},
        {"label": "Modern Farmhouse", "description": "Rustic warmth, wood, black accents, cozy textiles, white and neutral palette, simple practical charm."},
        {"label": "Luxury Modern", "description": "Upscale elegant refined materials, sophisticated palette, polished finishes, premium visual composition."},
    ],
    "color": [
        {"label": "Warm Beige", "description": "Warm beige, creamy neutral, soft tan, warm off-white, cozy neutral base."},
        {"label": "Cream / Ivory", "description": "Cream, ivory, warm white, soft white, light neutral interior palette."},
        {"label": "Light Oak", "description": "Light oak tone, pale honey wood, blonde wood, natural light wood color."},
        {"label": "Walnut Brown", "description": "Walnut brown, medium brown wood, rich warm brown furniture tone."},
        {"label": "Greige", "description": "Greige, gray-beige, muted taupe, balanced warm-cool neutral."},
        {"label": "Soft Gray", "description": "Soft gray, dove gray, light cool neutral with warm undertones."},
        {"label": "Charcoal", "description": "Charcoal, deep gray, near-black, sophisticated dark neutral accent."},
        {"label": "Forest Green", "description": "Forest green, deep botanical green, earthy green accents."},
        {"label": "Muted Sage", "description": "Muted sage green, soft eucalyptus, calm botanical pale green."},
        {"label": "Terracotta", "description": "Terracotta, warm earthy clay tones, soft burnt orange accents."},
        {"label": "Black Accent", "description": "Black accent palette, deep contrast tones, sophisticated graphic dark elements."},
    ],
    "material": [
        {"label": "Light Oak", "description": "Light oak wood, pale honey grain, natural blonde wood surfaces."},
        {"label": "Walnut", "description": "Walnut wood, rich medium-dark grain, warm brown furniture wood."},
        {"label": "Ash Wood", "description": "Ash wood, pale neutral wood with subtle grain, light contemporary wood."},
        {"label": "Linen", "description": "Natural linen textile, breathable fiber, soft drape, matte texture."},
        {"label": "Boucle", "description": "Boucle upholstery, looped soft textured fabric, cozy contemporary feel."},
        {"label": "Velvet", "description": "Velvet fabric, soft pile, rich color depth, luxurious finish."},
        {"label": "Leather", "description": "Leather, smooth or grained, warm tan, cognac or dark premium tactile finish."},
        {"label": "Rattan", "description": "Rattan, woven natural fiber, light airy organic texture."},
        {"label": "Glass", "description": "Glass, transparent or frosted, light reflective clean surface."},
        {"label": "Brass", "description": "Brass, warm metallic gold tone, refined metal accent finish."},
        {"label": "Travertine", "description": "Travertine stone, natural beige stone with pitted texture, contemporary luxury surface."},
        {"label": "Ceramic", "description": "Ceramic tile or pottery, matte or glazed finish, handcrafted surface."},
        {"label": "Linen Curtains", "description": "Natural linen curtains, soft draping panels, contemporary minimalist window treatment."},
        {"label": "Sheer Drapery", "description": "Sheer light filtering drapery, soft translucent fabric, airy window treatment."},
    ],
}


def _architecture_keep_clause() -> str:
    return (
        "Preserve the original room architecture exactly: walls, windows, doors, ceiling, "
        "floor plan, camera angle, and structural openings. Do not remodel the room shell."
    )


def _design_brief_or_default(brief: str) -> str:
    text = (brief or "").strip()
    if text:
        return text
    return (
        "Furnish this room as a comfortable modern living room with a sofa, coffee table, rug, "
        "accent seating, warm lighting, plants, textiles, and tasteful wall decor."
    )


def _describe_room(image: Image.Image) -> str:
    prompt = (
        "Describe this room for an interior-design image editing system. Mention room type, "
        "layout, visible architecture, light, floor/wall materials, and constraints to preserve. "
        "Be concise."
    )
    try:
        return generate_text_from_image(prompt, image)
    except Exception as exc:  # noqa: BLE001
        return f"Room description unavailable because vision description failed: {type(exc).__name__}: {exc}"


def taxonomy_block(facet: str) -> str:
    return "\n".join([f"- {x['label']}: {x['description']}" for x in TAXONOMY[facet]])


GLOBAL_PROMPT = """
You are creating a GLOBAL multimodal embedding for an interior design shopping copilot.
Represent the room image and user intent holistically: room context, broad style,
color palette, visible materials, visual compatibility cues, and product fit.
Do not over-specialize in one facet.
""".strip()


def _make_facet_prompt(facet: str, focus_question: str) -> str:
    return f"""
You are creating a {facet.upper()}-CONDITIONED multimodal embedding.
Represent ONLY the {facet} visible or strongly implied in the room image.

Use this {facet} taxonomy as the candidate space:
{taxonomy_block(facet)}

The embedding should answer: "{focus_question}"
""".strip()


FACET_PROMPTS = {
    "style": _make_facet_prompt("style", "What named interior design style does this room visually express?"),
    "color": _make_facet_prompt("color", "What color palette should matching products and generated concepts follow?"),
    "material": _make_facet_prompt("material", "What materials and finishes should matching products and generated concepts use?"),
}


def _text_part(text: str):
    from google.genai import types
    return types.Part.from_text(text=text)


def _response_to_numpy_embedding(resp: Any) -> np.ndarray:
    if hasattr(resp, "embeddings") and resp.embeddings:
        emb = resp.embeddings[0]
        if hasattr(emb, "values"):
            return np.array(emb.values, dtype=np.float32)
    if hasattr(resp, "embedding"):
        emb = resp.embedding
        if hasattr(emb, "values"):
            return np.array(emb.values, dtype=np.float32)
    raise ValueError(f"Unexpected embedding response format: {type(resp)}")


def embed_text(text: str, output_dimensionality: int = EMBED_DIM) -> np.ndarray:
    from google.genai import types
    content = types.Content(role="user", parts=[_text_part(text)])
    resp = _get_embedding_client().models.embed_content(
        model=EMBED_MODEL,
        contents=[content],
        config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
    )
    return _response_to_numpy_embedding(resp)


def embed_multimodal_prompt_image(
    prompt: str,
    img: Image.Image,
    user_intent: str,
    output_dimensionality: int = EMBED_DIM,
) -> np.ndarray:
    from google.genai import types
    full_prompt = f"{prompt}\n\nUser intent:\n{user_intent}".strip()
    content = types.Content(role="user", parts=[_text_part(full_prompt), _pil_to_genai_part(img)])
    resp = _get_embedding_client().models.embed_content(
        model=EMBED_MODEL,
        contents=[content],
        config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
    )
    return _response_to_numpy_embedding(resp)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def softmax(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32) / max(temperature, 1e-6)
    z = z - np.max(z)
    ex = np.exp(z)
    return ex / np.sum(ex)


def build_facet_embeddings(img: Image.Image, user_intent: str) -> Dict[str, np.ndarray]:
    return {
        facet: embed_multimodal_prompt_image(FACET_PROMPTS[facet], img, user_intent)
        for facet in FACETS
    }


def build_anchor_embeddings() -> Dict[str, Dict[str, Any]]:
    anchor_cache: Dict[str, Dict[str, Any]] = {}
    for facet in FACETS:
        labels, descriptions, vectors = [], [], []
        for item in TAXONOMY[facet]:
            labels.append(item["label"])
            descriptions.append(item["description"])
            anchor_text = f"""
Facet: {facet}
Candidate label: {item['label']}
Candidate description: {item['description']}
This candidate should be compared against a {facet}-conditioned room-image embedding.
""".strip()
            vectors.append(embed_text(anchor_text))
        anchor_cache[facet] = {"labels": labels, "descriptions": descriptions, "vectors": np.vstack(vectors)}
    return anchor_cache


def get_or_build_anchors() -> Dict[str, Dict[str, Any]]:
    global _ANCHOR_CACHE
    if _ANCHOR_CACHE is None:
        _ANCHOR_CACHE = build_anchor_embeddings()
    return _ANCHOR_CACHE


def score_facet_candidates(
    facet: str,
    facet_vec: np.ndarray,
    anchor_cache: Dict[str, Dict[str, Any]],
    top_k: int = TOP_K,
    temperature: float = 0.07,
) -> List[Dict[str, Any]]:
    anchors = anchor_cache[facet]
    sims = np.array([cosine(facet_vec, v) for v in anchors["vectors"]], dtype=np.float32)
    conf = softmax(sims, temperature=temperature)
    rows = []
    for label, desc, sim, c in zip(anchors["labels"], anchors["descriptions"], sims, conf):
        rows.append({"facet": facet, "label": label, "description": desc, "similarity": float(sim), "confidence": float(c)})
    rows.sort(key=lambda r: r["similarity"], reverse=True)
    return rows[:top_k]


def score_all_facets(
    facet_embs: Dict[str, np.ndarray],
    anchor_embs: Dict[str, Dict[str, Any]],
    top_k: int = TOP_K,
) -> Dict[str, List[Dict[str, Any]]]:
    return {facet: score_facet_candidates(facet, facet_embs[facet], anchor_embs, top_k) for facet in FACETS}


def top_labels(scored_facets: Dict[str, List[Dict[str, Any]]], facet: str, n: int) -> List[str]:
    return [r["label"] for r in scored_facets.get(facet, [])[:n]]


def build_style_branch_prompt(
    style_label: str,
    color_labels: List[str],
    material_labels: List[str],
    user_intent: str,
) -> str:
    return textwrap.dedent(f"""
    Use the uploaded room image as the base.

    STRICT PRESERVATION REQUIREMENTS:
    - Every door in the original image must remain in the exact same position, shape, and size.
    - Every window must remain identical: same count, shape, position, and panes.
    - The ceiling must be preserved exactly.
    - Walls and corners must stay in their original positions.
    - Floor material and pattern must remain unchanged.
    - Camera angle and room perspective must be identical.

    Create a STYLE FAN-OUT branch.

    Selected style:
    {style_label}

    Keep these extracted context cues softly in the background:
    Color palette: {", ".join(color_labels)}
    Materials and finishes: {", ".join(material_labels)}

    User intent:
    {user_intent}

    Design instructions:
    - Add the style only through furniture, decor, lighting, textiles, and finishes.
    - Do not remove, replace, or modify architectural elements.
    - Do not create a completely different room.
    - Maintain visual coherence with the original room.
    - Avoid clutter.
    - No text, labels, logos, watermarks, or UI elements.
    - Output a realistic interior design concept render of the same room.
    """).strip()


def build_color_fusion_prompt(
    selected_style: str,
    selected_colors: List[str],
    material_context: List[str],
    user_intent: str,
) -> str:
    return textwrap.dedent(f"""
    Use the uploaded room image as the current selected design branch.

    This is the COLOR FUSION step.

    Preserve the room architecture exactly:
    doors, windows, ceiling, walls, floor, camera angle, and perspective must not change.

    Selected style:
    {selected_style}

    Apply this color palette through furniture, decor, textiles, lighting, and accessories:
    {", ".join(selected_colors)}

    Material context:
    {", ".join(material_context)}

    User intent:
    {user_intent}

    Keep the design realistic, coherent, uncluttered, and residential.
    No text, labels, logos, watermarks, or UI elements.
    """).strip()


def build_material_fusion_prompt(
    selected_style: str,
    selected_colors: List[str],
    selected_materials: List[str],
    user_intent: str,
) -> str:
    return textwrap.dedent(f"""
    Use the uploaded room image as the current selected design branch.

    This is the MATERIAL FUSION step.

    Preserve the room architecture exactly:
    doors, windows, ceiling, walls, floor, camera angle, and perspective must not change.

    Selected style:
    {selected_style}

    Color palette:
    {", ".join(selected_colors)}

    Apply these materials and finishes through furniture, decor, lighting, and textiles:
    {", ".join(selected_materials)}

    User intent:
    {user_intent}

    Keep the design realistic, coherent, uncluttered, and residential.
    No text, labels, logos, watermarks, or UI elements.
    """).strip()


def select_style_labels_for_fanout(
    scored_facets: Dict[str, List[Dict[str, Any]]],
    n_branches: int = STYLE_BRANCHES,
    sampling_strategy: str = "top1_plus_random",
    random_seed: Optional[int] = None,
) -> List[str]:
    if sampling_strategy == "top_n":
        return [r["label"] for r in scored_facets["style"][:n_branches]]
    if sampling_strategy == "top1_plus_random":
        rng = random.Random(random_seed)
        top1 = scored_facets["style"][0]["label"]
        all_styles = [item["label"] for item in TAXONOMY["style"]]
        remaining = [s for s in all_styles if s != top1]
        sampled = rng.sample(remaining, min(n_branches - 1, len(remaining)))
        return [top1] + sampled
    raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")


def generate_style_fanout_branches(
    base_img: Image.Image,
    scored_facets: Dict[str, List[Dict[str, Any]]],
    user_intent: str,
    n_branches: int = STYLE_BRANCHES,
    sampling_strategy: str = "top1_plus_random",
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    styles = select_style_labels_for_fanout(scored_facets, n_branches, sampling_strategy, random_seed)
    color_context = top_labels(scored_facets, "color", COLOR_TOP_N)
    material_context = top_labels(scored_facets, "material", MATERIAL_TOP_N)

    def gen_one(i: int, style_label: str) -> Dict[str, Any]:
        prompt = build_style_branch_prompt(style_label, color_context, material_context, user_intent)
        img = generate_image_from_prompt(prompt, base_img, aspect_ratio="4:3")
        return {"branch_id": i, "style": style_label, "color_context": color_context, "material_context": material_context, "image": img}

    branches: List[Optional[Dict[str, Any]]] = [None] * len(styles)
    max_workers = max(1, min(MAX_CONCURRENT_IMAGE_GEN, len(styles)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(gen_one, i, s): i for i, s in enumerate(styles)}
        for fut in _as_completed(futures):
            idx = futures[fut]
            branches[idx] = fut.result()
    return [b for b in branches if b is not None]


def fuse_color_then_material(
    selected_branch: Dict[str, Any],
    scored_facets: Dict[str, List[Dict[str, Any]]],
    user_intent: str,
) -> Image.Image:
    selected_style = selected_branch["style"]
    selected_colors = top_labels(scored_facets, "color", COLOR_TOP_N)
    selected_materials = top_labels(scored_facets, "material", MATERIAL_TOP_N)

    color_prompt = build_color_fusion_prompt(selected_style, selected_colors, selected_materials[:1], user_intent)
    color_img = generate_image_from_prompt(color_prompt, selected_branch["image"], aspect_ratio="4:3")

    material_prompt = build_material_fusion_prompt(selected_style, selected_colors, selected_materials, user_intent)
    material_img = generate_image_from_prompt(material_prompt, color_img, aspect_ratio="4:3")
    return material_img


def run_full_facet_pipeline(
    image: Image.Image,
    user_intent: str,
    stage2_input: str = "original",
    sampling_strategy: str = "top1_plus_random",
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    anchor_embs = get_or_build_anchors()
    facet_embs = build_facet_embeddings(image, user_intent)
    scored_facets = score_all_facets(facet_embs, anchor_embs, top_k=TOP_K)

    branches = generate_style_fanout_branches(
        image,
        scored_facets,
        user_intent,
        n_branches=STYLE_BRANCHES,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    )
    if not branches:
        raise RuntimeError("Facet pipeline generated no style branches.")

    selected_branch = branches[0]
    if stage2_input == "original":
        fusion_input = dict(selected_branch)
        fusion_input["image"] = image
    elif stage2_input == "branch":
        fusion_input = selected_branch
    else:
        raise ValueError(f"Unknown stage2_input: {stage2_input}")

    final_image = fuse_color_then_material(fusion_input, scored_facets, user_intent)
    return {
        "final_image": final_image,
        "branches": branches,
        "selected_branch": selected_branch,
        "scored_facets": scored_facets,
        "stage2_input": stage2_input,
        "sampling_strategy": sampling_strategy,
    }


def pipeline_a_naive(image: Image.Image, intent: str) -> Image.Image:
    prompt = (
        f"Design/furnish this room according to the user brief: {_design_brief_or_default(intent)}. "
        f"{_architecture_keep_clause()} Keep it photorealistic and shoppable."
    )
    return generate_image_from_prompt(prompt, image, aspect_ratio="4:3")


def pipeline_b_vision_described(image: Image.Image, intent: str) -> Image.Image:
    description = _describe_room(image)
    prompt = (
        f"Original room description: {description}\n\n"
        f"User brief: {_design_brief_or_default(intent)}\n\n"
        "Generate a furnished/decorated design that respects the described architecture and constraints. "
        f"{_architecture_keep_clause()} Keep furniture and decor realistic and shoppable."
    )
    return generate_image_from_prompt(prompt, image, aspect_ratio="4:3")


def pipeline_e_facet_diverse_fanout(image: Image.Image, intent: str) -> Image.Image:
    result = run_full_facet_pipeline(
        image,
        intent,
        stage2_input="original",
        sampling_strategy="top1_plus_random",
        random_seed=42,
    )
    return result["final_image"]


PIPELINES: Dict[str, Callable[[Image.Image, str], Image.Image]] = {
    "A_naive": pipeline_a_naive,
    "B_vision_described": pipeline_b_vision_described,
    "E_facet_diverse_fanout": pipeline_e_facet_diverse_fanout,
}


# ============================================================
# 4. Imagen semantic-mask operations + Gemini scene operations
# ============================================================

CLASS_IDS: Dict[str, int] = {
    "sofa": 62,
    "couch": 62,
    "armchair": 58,
    "chair": 57,
    "table": 67,
    "coffee table": 67,
    "coffee_table": 67,
    "rug": 173,
    "carpet": 173,
    "floormat": 173,
    "plant": 64,
    "potted plant": 64,
    "lamp": 82,
    "floor lamp": 82,
    "cabinet": 90,
    "bookshelf": 76,
    "mirror": 85,
    "floor": 43,
}
ARCHITECTURE = {"wall": 191, "ceiling": 36, "window": 192, "door": 80}


def resolve_target(name: str) -> List[int]:
    key = (name or "").strip().lower()
    if key in CLASS_IDS:
        return [CLASS_IDS[key]]
    for word, cid in CLASS_IDS.items():
        if word in key:
            return [cid]
    raise KeyError(f"Unknown target object '{name}'. Known: {sorted(set(CLASS_IDS))}")


_imagen_model = None


def _get_imagen():
    global _imagen_model
    if _imagen_model is None:
        project_id = _require_project_id()

        # Critical: ImageGenerationModel.from_pretrained(...) does NOT reliably bind
        # the current project/location by itself inside ADK. Without this explicit init,
        # Vertex can fall back to a stale/default context and raise errors such as:
        #   current project 0 / publisher model not visible.
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel

        vertexai.init(project=project_id, location=IMAGEN_LOCATION)
        _imagen_model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL)
    return _imagen_model


def _mk_ref(cls, base_img, reference_id, **extra):
    errs = []
    for kw in ("reference_image", "image"):
        try:
            return cls(reference_id=reference_id, **{kw: base_img}, **extra)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kw}={e}")
    raise RuntimeError(f"{cls.__name__} construction failed: " + " | ".join(errs))


def _pil_to_vimage(img: Image.Image):
    from vertexai.preview.vision_models import Image as VImage

    return VImage(image_bytes=_pil_png_bytes(img))


def _vimage_to_pil(vimg) -> Image.Image:
    for attr in ("_pil_image", "_image_bytes"):
        val = getattr(vimg, attr, None)
        if val is not None:
            return val.convert("RGB") if isinstance(val, Image.Image) else Image.open(io.BytesIO(val)).convert("RGB")
    buf = io.BytesIO()
    vimg.save(location=buf)
    return Image.open(buf).convert("RGB")


def _imagen_edit(
    image: Image.Image,
    classes: List[int],
    prompt: str,
    edit_mode: str = "inpainting-insert",
    dilation: float = 0.02,
) -> Image.Image:
    model = _get_imagen()
    base = _pil_to_vimage(image)
    from vertexai.preview.vision_models import MaskReferenceImage, RawReferenceImage

    raw = _mk_ref(RawReferenceImage, base, reference_id=0)
    mask = _mk_ref(
        MaskReferenceImage,
        None,
        reference_id=1,
        mask_mode="semantic",
        segmentation_classes=classes,
        dilation=dilation,
    )
    out = model.edit_image(
        prompt=prompt,
        edit_mode=edit_mode,
        reference_images=[raw, mask],
        number_of_images=1,
        safety_filter_level="block_some",
        person_generation="dont_allow",
    )
    return _vimage_to_pil(out.images[0])


def op_recolor(image: Image.Image, target: str, color: str, **_: Any) -> Image.Image:
    classes = resolve_target(target)
    if classes == [CLASS_IDS["rug"]]:
        prompt = (
            f"Recolour the area rug to a muted, low-saturation {color}. The rug colour must clearly "
            f"change to {color} across the rug. Keep the rug's shape, size and position. Do not change, "
            "recolour, move, or duplicate any furniture, coffee table, legs, or objects standing on the rug."
        )
        return _imagen_edit(image, classes, prompt, dilation=0.0)
    dil = 0.01 if classes == [CLASS_IDS["armchair"]] else 0.02
    prompt = (
        f"Reupholster this {target} in {color}. The entire {target} must clearly become {color}. "
        "Keep the same shape, size and position. Do not alter anything else."
    )
    return _imagen_edit(image, classes, prompt, dilation=dil)


def op_swap(image: Image.Image, target: str, new_object: str, **_: Any) -> Image.Image:
    return _imagen_edit(
        image,
        resolve_target(target),
        f"Replace this {target} with {new_object}, same position and scale. Keep the rest unchanged.",
    )


def op_restyle_object(image: Image.Image, target: str, style: str, **_: Any) -> Image.Image:
    classes = resolve_target(target)
    dil = 0.01 if classes == [CLASS_IDS["armchair"]] else 0.02
    return _imagen_edit(
        image,
        classes,
        f"Restyle this {target}: {style}. Keep all legs fully visible and intact, keep its footprint "
        "and position, and keep the rest of the room unchanged. Do not crop or remove legs.",
        dilation=dil,
    )


def op_change_material(image: Image.Image, target: str, material: str, **_: Any) -> Image.Image:
    return _imagen_edit(
        image,
        resolve_target(target),
        f"Change the material of this {target} to {material}. Keep the exact same shape, size and position. "
        "Only change the material/texture.",
    )


def _gemini(image: Image.Image, prompt: str) -> Image.Image:
    return generate_image_from_prompt(prompt, image, aspect_ratio="4:3")


_KEEP = (
    "Keep the room architecture, walls, windows, ceiling and any items not mentioned unchanged. "
    "Do not add or duplicate furniture beyond what is explicitly requested."
)
_KEEP_ARCH = (
    "Keep the room architecture — walls, windows, ceiling and floor — unchanged. "
    "Do not add or duplicate furniture."
)


def op_remove(image: Image.Image, target: str, use_gemini: bool = False, **_: Any) -> Image.Image:
    if use_gemini:
        return _gemini(
            image,
            f"Remove the {target} from the room completely and fill the space it occupied with matching "
            "floor, wall and background, so it looks like it was never there. Keep all other furniture "
            "and the architecture unchanged.",
        )
    return _imagen_edit(image, resolve_target(target), prompt="", edit_mode="inpainting-remove", dilation=0.05)


def op_furnish(image: Image.Image, prompt: Optional[str] = None, **_: Any) -> Image.Image:
    p = prompt or (
        "Furnish this empty room into a fully decorated, cozy modern living room with a sofa, coffee table, "
        "rug, floor lamp, armchair, potted plant and wall art. "
        + _KEEP
    )
    return _gemini(image, p)


def op_rearrange(image: Image.Image, instruction: str, **_: Any) -> Image.Image:
    return _gemini(image, instruction + " " + _KEEP_ARCH)


def op_restyle_scene(image: Image.Image, style: str, **_: Any) -> Image.Image:
    return _gemini(image, f"Restyle the whole room in {style} style. Keep the architecture and layout. " + _KEEP)


def op_relight(image: Image.Image, lighting: str, **_: Any) -> Image.Image:
    return _gemini(
        image,
        f"Change ONLY the lighting to {lighting}. Keep all furniture and architecture exactly the same; "
        "only change the light.",
    )


def op_place_product(image: Image.Image, product_image: Image.Image, instruction: str, **_: Any) -> Image.Image:
    return generate_image_from_prompt(instruction + " " + _KEEP, [image, product_image], aspect_ratio="4:3")


OPERATIONS: Dict[str, Dict[str, Any]] = {
    "recolor": {"fn": op_recolor, "backend": "imagen", "deterministic": True},
    "swap": {"fn": op_swap, "backend": "imagen", "deterministic": True},
    "restyle_object": {"fn": op_restyle_object, "backend": "imagen", "deterministic": True},
    "change_material": {"fn": op_change_material, "backend": "imagen", "deterministic": True},
    "remove": {"fn": op_remove, "backend": "imagen", "deterministic": True},
    "furnish": {"fn": op_furnish, "backend": "gemini", "deterministic": False},
    "rearrange": {"fn": op_rearrange, "backend": "gemini", "deterministic": False, "spatial": True},
    "restyle_scene": {"fn": op_restyle_scene, "backend": "gemini", "deterministic": False},
    "relight": {"fn": op_relight, "backend": "gemini", "deterministic": False},
    "place_product": {"fn": op_place_product, "backend": "gemini", "deterministic": False},
}


def apply_operation(
    image: Image.Image,
    op: str,
    best_of_n: int = 1,
    judge_user_prompt: Optional[str] = None,
    **kwargs: Any,
) -> Image.Image:
    if op not in OPERATIONS:
        raise ValueError(f"Unknown op '{op}'. Available: {sorted(OPERATIONS)}")
    spec = OPERATIONS[op]
    fn = spec["fn"]

    if spec["deterministic"] or best_of_n <= 1:
        return fn(image=image, **kwargs)

    candidates = [fn(image=image, **kwargs) for _ in range(best_of_n)]
    up = judge_user_prompt or f"{op}: {kwargs}"

    if spec.get("spatial"):
        scored = []
        for c in candidates:
            s, _info = _spatial_score(image, c, up)
            if _mean_abs_diff(image, c) < _NEAR_IDENTITY_EPS:
                s -= 100.0
            scored.append((s, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    # For non-spatial scene edits, do not call Claude just to choose aesthetics.
    # Reliability verification happens in apply_reliable(...).
    return candidates[0]


# ============================================================
# 5. Claude verifiers used only for required logic
# ============================================================

_anthropic_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _require_project_id()
        from anthropic import AnthropicVertex

        _anthropic_client = AnthropicVertex(region=_JUDGE_REGION, project_id=PROJECT_ID)
    return _anthropic_client


_NEAR_IDENTITY_EPS = 2.0

SPATIAL_VERIFY_PROMPT = """You verify a spatial furniture edit, not aesthetics.

LEFT: room BEFORE edit.
RIGHT: room AFTER edit.
Requested change: "{instruction}"

Return STRICT JSON only:
{{
  "target_moved": <int 0-10>,
  "others_unchanged": <int 0-10>,
  "is_near_identity": <true|false>,
  "note": "one short sentence"
}}"""


def _spatial_score(before: Image.Image, after: Image.Image, instruction: str, runs: int = 1) -> Tuple[float, Dict[str, Any]]:
    png = _side_by_side_png(before, after)
    b64 = base64.b64encode(png).decode("utf-8")
    prompt = SPATIAL_VERIFY_PROMPT.format(instruction=instruction)
    verdicts: List[Dict[str, Any]] = []

    for _ in range(max(1, runs)):
        try:
            client = _get_anthropic()
            msg = client.messages.create(
                model=_JUDGE_MODEL,
                max_tokens=512,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": b64},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            parsed = _parse_json(text)
            if parsed:
                verdicts.append(parsed)
        except Exception as e:  # noqa: BLE001
            print(f"[spatial] {e}")

    if not verdicts:
        return 0.0, {"error": "spatial verifier returned no verdict"}

    def _mean(key: str) -> float:
        vals = [float(v[key]) for v in verdicts if isinstance(v.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    moved = _mean("target_moved")
    stayed = _mean("others_unchanged")
    near = sum(1 for v in verdicts if v.get("is_near_identity") is True) > len(verdicts) / 2
    score = moved + stayed
    if near:
        score -= 100.0
    return score, {"target_moved": moved, "others_unchanged": stayed, "near_identity": near}


VERIFY_PROMPT = """You verify whether a requested interior edit was actually carried out.

LEFT: room BEFORE edit.
RIGHT: room AFTER edit.
Requested edit: "{instruction}"

Judge ONLY whether that specific change was performed correctly. Ignore overall taste.
Return STRICT JSON only:
{{
  "edit_applied": <int 0-10>,
  "side_effects": <int 0-10>,
  "duplicated": <true|false>,
  "note": "one short sentence"
}}"""


def verify_edit(before: Image.Image, after: Image.Image, instruction: str, runs: int = 1) -> Dict[str, Any]:
    png = _side_by_side_png(before, after)
    b64 = base64.b64encode(png).decode("utf-8")
    prompt = VERIFY_PROMPT.format(instruction=instruction)
    verdicts: List[Dict[str, Any]] = []

    for _ in range(max(1, runs)):
        try:
            client = _get_anthropic()
            msg = client.messages.create(
                model=_JUDGE_MODEL,
                max_tokens=400,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": b64},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            parsed = _parse_json(text)
            if parsed:
                verdicts.append(parsed)
        except Exception as e:  # noqa: BLE001
            print(f"[verify] {e}")

    if not verdicts:
        return {
            "score": 0.0,
            "edit_applied": 0.0,
            "side_effects": 0.0,
            "duplicated": False,
            "note": "verifier returned nothing",
        }

    def _mean(key: str) -> float:
        vals = [float(v[key]) for v in verdicts if isinstance(v.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    edit_applied = _mean("edit_applied")
    side_effects = _mean("side_effects")
    duplicated = sum(1 for v in verdicts if v.get("duplicated") is True) > len(verdicts) / 2

    score = edit_applied
    if duplicated:
        score = min(score, 2.0)
    if side_effects < 5:
        score = min(score, side_effects)

    return {
        "score": score,
        "edit_applied": edit_applied,
        "side_effects": side_effects,
        "duplicated": duplicated,
        "note": verdicts[-1].get("note", ""),
    }


def _remove_fallback(image: Image.Image, **kwargs: Any) -> Image.Image:
    return apply_operation(image, "remove", use_gemini=True, **kwargs)


_FALLBACK: Dict[str, Callable[..., Image.Image]] = {}  # strict Imagen object edits; no Gemini fallback


def apply_reliable(
    image: Image.Image,
    op: str,
    goal: str,
    tries: int = 3,
    min_score: float = 6.0,
    verify_runs: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    if op not in OPERATIONS:
        raise ValueError(f"Unknown op '{op}'. Available: {sorted(OPERATIONS)}")

    attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    def _record(label: Any, img: Image.Image, verdict: Dict[str, Any]) -> None:
        nonlocal best
        attempts.append(
            {
                "attempt": label,
                "score": round(float(verdict.get("score", 0.0)), 1),
                "duplicated": bool(verdict.get("duplicated")),
                "note": verdict.get("note", ""),
            }
        )
        if best is None or float(verdict.get("score", 0.0)) > float(best["verdict"].get("score", 0.0)):
            best = {"image": img, "verdict": verdict}

    for idx in range(max(1, tries)):
        img = apply_operation(image, op, **kwargs)
        verdict = verify_edit(image, img, goal, runs=verify_runs)
        _record(idx + 1, img, verdict)
        if verdict["score"] >= min_score and not verdict["duplicated"]:
            return {
                "image": img,
                "verified": True,
                "score": verdict["score"],
                "verdict": verdict,
                "attempts": attempts,
                "used_fallback": False,
            }

    if op in _FALLBACK:
        try:
            img = _FALLBACK[op](image, **kwargs)
            verdict = verify_edit(image, img, goal, runs=verify_runs)
            _record("fallback", img, verdict)
            if verdict["score"] >= min_score and not verdict["duplicated"]:
                return {
                    "image": img,
                    "verified": True,
                    "score": verdict["score"],
                    "verdict": verdict,
                    "attempts": attempts,
                    "used_fallback": True,
                }
        except Exception as e:  # noqa: BLE001
            print(f"[reliable] fallback for '{op}' failed: {e}")

    if best is None:
        raise RuntimeError("Reliable edit failed before any candidate image was produced.")

    return {
        "image": best["image"],
        "verified": False,
        "score": best["verdict"].get("score", 0.0),
        "verdict": best["verdict"],
        "attempts": attempts,
        "used_fallback": any(a["attempt"] == "fallback" for a in attempts),
    }


# ============================================================
# 6. Design session, now local
# ============================================================

class DesignSession:
    """Per-user state for design variant generation and iterative refinement."""

    def __init__(self, room_image: Image.Image):
        self.room: Image.Image = room_image.convert("RGB")
        self.variants: Dict[str, Image.Image] = {}
        self.current: Optional[Image.Image] = None
        self.history: List[Dict[str, Any]] = []

    def generate_variants(self, brief: str, pipelines: Optional[List[str]] = None) -> Dict[str, Image.Image]:
        names = list(pipelines or VARIANT_PIPELINES)
        out: Dict[str, Image.Image] = {}

        def run_one(name: str) -> Tuple[str, Optional[Image.Image]]:
            fn = PIPELINES.get(name)
            if fn is None:
                print(f"[variants] unknown pipeline '{name}', skipping")
                return name, None
            try:
                return name, fn(self.room, brief)
            except Exception as e:  # noqa: BLE001
                print(f"[variants] {name} failed: {e}")
                return name, None

        max_workers = max(1, min(len(names), MAX_CONCURRENT_IMAGE_GEN))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_one, name): name for name in names}
            for fut in _as_completed(futures):
                name, img = fut.result()
                if img is not None:
                    out[name] = img

        # Preserve display order even though execution is parallel.
        self.variants = {name: out[name] for name in names if name in out}
        return self.variants

    def choose(self, pipeline_name: str) -> Image.Image:
        if pipeline_name not in self.variants:
            raise KeyError(f"'{pipeline_name}' not in variants {list(self.variants)}")
        self.current = self.variants[pipeline_name]
        self.history = [{"step": "choose", "variant": pipeline_name}]
        return self.current

    def set_design(self, image: Image.Image) -> Image.Image:
        self.current = image.convert("RGB")
        self.history = [{"step": "set_design"}]
        return self.current

    def edit(self, op: str, goal: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if self.current is None:
            raise RuntimeError("No current design. Call generate_variants()+choose() or set_design() first.")
        prev = self.current
        result = apply_reliable(prev, op, goal=goal or f"{op}: {kwargs}", **kwargs)
        self.current = result["image"]
        self.history.append(
            {
                "step": "edit",
                "op": op,
                "args": {k: ("<PIL.Image>" if isinstance(v, Image.Image) else v) for k, v in kwargs.items()},
                "verified": result.get("verified"),
                "score": result.get("score"),
            }
        )
        return result


# ============================================================
# 7. User-request parser / operation router
# ============================================================

OBJECT_ALIASES = {
    "sofa": ["sofa", "couch"],
    "armchair": ["armchair", "chair", "accent chair"],
    "rug": ["rug", "carpet", "area rug"],
    "table": ["coffee table", "side table", "table"],
    "plant": ["potted plant", "plant"],
    "lamp": ["floor lamp", "table lamp", "lamp"],
    "cabinet": ["media console", "console", "cabinet", "cupboard", "storage cabinet"],
    "bookshelf": ["bookshelf", "bookcase", "shelf"],
    "mirror": ["mirror"],
    "floor": ["flooring", "floor"],
}

MATERIAL_WORDS = [
    "wood",
    "oak",
    "walnut",
    "pine",
    "metal",
    "brass",
    "steel",
    "glass",
    "marble",
    "stone",
    "leather",
    "linen",
    "velvet",
    "boucle",
    "rattan",
    "cane",
    "ceramic",
    "fabric",
]

COLOR_PATTERN = re.compile(
    r"\b(?:to|into|in|make(?: it| the [a-z ]+)?|recolor(?:ed)?(?: to)?)\s+"
    r"([a-z][a-z\- ]{2,40})\b",
    flags=re.IGNORECASE,
)


@dataclass
class EditPlan:
    op: str
    goal: str
    kwargs: Dict[str, Any]
    best_of_n: int = 1
    parser_confidence: float = 0.7

    def to_state(self) -> Dict[str, Any]:
        data = asdict(self)
        safe_kwargs = dict(data.get("kwargs") or {})
        if "product_image" in safe_kwargs:
            safe_kwargs["product_image"] = "<PIL.Image>"
        data["kwargs"] = safe_kwargs
        return data


def infer_target(text: str) -> Optional[str]:
    lower = (text or "").lower()
    candidates: List[Tuple[int, str]] = []
    for canonical, aliases in OBJECT_ALIASES.items():
        for alias in aliases:
            idx = lower.find(alias)
            if idx >= 0:
                candidates.append((idx, canonical))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def infer_material(text: str) -> Optional[str]:
    lower = (text or "").lower()
    for material in MATERIAL_WORDS:
        if re.search(rf"\b{re.escape(material)}\b", lower):
            return material
    return None


def infer_color(text: str) -> Optional[str]:
    lower = (text or "").lower()
    colors = [
        "off white",
        "warm grey",
        "warm gray",
        "white",
        "cream",
        "beige",
        "taupe",
        "brown",
        "walnut",
        "black",
        "charcoal",
        "grey",
        "gray",
        "blue",
        "navy",
        "green",
        "sage",
        "olive",
        "terracotta",
        "rust",
        "red",
        "pink",
        "yellow",
        "gold",
        "brass",
        "mustard",
        "teal",
    ]
    for color in colors:
        if re.search(rf"\b{re.escape(color)}\b", lower):
            return color
    match = COLOR_PATTERN.search(text or "")
    if not match:
        return None
    phrase = match.group(1).strip()
    phrase = re.split(r"\b(?:but|and|while|without|with|keep|preserve)\b", phrase)[0].strip()
    if len(phrase.split()) > 4:
        return None
    return phrase or None


def is_merge_or_product_placement(text: str) -> bool:
    lower = (text or "").lower()
    return any(
        phrase in lower
        for phrase in [
            "place this",
            "put this",
            "add this",
            "use this product",
            "catalog product",
            "merge",
            "combine",
            "from this image",
            "reference image",
            "product image",
        ]
    )


def parse_edit_request(user_query: str, product_image: Optional[Image.Image] = None) -> Optional[EditPlan]:
    text = (user_query or "").strip()
    lower = text.lower()
    if not text:
        return None

    if product_image is not None and is_merge_or_product_placement(text):
        return EditPlan(
            op="place_product",
            goal=f"Place the referenced product into the room: {text}",
            kwargs={"product_image": product_image, "instruction": text, "best_of_n": SCENE_EDIT_BEST_OF_N},
            best_of_n=SCENE_EDIT_BEST_OF_N,
            parser_confidence=0.8,
        )

    target = infer_target(text)

    if re.search(r"\b(remove|delete|get rid of|take out)\b", lower) and target:
        return EditPlan(
            op="remove",
            goal=f"Remove the {target} from the room without changing the rest of the room.",
            kwargs={"target": target},
            parser_confidence=0.9,
        )

    material = infer_material(text)
    if target and material and re.search(r"\b(material|make|change|turn)\b", lower):
        return EditPlan(
            op="change_material",
            goal=f"Change the {target} material to {material}.",
            kwargs={"target": target, "material": material},
            parser_confidence=0.8,
        )

    color = infer_color(text)
    if target and color and re.search(r"\b(recolor|recolour|colour|color|make|turn|change)\b", lower):
        return EditPlan(
            op="recolor",
            goal=f"Recolor the {target} to {color}.",
            kwargs={"target": target, "color": color},
            parser_confidence=0.85,
        )

    if re.search(r"\b(replace|swap)\b", lower) and target:
        match = re.search(r"\b(?:with|to|into)\s+(.+)$", text, flags=re.IGNORECASE)
        new_object = match.group(1).strip(" .") if match else "a visually compatible replacement"
        return EditPlan(
            op="swap",
            goal=f"Replace the {target} with {new_object} while preserving room architecture.",
            kwargs={"target": target, "new_object": new_object},
            parser_confidence=0.75,
        )

    if re.search(r"\b(move|rearrange|shift|swap positions|layout)\b", lower):
        return EditPlan(
            op="rearrange",
            goal=f"Apply the requested spatial layout edit: {text}",
            kwargs={"instruction": text, "best_of_n": SCENE_EDIT_BEST_OF_N},
            best_of_n=SCENE_EDIT_BEST_OF_N,
            parser_confidence=0.75,
        )

    if re.search(r"\b(light|lighting|relight|brighter|darker|evening|night|daylight|warm glow)\b", lower):
        return EditPlan(
            op="relight",
            goal=f"Change only the room lighting as requested: {text}",
            kwargs={"lighting": text, "best_of_n": SCENE_EDIT_BEST_OF_N},
            best_of_n=SCENE_EDIT_BEST_OF_N,
            parser_confidence=0.8,
        )

    if target and re.search(r"\b(restyle|style|make|modern|cozy|japandi|scandinavian|minimal|boho|industrial)\b", lower):
        return EditPlan(
            op="restyle_object",
            goal=f"Restyle the {target} according to: {text}",
            kwargs={"target": target, "style": text},
            parser_confidence=0.65,
        )

    if re.search(r"\b(restyle|redesign|make it|make this|cozier|more cozy|more modern|more minimal|japandi|scandinavian)\b", lower):
        return EditPlan(
            op="restyle_scene",
            goal=f"Restyle the whole room according to: {text}",
            kwargs={"style": text, "best_of_n": SCENE_EDIT_BEST_OF_N},
            best_of_n=SCENE_EDIT_BEST_OF_N,
            parser_confidence=0.65,
        )

    if re.search(r"\b(furnish|decorate|design this room|fill this room)\b", lower):
        return EditPlan(
            op="furnish",
            goal=f"Furnish/decorate the room according to: {text}",
            kwargs={"prompt": text, "best_of_n": SCENE_EDIT_BEST_OF_N},
            best_of_n=SCENE_EDIT_BEST_OF_N,
            parser_confidence=0.75,
        )

    return None



def reset_runtime_clients() -> None:
    """Clear cached SDK clients/models after changing env vars or project/location config."""
    global _genai_client, _embedding_client, _imagen_model, _anthropic_client
    _genai_client = None
    _embedding_client = None
    _imagen_model = None
    _anthropic_client = None


def runtime_location_summary() -> Dict[str, Any]:
    return {
        "PROJECT_ID": PROJECT_ID,
        "GOOGLE_CLOUD_PROJECT_env": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "GOOGLE_PROJECT_ID_env": os.getenv("GOOGLE_PROJECT_ID"),
        "PROJECT_ID_env": os.getenv("PROJECT_ID"),
        "LOCATION": LOCATION,
        "GOOGLE_CLOUD_LOCATION_env": os.getenv("GOOGLE_CLOUD_LOCATION"),
        "GEMINI_IMAGE_LOCATION": GEMINI_IMAGE_LOCATION,
        "GEMINI_IMAGE_MODEL": GEMINI_IMAGE_MODEL,
        "IMAGEN_LOCATION": IMAGEN_LOCATION,
        "IMAGEN_MODEL": IMAGEN_MODEL,
        "IMAGEN_GEN_MODEL": IMAGEN_GEN_MODEL,
        "EMBED_LOCATION": EMBED_LOCATION,
        "EMBED_MODEL": EMBED_MODEL,
        "imagen_object_edit_backend": "vertexai.preview.vision_models",
        "imagen_fallbacks_enabled": bool(_FALLBACK),
    }


def preflight_imagen_access() -> Dict[str, Any]:
    """Validate the exact Imagen SDK binding used by object edits."""
    try:
        reset_runtime_clients()
        project_id = _require_project_id()
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel

        vertexai.init(project=project_id, location=IMAGEN_LOCATION)
        model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL)
        return {
            "ok": True,
            "project_id": project_id,
            "imagen_location": IMAGEN_LOCATION,
            "imagen_model": IMAGEN_MODEL,
            "model_type": type(model).__name__,
            "note": "from_pretrained succeeded after explicit vertexai.init(project, location).",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "project_id": PROJECT_ID,
            "imagen_location": IMAGEN_LOCATION,
            "imagen_model": IMAGEN_MODEL,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": "If your standalone snippet works, ADK is likely importing an old file or cached process. Check vio.__file__ and restart ADK.",
        }

# ============================================================
# 8. Public entry points for VisualPreferenceAgent
# ============================================================

def run_design_variants(
    image: Image.Image,
    source_image_path: Optional[str],
    user_intent: str,
    trigger: str,
    output_dir: Path = IMAGE_ITERATION_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Phase 1: generate A/B/E design variants in parallel. Returns serializable state."""
    run_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    try:
        sess = DesignSession(image)
        variants = sess.generate_variants(user_intent)
    except Exception as exc:  # noqa: BLE001
        return {
            "run_id": run_id,
            "trigger": trigger,
            "mode": "design_variants",
            "ok": False,
            "requested": True,
            "source_image_path": source_image_path,
            "user_intent": user_intent,
            "options": [],
            "errors": [f"Design variant generation failed: {type(exc).__name__}: {exc}"],
            "pending_selection": False,
            "notes": ["Variant generation failed before any selectable image was produced."],
        }

    options: List[Dict[str, Any]] = []
    for pipeline_name, variant_img in variants.items():
        try:
            path = save_pil_image(variant_img, run_id, pipeline_name, output_dir)
            options.append(
                {
                    "pipeline": pipeline_name,
                    "option_id": DESIGN_VARIANT_OPTION_IDS.get(pipeline_name, pipeline_name),
                    "ok": True,
                    "sec": round(time.perf_counter() - t0, 2),
                    "image_path": path,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            options.append(
                {
                    "pipeline": pipeline_name,
                    "option_id": DESIGN_VARIANT_OPTION_IDS.get(pipeline_name, pipeline_name),
                    "ok": False,
                    "sec": round(time.perf_counter() - t0, 2),
                    "image_path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    order = {"A": 0, "B": 1, "E": 2}
    options.sort(key=lambda x: order.get(str(x.get("option_id")), 99))
    ok = any(option.get("ok") for option in options)

    return {
        "run_id": run_id,
        "trigger": trigger,
        "mode": "design_variants",
        "ok": ok,
        "requested": True,
        "source_image_path": source_image_path,
        "user_intent": user_intent,
        "options": options,
        "errors": [f"{o.get('pipeline')}: {o.get('error')}" for o in options if not o.get("ok")],
        "pending_selection": ok,
        "notes": [
            "Generated Phase-1 design variants through self-contained visual_iteration_ops.PIPELINES.",
            "Use variants once for exploration; use verified operations for iterative edits after selection.",
        ],
    }


def run_reliable_edit(
    base_image: Image.Image,
    base_image_path: Optional[str],
    user_query: str,
    product_image: Optional[Image.Image] = None,
    output_dir: Path = IMAGE_ITERATION_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Phase 2: parse request, execute verified operation, save image, return state."""
    run_id = uuid.uuid4().hex[:12]
    plan = parse_edit_request(user_query, product_image=product_image)

    if plan is None:
        return {
            "run_id": run_id,
            "mode": "reliable_edit",
            "ok": False,
            "requested": True,
            "pending_selection": False,
            "source_image_path": base_image_path,
            "user_intent": user_query,
            "options": [],
            "errors": ["Could not map the user request to a supported image operation."],
            "notes": [
                "Supported edits: recolor, swap, restyle_object, change_material, remove, furnish, rearrange, restyle_scene, relight, place_product.",
            ],
        }

    t0 = time.perf_counter()
    try:
        result = apply_reliable(
            base_image,
            plan.op,
            goal=plan.goal,
            tries=RELIABLE_IMAGE_EDIT_TRIES,
            min_score=RELIABLE_IMAGE_EDIT_MIN_SCORE,
            verify_runs=RELIABLE_IMAGE_EDIT_VERIFY_RUNS,
            **plan.kwargs,
        )
        out_path = save_pil_image(result["image"], run_id, f"edit_{plan.op}", output_dir)
        option = {
            "pipeline": f"reliable_{plan.op}",
            "option_id": "EDIT",
            "ok": True,
            "sec": round(time.perf_counter() - t0, 2),
            "image_path": out_path,
            "error": None,
            "verified": bool(result.get("verified")),
            "score": result.get("score"),
            "used_fallback": result.get("used_fallback"),
            "attempts": result.get("attempts", []),
            "verdict": result.get("verdict", {}),
            "edit_plan": plan.to_state(),
        }
        return {
            "run_id": run_id,
            "mode": "reliable_edit",
            "ok": True,
            "requested": True,
            "pending_selection": False,
            "source_image_path": base_image_path,
            "user_intent": user_query,
            "selected_output_image_path": out_path,
            "options": [option],
            "errors": [] if result.get("verified") else ["Edit produced an image but verification did not pass."],
            "notes": [
                "Applied one image operation through local apply_reliable().",
                "Do not claim success to the user unless verified=true.",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "run_id": run_id,
            "mode": "reliable_edit",
            "ok": False,
            "requested": True,
            "pending_selection": False,
            "source_image_path": base_image_path,
            "user_intent": user_query,
            "options": [],
            "errors": [f"Reliable edit failed: {type(exc).__name__}: {exc}"],
            "notes": ["No edited image was saved."],
            "edit_plan": plan.to_state(),
        }
