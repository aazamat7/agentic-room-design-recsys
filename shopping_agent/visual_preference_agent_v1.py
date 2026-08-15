# # shopping_agent/visual_preference_agent.py
# from __future__ import annotations

# import asyncio
# import base64
# import io
# import json
# import logging
# import os
# import random
# import re
# import tempfile
# import textwrap
# import threading
# import time
# import uuid
# from concurrent.futures import ThreadPoolExecutor, as_completed as _cf_as_completed
# from pathlib import Path
# from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

# import numpy as np
# from PIL import Image
# from typing_extensions import override

# from google import genai
# from google.genai import types

# from google.adk.agents import BaseAgent
# from google.adk.agents.invocation_context import InvocationContext
# from google.adk.events import Event, EventActions

# from shopping_agent.tools.visual_preference_extractor import (
#     GeminiEmbedding2VisualPreferenceExtractor,
#     ImageInput,
#     empty_visual_preference_output,
#     extract_text_from_content,
#     find_image_input,
#     safe_json_loads,
# )


# logger = logging.getLogger(__name__)


# # ============================================================
# # 1. Runtime config
# # ============================================================

# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "adsp-s26-reccys")
# EMBED_LOCATION = (
#     os.getenv("GOOGLE_CLOUD_EMBED_LOCATION")
#     or os.getenv("GOOGLE_CLOUD_EMBEDDING_LOCATION")
#     or os.getenv("GOOGLE_CLOUD_LOCATION")
#     or "global"
# )
# GEN_LOCATION = ( "global"
# )

# EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
# IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")
# VISION_DESCRIPTION_MODEL = os.getenv("VISION_DESCRIPTION_MODEL", "gemini-2.5-flash")

# EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# TOP_K = int(os.getenv("TOP_K", "5"))
# STYLE_BRANCHES = int(os.getenv("STYLE_BRANCHES", "4"))
# COLOR_TOP_N = int(os.getenv("COLOR_TOP_N", "3"))
# MATERIAL_TOP_N = int(os.getenv("MATERIAL_TOP_N", "3"))

# IMAGE_ITERATION_OUTPUT_DIR = Path(
#     os.getenv(
#         "IMAGE_ITERATION_OUTPUT_DIR",
#         "shopping_agent/data/generated/image_iterations",
#     )
# )

# ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE = (
#     os.getenv("ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE", "true").lower()
#     in {"1", "true", "yes", "y"}
# )

# MAX_CONCURRENT_IMAGE_GEN = int(os.getenv("MAX_CONCURRENT_IMAGE_GEN", "3"))
# IMAGE_GEN_MAX_RETRIES = int(os.getenv("IMAGE_GEN_MAX_RETRIES", "5"))
# IMAGE_GEN_BASE_DELAY_SEC = float(os.getenv("IMAGE_GEN_BASE_DELAY_SEC", "2.0"))

# _TRANSIENT_IMAGE_GEN_ERRORS = (
#     "429",
#     "quota",
#     "rate",
#     "resource_exhausted",
#     "503",
#     "unavailable",
#     "timeout",
#     "deadline",
#     "overloaded",
#     "500",
#     "internal",
# )

# _embedding_client = None
# _generation_client = None
# _ANCHOR_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
# _GEN_SEMAPHORE: Optional[threading.Semaphore] = None
# _PREWARMED_IMAGE_RUNTIME = False
# _PREWARM_LOCK = threading.Lock()


# # ============================================================
# # 2. Inlined facet taxonomy
# # ============================================================

# FACETS = ["style", "color", "material"]

# TAXONOMY: Dict[str, List[Dict[str, str]]] = {
#     "style": [
#         {"label": "Japandi", "description": "Japanese minimalism with Scandinavian warmth, light wood, natural textures, low clutter, calm neutral palette."},
#         {"label": "Scandinavian", "description": "Bright functional cozy style, pale woods, soft textiles, simple forms, warm minimalism, natural light."},
#         {"label": "Modern Organic", "description": "Contemporary clean forms with earthy materials, warm neutrals, curved lines, wood, stone, linen, tactile natural surfaces."},
#         {"label": "Minimalist", "description": "Sparse uncluttered clean geometry, restrained palette, functional furniture, low visual noise."},
#         {"label": "Contemporary", "description": "Current refined design, clean but comfortable, polished surfaces, balanced neutrals, simple modern furniture."},
#         {"label": "Mid-Century Modern", "description": "Tapered legs, walnut or teak wood, retro-modern silhouettes, simple lines, warm wood tones."},
#         {"label": "Bohemian", "description": "Eclectic relaxed style, layered textiles, woven textures, plants, global patterns, artistic mood."},
#         {"label": "Industrial", "description": "Black metal, raw wood, exposed materials, warehouse influence, darker palette, utilitarian forms."},
#         {"label": "Coastal", "description": "Airy beach-inspired palette, whites, creams, light blues, sandy neutrals, linen, rattan."},
#         {"label": "Traditional", "description": "Classic formal furniture, ornate details, balanced symmetry, rich finishes, timeless room composition."},
#         {"label": "Modern Farmhouse", "description": "Rustic warmth, wood, black accents, cozy textiles, white and neutral palette, simple practical charm."},
#         {"label": "Luxury Modern", "description": "Upscale elegant refined materials, sophisticated palette, polished finishes, premium visual composition."},
#     ],
#     "color": [
#         {"label": "Warm Beige", "description": "Warm beige, creamy neutral, soft tan, warm off-white, cozy neutral base."},
#         {"label": "Cream / Ivory", "description": "Cream, ivory, warm white, soft white, light neutral interior palette."},
#         {"label": "Light Oak", "description": "Light oak tone, pale honey wood, blonde wood, natural light wood color."},
#         {"label": "Walnut Brown", "description": "Walnut brown, medium brown wood, rich warm brown furniture tone."},
#         {"label": "Greige", "description": "Greige, gray-beige, muted taupe, balanced warm-cool neutral."},
#         {"label": "Soft Gray", "description": "Soft gray, dove gray, light cool neutral with warm undertones."},
#         {"label": "Charcoal", "description": "Charcoal, deep gray, near-black, sophisticated dark neutral accent."},
#         {"label": "Forest Green", "description": "Forest green, deep botanical green, earthy green accents."},
#         {"label": "Muted Sage", "description": "Muted sage green, soft eucalyptus, calm botanical pale green."},
#         {"label": "Terracotta", "description": "Terracotta, warm earthy clay tones, soft burnt orange accents."},
#         {"label": "Black Accent", "description": "Black accent palette, deep contrast tones, sophisticated graphic dark elements."},
#     ],
#     "material": [
#         {"label": "Light Oak", "description": "Light oak wood, pale honey grain, natural blonde wood surfaces."},
#         {"label": "Walnut", "description": "Walnut wood, rich medium-dark grain, warm brown furniture wood."},
#         {"label": "Ash Wood", "description": "Ash wood, pale neutral wood with subtle grain, light contemporary wood."},
#         {"label": "Linen", "description": "Natural linen textile, breathable fiber, soft drape, matte texture."},
#         {"label": "Boucle", "description": "Boucle upholstery, looped soft textured fabric, cozy contemporary feel."},
#         {"label": "Velvet", "description": "Velvet fabric, soft pile, rich color depth, luxurious finish."},
#         {"label": "Leather", "description": "Leather, smooth or grained, warm tan, cognac or dark premium tactile finish."},
#         {"label": "Rattan", "description": "Rattan, woven natural fiber, light airy organic texture."},
#         {"label": "Glass", "description": "Glass, transparent or frosted, light reflective clean surface."},
#         {"label": "Brass", "description": "Brass, warm metallic gold tone, refined metal accent finish."},
#         {"label": "Travertine", "description": "Travertine stone, natural beige stone with pitted texture, contemporary luxury surface."},
#         {"label": "Ceramic", "description": "Ceramic tile or pottery, matte or glazed finish, handcrafted surface."},
#         {"label": "Linen Curtains", "description": "Natural linen curtains, soft draping panels, contemporary minimalist window treatment."},
#         {"label": "Sheer Drapery", "description": "Sheer light filtering drapery, soft translucent fabric, airy window treatment."},
#     ],
# }


# # ============================================================
# # 3. User intent detection
# # ============================================================

# MODIFY_IMAGE_PATTERNS = [
#     r"\bmodify\b",
#     r"\biterate\b",
#     r"\biteration\b",
#     r"\bvariation\b",
#     r"\bvariations\b",
#     r"\bredo\b",
#     r"\bremake\b",
#     r"\bretry\b",
#     r"\bchange\b",
#     r"\bedit\b",
#     r"\brevise\b",
#     r"\bmake it\b",
#     r"\bmake this\b",
#     r"\bmake the room\b",
#     r"\bdecorate\b",
#     r"\bfurnish\b",
#     r"\bredesign\b",
#     r"\brestyle\b",
#     r"\brenovate\b",
#     r"\badd\b",
#     r"\bremove\b",
#     r"\breplace\b",
#     r"\bcozier\b",
#     r"\bmore cozy\b",
#     r"\bpet friendly\b",
#     r"\bpet-friendly\b",
#     r"\bbrighter\b",
#     r"\bdarker\b",
#     r"\bwarmer\b",
#     r"\bmore minimal\b",
#     r"\bmore modern\b",
# ]


# def _is_modify_image_intent(user_query: str) -> bool:
#     text = (user_query or "").lower()
#     return any(re.search(pattern, text) for pattern in MODIFY_IMAGE_PATTERNS)


# def _make_generation_intent(user_query: str) -> str:
#     text = (user_query or "").strip()
#     if text:
#         return text

#     return (
#         "Furnish this room as a comfortable residential living room with sofa, "
#         "coffee table, chairs, lighting, textiles, and home decor. Preserve the "
#         "original architecture exactly."
#     )


# # ============================================================
# # 4. Generic state / image helpers
# # ============================================================

# def _as_dict(value: Any) -> Dict[str, Any]:
#     if isinstance(value, dict):
#         return value

#     parsed = safe_json_loads(value)
#     if isinstance(parsed, dict):
#         return parsed

#     return {}


# def _json_dumps(value: Any) -> str:
#     return json.dumps(value, ensure_ascii=False, default=str)


# def _has_successful_visual(value: Any) -> bool:
#     data = _as_dict(value)
#     return bool(data.get("has_image") is True)


# def _clean_path(value: Any) -> str:
#     text = str(value or "").strip()

#     if len(text) >= 2 and (
#         (text[0] == text[-1] == '"')
#         or (text[0] == text[-1] == "'")
#     ):
#         text = text[1:-1].strip()

#     return os.path.expanduser(os.path.expandvars(text))


# def _reuse_previous_visual(previous_visual: Dict[str, Any]) -> Dict[str, Any]:
#     output = dict(previous_visual)
#     notes = list(output.get("notes") or [])
#     note = "No new image provided; reused previous visual preferences for conversational continuity."

#     if note not in notes:
#         notes.append(note)

#     output["notes"] = notes
#     output["reused_from_previous_turn"] = True
#     output["image_available_this_turn"] = False
#     output["visual_extraction_failed_this_turn"] = False

#     return output


# def _save_pil_image(img: Image.Image, run_id: str, name: str) -> str:
#     IMAGE_ITERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
#     safe = safe[:160] or "image"
#     path = IMAGE_ITERATION_OUTPUT_DIR / f"{run_id}_{safe}.png"
#     img.convert("RGB").save(path)
#     return str(path)


# def _load_pil_from_path(path: str) -> Image.Image:
#     return Image.open(_clean_path(path)).convert("RGB")


# def _decode_possible_base64(value: Any) -> Optional[bytes]:
#     if not value:
#         return None

#     if isinstance(value, (bytes, bytearray)):
#         return bytes(value)

#     text = str(value).strip()
#     if not text:
#         return None

#     if "," in text and text.lower().startswith("data:"):
#         text = text.split(",", 1)[1].strip()

#     try:
#         return base64.b64decode(text)
#     except Exception:
#         return None


# def _part_inline_bytes(part: Any) -> Optional[bytes]:
#     if part is None:
#         return None

#     inline_data = getattr(part, "inline_data", None)
#     if inline_data is None:
#         return None

#     return _decode_possible_base64(getattr(inline_data, "data", None))


# def _get_image_input_attr(image_input: ImageInput, *names: str) -> Any:
#     for name in names:
#         value = getattr(image_input, name, None)
#         if value:
#             return value
#     return None


# def _image_input_to_pil_and_path(
#     image_input: ImageInput,
#     run_id: str,
# ) -> Tuple[Image.Image, Optional[str]]:
#     image_path = _get_image_input_attr(
#         image_input,
#         "image_path",
#         "path",
#         "local_path",
#         "input_image_path",
#     )

#     if image_path:
#         path = _clean_path(image_path)
#         return _load_pil_from_path(path), path

#     image_bytes = _get_image_input_attr(image_input, "image_bytes", "bytes", "data")
#     if image_bytes:
#         raw = _decode_possible_base64(image_bytes)
#         if raw:
#             img = Image.open(io.BytesIO(raw)).convert("RGB")
#             saved_path = _save_pil_image(img, run_id, "input_image")
#             return img, saved_path

#     image_base64 = _get_image_input_attr(
#         image_input,
#         "image_base64",
#         "base64",
#         "input_image_base64",
#     )
#     if image_base64:
#         raw = _decode_possible_base64(image_base64)
#         if raw:
#             img = Image.open(io.BytesIO(raw)).convert("RGB")
#             saved_path = _save_pil_image(img, run_id, "input_image")
#             return img, saved_path

#     provided_part = _get_image_input_attr(image_input, "provided_part", "part", "image_part")
#     if provided_part is not None:
#         raw = _part_inline_bytes(provided_part)
#         if raw:
#             img = Image.open(io.BytesIO(raw)).convert("RGB")
#             saved_path = _save_pil_image(img, run_id, "input_image")
#             return img, saved_path

#     gcs_uri = _get_image_input_attr(
#         image_input,
#         "gcs_uri",
#         "image_gcs_uri",
#         "input_image_gcs_uri",
#         "uri",
#     )

#     if gcs_uri:
#         raise ValueError(
#             "Image iteration pipelines A/B/D require a local or inline image. "
#             f"Got GCS URI: {gcs_uri}. Download it locally or pass image_path / "
#             "input_image_base64 for image-generation iteration."
#         )

#     raise ValueError("Could not convert ImageInput to PIL image for image iteration.")


# # ============================================================
# # 5. Inlined Gemini / facet-pipeline utilities
# # ============================================================

# def get_embedding_client():
#     global _embedding_client
#     if _embedding_client is None:
#         _embedding_client = genai.Client(
#             vertexai=True,
#             project=PROJECT_ID,
#             location=EMBED_LOCATION,
#         )
#     return _embedding_client


# def get_generation_client():
#     global _generation_client
#     if _generation_client is None:
#         _generation_client = genai.Client(
#             vertexai=True,
#             project=PROJECT_ID,
#             location=GEN_LOCATION,
#         )
#     return _generation_client


# def normalize_image(img: Image.Image, max_side: int = 1400) -> Image.Image:
#     img = img.convert("RGB")
#     w, h = img.size
#     scale = min(1.0, max_side / max(w, h))

#     if scale < 1.0:
#         img = img.resize((int(w * scale), int(h * scale)))

#     return img


# def image_to_png_bytes(img: Image.Image) -> bytes:
#     buf = io.BytesIO()
#     img.convert("RGB").save(buf, format="PNG")
#     return buf.getvalue()


# def text_part(text: str) -> types.Part:
#     return types.Part.from_text(text=text)


# def image_part_from_pil(img: Image.Image) -> types.Part:
#     img = normalize_image(img)
#     return types.Part.from_bytes(data=image_to_png_bytes(img), mime_type="image/png")


# def response_to_numpy_embedding(resp: Any) -> np.ndarray:
#     if hasattr(resp, "embeddings") and resp.embeddings:
#         emb = resp.embeddings[0]
#         if hasattr(emb, "values"):
#             return np.array(emb.values, dtype=np.float32)

#     if hasattr(resp, "embedding"):
#         emb = resp.embedding
#         if hasattr(emb, "values"):
#             return np.array(emb.values, dtype=np.float32)

#     raise ValueError(f"Unexpected embedding response format: {type(resp)}")


# def embed_text(text: str, output_dimensionality: int = EMBED_DIM) -> np.ndarray:
#     content = types.Content(role="user", parts=[text_part(text)])
#     resp = get_embedding_client().models.embed_content(
#         model=EMBED_MODEL,
#         contents=[content],
#         config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
#     )
#     return response_to_numpy_embedding(resp)


# def embed_multimodal_prompt_image(
#     prompt: str,
#     img: Image.Image,
#     user_intent: str,
#     output_dimensionality: int = EMBED_DIM,
# ) -> np.ndarray:
#     full_prompt = f"{prompt}\n\nUser intent:\n{user_intent}".strip()
#     content = types.Content(
#         role="user",
#         parts=[
#             text_part(full_prompt),
#             image_part_from_pil(img),
#         ],
#     )
#     resp = get_embedding_client().models.embed_content(
#         model=EMBED_MODEL,
#         contents=[content],
#         config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
#     )
#     return response_to_numpy_embedding(resp)


# def l2_normalize(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, dtype=np.float32)
#     return x / (np.linalg.norm(x) + 1e-12)


# def cosine(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.dot(l2_normalize(a), l2_normalize(b)))


# def softmax(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
#     z = np.asarray(x, dtype=np.float32) / max(temperature, 1e-6)
#     z = z - np.max(z)
#     ex = np.exp(z)
#     return ex / np.sum(ex)


# def taxonomy_block(facet: str) -> str:
#     return "\n".join(
#         [f"- {x['label']}: {x['description']}" for x in TAXONOMY[facet]]
#     )


# GLOBAL_PROMPT = """
# You are creating a GLOBAL multimodal embedding for an interior design shopping copilot.
# Represent the room image and user intent holistically: room context, broad style,
# color palette, visible materials, visual compatibility cues, and product fit.
# Do not over-specialize in one facet.
# """.strip()


# def _make_facet_prompt(facet: str, focus_question: str) -> str:
#     return f"""
# You are creating a {facet.upper()}-CONDITIONED multimodal embedding.
# Represent ONLY the {facet} visible or strongly implied in the room image.

# Use this {facet} taxonomy as the candidate space:
# {taxonomy_block(facet)}

# The embedding should answer: "{focus_question}"
# """.strip()


# FACET_PROMPTS = {
#     "style": _make_facet_prompt("style", "What named interior design style does this room visually express?"),
#     "color": _make_facet_prompt("color", "What color palette should matching products and generated concepts follow?"),
#     "material": _make_facet_prompt("material", "What materials and finishes should matching products and generated concepts use?"),
# }


# def build_facet_embeddings(img: Image.Image, user_intent: str) -> Dict[str, np.ndarray]:
#     return {
#         facet: embed_multimodal_prompt_image(FACET_PROMPTS[facet], img, user_intent)
#         for facet in FACETS
#     }


# def build_anchor_embeddings() -> Dict[str, Dict[str, Any]]:
#     anchor_cache: Dict[str, Dict[str, Any]] = {}

#     for facet in FACETS:
#         labels, descriptions, vectors = [], [], []

#         for item in TAXONOMY[facet]:
#             labels.append(item["label"])
#             descriptions.append(item["description"])

#             anchor_text = f"""
# Facet: {facet}
# Candidate label: {item['label']}
# Candidate description: {item['description']}
# This candidate should be compared against a {facet}-conditioned room-image embedding.
# """.strip()

#             vectors.append(embed_text(anchor_text))

#         anchor_cache[facet] = {
#             "labels": labels,
#             "descriptions": descriptions,
#             "vectors": np.vstack(vectors),
#         }

#     return anchor_cache


# def get_or_build_anchors() -> Dict[str, Dict[str, Any]]:
#     global _ANCHOR_CACHE
#     if _ANCHOR_CACHE is None:
#         logger.info("[visual_preference_agent] Building inlined anchor embeddings once.")
#         _ANCHOR_CACHE = build_anchor_embeddings()
#     return _ANCHOR_CACHE


# def score_facet_candidates(
#     facet: str,
#     facet_vec: np.ndarray,
#     anchor_cache: Dict[str, Dict[str, Any]],
#     top_k: int = TOP_K,
#     temperature: float = 0.07,
# ) -> List[Dict[str, Any]]:
#     anchors = anchor_cache[facet]
#     sims = np.array([cosine(facet_vec, v) for v in anchors["vectors"]], dtype=np.float32)
#     conf = softmax(sims, temperature=temperature)

#     rows = []
#     for label, desc, sim, c in zip(
#         anchors["labels"],
#         anchors["descriptions"],
#         sims,
#         conf,
#     ):
#         rows.append(
#             {
#                 "facet": facet,
#                 "label": label,
#                 "description": desc,
#                 "similarity": float(sim),
#                 "confidence": float(c),
#             }
#         )

#     rows.sort(key=lambda r: r["similarity"], reverse=True)
#     return rows[:top_k]


# def score_all_facets(
#     facet_embs: Dict[str, np.ndarray],
#     anchor_embs: Dict[str, Dict[str, Any]],
#     top_k: int = TOP_K,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     return {
#         facet: score_facet_candidates(facet, facet_embs[facet], anchor_embs, top_k)
#         for facet in FACETS
#     }


# def top_labels(scored_facets: Dict[str, List[Dict[str, Any]]], facet: str, n: int) -> List[str]:
#     return [r["label"] for r in scored_facets.get(facet, [])[:n]]


# def _bytes_to_pil_image(data: bytes) -> Image.Image:
#     return Image.open(io.BytesIO(data)).convert("RGB")


# def coerce_to_pil_image(obj: Any) -> Image.Image:
#     if isinstance(obj, Image.Image):
#         return obj.convert("RGB")

#     for attr in ["image_bytes", "bytes", "data"]:
#         value = getattr(obj, attr, None)
#         if value is None:
#             continue
#         try:
#             if isinstance(value, str):
#                 return _bytes_to_pil_image(base64.b64decode(value))
#             if isinstance(value, (bytes, bytearray)):
#                 try:
#                     return _bytes_to_pil_image(bytes(value))
#                 except Exception:
#                     return _bytes_to_pil_image(base64.b64decode(value))
#         except Exception:
#             pass

#     save_method = getattr(obj, "save", None)
#     if callable(save_method):
#         tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#         tmp.close()
#         try:
#             save_method(tmp.name)
#             return Image.open(tmp.name).convert("RGB")
#         finally:
#             try:
#                 os.remove(tmp.name)
#             except Exception:
#                 pass

#     raise TypeError(f"Cannot convert generated image object to PIL: {type(obj)}")


# def part_to_pil_image(part: Any) -> Image.Image:
#     inline_data = getattr(part, "inline_data", None)

#     if inline_data is not None:
#         data = getattr(inline_data, "data", None)
#         if data is not None:
#             if isinstance(data, str):
#                 return _bytes_to_pil_image(base64.b64decode(data))
#             if isinstance(data, (bytes, bytearray)):
#                 try:
#                     return _bytes_to_pil_image(bytes(data))
#                 except Exception:
#                     return _bytes_to_pil_image(base64.b64decode(data))

#     as_image = getattr(part, "as_image", None)
#     if callable(as_image):
#         return coerce_to_pil_image(as_image())

#     raise TypeError(f"Part does not contain a convertible image: {type(part)}")


# def extract_generated_images(response: Any) -> List[Image.Image]:
#     images: List[Image.Image] = []

#     if hasattr(response, "parts") and response.parts:
#         for part in response.parts:
#             if getattr(part, "inline_data", None) is not None or callable(getattr(part, "as_image", None)):
#                 try:
#                     images.append(part_to_pil_image(part))
#                 except Exception as exc:
#                     logger.warning("Could not convert response.parts image part: %s", exc)

#     if not images and hasattr(response, "candidates"):
#         for cand in response.candidates or []:
#             content = getattr(cand, "content", None)
#             for part in getattr(content, "parts", []) or []:
#                 if getattr(part, "inline_data", None) is not None or callable(getattr(part, "as_image", None)):
#                     try:
#                         images.append(part_to_pil_image(part))
#                     except Exception as exc:
#                         logger.warning("Could not convert candidate image part: %s", exc)

#     return images


# def build_image_generation_config(aspect_ratio: str = "4:3"):
#     try:
#         return types.GenerateContentConfig(
#             response_modalities=["IMAGE"],
#             image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
#         )
#     except Exception:
#         return types.GenerateContentConfig(response_modalities=["IMAGE"])


# def _ensure_generation_semaphore() -> threading.Semaphore:
#     global _GEN_SEMAPHORE
#     if _GEN_SEMAPHORE is None:
#         _GEN_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_IMAGE_GEN)
#     return _GEN_SEMAPHORE


# def generate_image_from_prompt(
#     prompt: str,
#     base_img: Image.Image,
#     aspect_ratio: str = "4:3",
# ) -> Image.Image:
#     base_img = normalize_image(base_img)
#     config = build_image_generation_config(aspect_ratio=aspect_ratio)
#     sem = _ensure_generation_semaphore()

#     last_err: Optional[Exception] = None

#     for attempt in range(IMAGE_GEN_MAX_RETRIES):
#         with sem:
#             try:
#                 content = types.Content(
#                     role="user",
#                     parts=[
#                         text_part(prompt),
#                         image_part_from_pil(base_img),
#                     ],
#                 )

#                 response = get_generation_client().models.generate_content(
#                     model=IMAGE_MODEL,
#                     contents=[content],
#                     config=config,
#                 )

#                 images = extract_generated_images(response)
#                 if not images:
#                     raise RuntimeError("No image returned by image generation model.")

#                 return images[0].convert("RGB")

#             except Exception as exc:
#                 last_err = exc
#                 msg = str(exc).lower()
#                 transient = any(k in msg for k in _TRANSIENT_IMAGE_GEN_ERRORS)

#                 if not transient or attempt == IMAGE_GEN_MAX_RETRIES - 1:
#                     raise

#         delay = IMAGE_GEN_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0.0, 1.0)
#         time.sleep(delay)

#     raise last_err or RuntimeError("Unknown image generation failure.")


# def _prewarm_image_runtime() -> None:
#     global _PREWARMED_IMAGE_RUNTIME

#     with _PREWARM_LOCK:
#         if _PREWARMED_IMAGE_RUNTIME:
#             return

#         get_generation_client()
#         get_or_build_anchors()
#         _PREWARMED_IMAGE_RUNTIME = True


# # ============================================================
# # 6. Inlined branching / fusion prompts
# # ============================================================

# def build_style_branch_prompt(
#     style_label: str,
#     color_labels: List[str],
#     material_labels: List[str],
#     user_intent: str,
# ) -> str:
#     return textwrap.dedent(f"""
#     Use the uploaded room image as the base.

#     STRICT PRESERVATION REQUIREMENTS:
#     - Every door in the original image must remain in the exact same position, shape, and size.
#     - Every window must remain identical: same count, shape, position, and panes.
#     - The ceiling must be preserved exactly.
#     - Walls and corners must stay in their original positions.
#     - Floor material and pattern must remain unchanged.
#     - Camera angle and room perspective must be identical.

#     Create a STYLE FAN-OUT branch.

#     Selected style:
#     {style_label}

#     Keep these extracted context cues softly in the background:
#     Color palette: {", ".join(color_labels)}
#     Materials and finishes: {", ".join(material_labels)}

#     User intent:
#     {user_intent}

#     Design instructions:
#     - Add the style only through furniture, decor, lighting, textiles, and finishes.
#     - Do not remove, replace, or modify architectural elements.
#     - Do not create a completely different room.
#     - Maintain visual coherence with the original room.
#     - Avoid clutter.
#     - No text, labels, logos, watermarks, or UI elements.
#     - Output a realistic interior design concept render of the same room.
#     """).strip()


# def build_color_fusion_prompt(
#     selected_style: str,
#     selected_colors: List[str],
#     material_context: List[str],
#     user_intent: str,
# ) -> str:
#     return textwrap.dedent(f"""
#     Use the uploaded room image as the current selected design branch.

#     This is the COLOR FUSION step.

#     Preserve the room architecture exactly:
#     doors, windows, ceiling, walls, floor, camera angle, and perspective must not change.

#     Selected style:
#     {selected_style}

#     Apply this color palette through furniture, decor, textiles, lighting, and accessories:
#     {", ".join(selected_colors)}

#     Material context:
#     {", ".join(material_context)}

#     User intent:
#     {user_intent}

#     Keep the design realistic, coherent, uncluttered, and residential.
#     No text, labels, logos, watermarks, or UI elements.
#     """).strip()


# def build_material_fusion_prompt(
#     selected_style: str,
#     selected_colors: List[str],
#     selected_materials: List[str],
#     user_intent: str,
# ) -> str:
#     return textwrap.dedent(f"""
#     Use the uploaded room image as the current selected design branch.

#     This is the MATERIAL FUSION step.

#     Preserve the room architecture exactly:
#     doors, windows, ceiling, walls, floor, camera angle, and perspective must not change.

#     Selected style:
#     {selected_style}

#     Color palette:
#     {", ".join(selected_colors)}

#     Apply these materials and finishes through furniture, decor, lighting, and textiles:
#     {", ".join(selected_materials)}

#     User intent:
#     {user_intent}

#     Keep the design realistic, coherent, uncluttered, and residential.
#     No text, labels, logos, watermarks, or UI elements.
#     """).strip()


# def select_style_labels_for_fanout(
#     scored_facets: Dict[str, List[Dict[str, Any]]],
#     n_branches: int = STYLE_BRANCHES,
#     sampling_strategy: str = "top_n",
#     random_seed: Optional[int] = None,
# ) -> List[str]:
#     if sampling_strategy == "top_n":
#         return [r["label"] for r in scored_facets["style"][:n_branches]]

#     if sampling_strategy == "top1_plus_random":
#         rng = random.Random(random_seed)
#         top1 = scored_facets["style"][0]["label"]
#         all_styles = [item["label"] for item in TAXONOMY["style"]]
#         remaining = [s for s in all_styles if s != top1]
#         sampled = rng.sample(remaining, min(n_branches - 1, len(remaining)))
#         return [top1] + sampled

#     raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")


# def generate_style_fanout_branches(
#     base_img: Image.Image,
#     scored_facets: Dict[str, List[Dict[str, Any]]],
#     user_intent: str,
#     n_branches: int = STYLE_BRANCHES,
#     sampling_strategy: str = "top_n",
#     random_seed: Optional[int] = None,
# ) -> List[Dict[str, Any]]:
#     styles = select_style_labels_for_fanout(
#         scored_facets,
#         n_branches,
#         sampling_strategy,
#         random_seed,
#     )
#     color_context = top_labels(scored_facets, "color", COLOR_TOP_N)
#     material_context = top_labels(scored_facets, "material", MATERIAL_TOP_N)

#     def gen_one(i: int, style_label: str) -> Dict[str, Any]:
#         prompt = build_style_branch_prompt(style_label, color_context, material_context, user_intent)
#         img = generate_image_from_prompt(prompt, base_img, aspect_ratio="4:3")
#         return {
#             "branch_id": i,
#             "style": style_label,
#             "color_context": color_context,
#             "material_context": material_context,
#             "image": img,
#         }

#     branches: List[Optional[Dict[str, Any]]] = [None] * len(styles)

#     with ThreadPoolExecutor(max_workers=max(1, len(styles))) as ex:
#         futures = {ex.submit(gen_one, i, s): i for i, s in enumerate(styles)}
#         for fut in _cf_as_completed(futures):
#             branches[futures[fut]] = fut.result()

#     return [b for b in branches if b is not None]


# def fuse_color_then_material(
#     selected_branch: Dict[str, Any],
#     scored_facets: Dict[str, List[Dict[str, Any]]],
#     user_intent: str,
# ) -> Image.Image:
#     selected_style = selected_branch["style"]
#     selected_colors = top_labels(scored_facets, "color", COLOR_TOP_N)
#     selected_materials = top_labels(scored_facets, "material", MATERIAL_TOP_N)

#     color_prompt = build_color_fusion_prompt(
#         selected_style,
#         selected_colors,
#         selected_materials[:1],
#         user_intent,
#     )
#     color_img = generate_image_from_prompt(
#         color_prompt,
#         selected_branch["image"],
#         aspect_ratio="4:3",
#     )

#     material_prompt = build_material_fusion_prompt(
#         selected_style,
#         selected_colors,
#         selected_materials,
#         user_intent,
#     )
#     material_img = generate_image_from_prompt(
#         material_prompt,
#         color_img,
#         aspect_ratio="4:3",
#     )

#     return material_img


# def run_full_facet_pipeline(
#     image: Image.Image,
#     user_intent: str,
#     stage2_input: str = "original",
#     sampling_strategy: str = "top_n",
#     random_seed: Optional[int] = 42,
# ) -> Dict[str, Any]:
#     anchor_embs = get_or_build_anchors()
#     facet_embs = build_facet_embeddings(image, user_intent)
#     scored_facets = score_all_facets(facet_embs, anchor_embs, top_k=TOP_K)

#     branches = generate_style_fanout_branches(
#         image,
#         scored_facets,
#         user_intent,
#         n_branches=STYLE_BRANCHES,
#         sampling_strategy=sampling_strategy,
#         random_seed=random_seed,
#     )

#     if not branches:
#         raise RuntimeError("Facet pipeline generated no style branches.")

#     selected_branch = branches[0]

#     if stage2_input == "original":
#         fusion_input = dict(selected_branch)
#         fusion_input["image"] = image
#     elif stage2_input == "branch":
#         fusion_input = selected_branch
#     else:
#         raise ValueError(f"Unknown stage2_input: {stage2_input}")

#     final_image = fuse_color_then_material(fusion_input, scored_facets, user_intent)

#     return {
#         "final_image": final_image,
#         "branches": branches,
#         "selected_branch": selected_branch,
#         "scored_facets": scored_facets,
#         "stage2_input": stage2_input,
#         "sampling_strategy": sampling_strategy,
#     }


# # ============================================================
# # 7. Inlined Pipeline A, B, D
# # ============================================================

# _IMAGE_PIPELINE_DETAILS: Dict[str, Optional[Dict[str, Any]]] = {}


# def _pipeline_A_naive(image: Image.Image, intent: str) -> Image.Image:
#     arch_describe_prompt = (
#         "List the architectural elements visible in this room as a structured description. "
#         "For each element, note its position and any distinctive features.\n"
#         "Format your response as a concise list covering:\n"
#         "- Doors (count, position in room, color, shape)\n"
#         "- Windows (count, position, shape, style)\n"
#         "- Ceiling features (drop ceiling panels, vents, sprinklers, smoke detectors, beams)\n"
#         "- Walls (any alcoves, niches, columns, distinctive features)\n"
#         "- Floor (material, pattern)\n"
#         "Be specific and brief. Output only the description, nothing else."
#     )

#     try:
#         response = get_generation_client().models.generate_content(
#             model=VISION_DESCRIPTION_MODEL,
#             contents=[
#                 types.Content(
#                     role="user",
#                     parts=[
#                         image_part_from_pil(image),
#                         text_part(arch_describe_prompt),
#                     ],
#                 )
#             ],
#         )
#         architecture_description = (response.text or "").strip()
#     except Exception as exc:
#         logger.warning("[pipeline_A] architecture description failed: %s", exc)
#         architecture_description = "(architecture description unavailable — rely on visual input)"

#     prompt = (
#         f"Add furniture, decor, lighting, and textiles to this room.\n\n"
#         f"User intent: {intent}\n\n"
#         f"ARCHITECTURAL ELEMENTS IN THIS ROOM — preserve these EXACTLY:\n"
#         f"{architecture_description}\n\n"
#         f"STRICT PRESERVATION REQUIREMENTS:\n"
#         f"- Every door listed above must remain in the exact same position, shape, and size.\n"
#         f"- Every window must remain identical: same count, shape, position, and panes.\n"
#         f"- All ceiling features must be preserved exactly.\n"
#         f"- Walls, corners, alcoves, and floor pattern must stay unchanged.\n"
#         f"- Camera angle and room perspective must be identical.\n"
#         f"Only add furniture and decor in the open floor area. "
#         f"Do not remove, replace, or modify any architectural element. "
#         f"Return a photorealistic image of the same room with added furniture."
#     )

#     return generate_image_from_prompt(prompt, image, aspect_ratio="4:3")


# def _pipeline_B_vision_described(image: Image.Image, intent: str) -> Image.Image:
#     describe_prompt = (
#         "Analyze this room image and describe in 2-3 sentences:\n"
#         "1. The interior design style visible or implied for empty rooms\n"
#         "2. The dominant color palette\n"
#         "3. The materials and textures present or appropriate\n"
#         "Be concise and specific. Output only the description, nothing else."
#     )

#     try:
#         response = get_generation_client().models.generate_content(
#             model=VISION_DESCRIPTION_MODEL,
#             contents=[
#                 types.Content(
#                     role="user",
#                     parts=[
#                         image_part_from_pil(image),
#                         text_part(describe_prompt),
#                     ],
#                 )
#             ],
#         )
#         style_description = (response.text or "").strip()
#     except Exception as exc:
#         logger.warning("[pipeline_B] description failed: %s", exc)
#         style_description = "modern style with a neutral palette"

#     gen_prompt = (
#         f"Add furniture, decor, lighting, and textiles to this room according to this style:\n"
#         f"{style_description}\n\n"
#         f"User request: {intent}\n\n"
#         f"STRICT PRESERVATION REQUIREMENTS:\n"
#         f"- Every door must remain in the exact same position with the same shape and size.\n"
#         f"- Every window must remain identical.\n"
#         f"- The ceiling must be preserved exactly.\n"
#         f"- Walls, corners, and floor pattern must stay unchanged.\n"
#         f"- Camera angle and room perspective must be identical.\n"
#         f"Only add furniture and decor. Do not modify architectural elements. "
#         f"Return a photorealistic image preserving the original room structure exactly."
#     )

#     return generate_image_from_prompt(gen_prompt, image, aspect_ratio="4:3")


# def _pipeline_D_facet_stage2_original(image: Image.Image, intent: str) -> Image.Image:
#     result = run_full_facet_pipeline(
#         image,
#         intent,
#         stage2_input="original",
#         sampling_strategy="top_n",
#     )
#     _IMAGE_PIPELINE_DETAILS["D_facet_stage2_original"] = result
#     return result["final_image"]


# _INLINE_IMAGE_PIPELINES: Dict[str, Callable[[Image.Image, str], Image.Image]] = {
#     "A_naive": _pipeline_A_naive,
#     "B_vision_described": _pipeline_B_vision_described,
#     "D_facet_stage2_original": _pipeline_D_facet_stage2_original,
# }

# _IMAGE_ITERATION_OPTION_IDS = {
#     "A_naive": "A",
#     "B_vision_described": "B",
#     "D_facet_stage2_original": "D",
# }


# def _serialize_scored_facets_summary(
#     pipeline_details: Optional[Dict[str, Any]],
#     top_n: int = 5,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     if not pipeline_details:
#         return {}

#     scored = pipeline_details.get("scored_facets")
#     if not isinstance(scored, dict):
#         return {}

#     out: Dict[str, List[Dict[str, Any]]] = {}

#     for facet, rows in scored.items():
#         out[str(facet)] = [
#             {
#                 "label": r.get("label"),
#                 "similarity": round(float(r.get("similarity")), 4)
#                 if r.get("similarity") is not None else None,
#                 "confidence": round(float(r.get("confidence")), 4)
#                 if r.get("confidence") is not None else None,
#             }
#             for r in rows[:top_n]
#         ]

#     return out


# # ============================================================
# # 8. A/B/D parallel runner
# # ============================================================

# async def _run_one_inline_image_pipeline(
#     pipeline_name: str,
#     fn: Callable[[Image.Image, str], Image.Image],
#     image: Image.Image,
#     intent: str,
#     run_id: str,
# ) -> Dict[str, Any]:
#     t0 = time.perf_counter()

#     try:
#         generated = await asyncio.to_thread(fn, image.copy(), intent)
#         sec = round(time.perf_counter() - t0, 2)

#         output_path = _save_pil_image(generated, run_id, pipeline_name)

#         result: Dict[str, Any] = {
#             "pipeline": pipeline_name,
#             "option_id": _IMAGE_ITERATION_OPTION_IDS.get(pipeline_name, pipeline_name),
#             "ok": True,
#             "sec": sec,
#             "image_path": output_path,
#             "error": None,
#         }

#         if pipeline_name == "D_facet_stage2_original":
#             details = _IMAGE_PIPELINE_DETAILS.get(pipeline_name)
#             result["scored_facets_summary"] = _serialize_scored_facets_summary(details)

#             if details and "branches" in details:
#                 branch_outputs: List[Dict[str, Any]] = []

#                 for branch in details.get("branches") or []:
#                     branch_id = branch.get("branch_id")
#                     style = branch.get("style")
#                     branch_img = branch.get("image")

#                     if branch_img is None:
#                         continue

#                     branch_path = _save_pil_image(
#                         branch_img,
#                         run_id,
#                         f"{pipeline_name}_branch_{branch_id}_{style}",
#                     )

#                     branch_outputs.append(
#                         {
#                             "branch_id": branch_id,
#                             "style": style,
#                             "image_path": branch_path,
#                             "color_context": branch.get("color_context", []),
#                             "material_context": branch.get("material_context", []),
#                         }
#                     )

#                 result["branches"] = branch_outputs

#         return result

#     except Exception as exc:
#         sec = round(time.perf_counter() - t0, 2)
#         logger.exception("Inline image pipeline failed: %s", pipeline_name)

#         return {
#             "pipeline": pipeline_name,
#             "option_id": _IMAGE_ITERATION_OPTION_IDS.get(pipeline_name, pipeline_name),
#             "ok": False,
#             "sec": sec,
#             "image_path": None,
#             "error": f"{type(exc).__name__}: {exc}",
#         }


# async def _run_abd_image_iteration(
#     image: Image.Image,
#     source_image_path: Optional[str],
#     user_intent: str,
#     trigger: str,
# ) -> Dict[str, Any]:
#     run_id = uuid.uuid4().hex[:12]

#     try:
#         await asyncio.to_thread(_prewarm_image_runtime)
#     except Exception as exc:
#         logger.warning("Image runtime prewarm failed: %s", exc)

#     tasks = [
#         asyncio.create_task(
#             _run_one_inline_image_pipeline(
#                 pipeline_name=name,
#                 fn=fn,
#                 image=image,
#                 intent=user_intent,
#                 run_id=run_id,
#             )
#         )
#         for name, fn in _INLINE_IMAGE_PIPELINES.items()
#     ]

#     options: List[Dict[str, Any]] = []

#     for fut in asyncio.as_completed(tasks):
#         options.append(await fut)

#     order = {"A": 0, "B": 1, "D": 2}
#     options.sort(key=lambda x: order.get(str(x.get("option_id")), 99))

#     ok = any(option.get("ok") for option in options)
#     errors = [
#         f"{option.get('pipeline')}: {option.get('error')}"
#         for option in options
#         if not option.get("ok")
#     ]

#     return {
#         "run_id": run_id,
#         "trigger": trigger,
#         "ok": ok,
#         "requested": True,
#         "source_image_path": source_image_path,
#         "user_intent": user_intent,
#         "options": options,
#         "errors": errors,
#         "pending_selection": ok,
#         "notes": [
#             "Generated image-iteration options A, B, and D in parallel.",
#             "No facet_pipeline.py, live_pipeline.py, or evaluation_pipeline.py import is used.",
#             "A = naive preservation baseline.",
#             "B = vision-described baseline.",
#             "D = inlined facet pipeline with Stage 2 receiving the original image.",
#             "User should select option A, B, or D before product browsing.",
#         ],
#     }


# def _empty_image_iteration_output(reason: str) -> Dict[str, Any]:
#     return {
#         "ok": False,
#         "requested": False,
#         "options": [],
#         "pending_selection": False,
#         "notes": [reason],
#     }


# # ============================================================
# # 9. Selection helpers
# # ============================================================

# def _get_last_iteration_output(state: Dict[str, Any]) -> Dict[str, Any]:
#     return _as_dict(
#         state.get("last_image_iteration_output")
#         or state.get("image_iteration_output")
#     )


# def _get_option_by_letter(
#     iteration_output: Dict[str, Any],
#     option_letter: str,
# ) -> Optional[Dict[str, Any]]:
#     wanted = option_letter.upper()

#     for option in iteration_output.get("options", []) or []:
#         if str(option.get("option_id", "")).upper() == wanted:
#             return option

#     return None


# def _detect_selected_option(
#     user_query: str,
#     state: Dict[str, Any],
# ) -> Optional[Dict[str, Any]]:
#     raw = (user_query or "").strip()
#     lower = raw.lower()

#     option_letter: Optional[str] = None

#     full = re.fullmatch(
#         r"(?:option\s+|pipeline\s+)?([abd])(?:\s+please)?[.!]?",
#         lower,
#     )
#     if full:
#         option_letter = full.group(1).upper()

#     if option_letter is None:
#         explicit = re.search(
#             r"\b(?:choose|pick|select|use|go\s+with)\s+(?:option\s+|pipeline\s+)?([abd])\b",
#             lower,
#         )
#         if explicit:
#             option_letter = explicit.group(1).upper()

#     if option_letter is None:
#         explicit_option = re.search(r"\b(?:option|pipeline)\s+([abd])\b", lower)
#         if explicit_option:
#             option_letter = explicit_option.group(1).upper()

#     if option_letter is None:
#         uppercase = re.search(r"\b([ABD])\b", raw)
#         if uppercase:
#             option_letter = uppercase.group(1).upper()

#     if option_letter is None:
#         return None

#     last_iter = _get_last_iteration_output(state)
#     if not last_iter:
#         return None

#     option = _get_option_by_letter(last_iter, option_letter)
#     if not option:
#         return None

#     return {
#         "selected_option_id": option_letter,
#         "selected_pipeline": option.get("pipeline"),
#         "selected_image_path": option.get("image_path"),
#         "selected_option": option,
#     }


# def _resolve_previous_iteration_base_image(
#     user_query: str,
#     state: Dict[str, Any],
# ) -> Tuple[Optional[Image.Image], Optional[str], Optional[Dict[str, Any]]]:
#     selected = _detect_selected_option(user_query, state)

#     if selected and selected.get("selected_image_path"):
#         path = str(selected["selected_image_path"])
#         if Path(path).exists():
#             return _load_pil_from_path(path), path, selected

#     selected_path = state.get("selected_image_iteration_image_path")
#     if selected_path and Path(str(selected_path)).exists():
#         path = str(selected_path)
#         return _load_pil_from_path(path), path, None

#     last_iter = _get_last_iteration_output(state)
#     source_path = last_iter.get("source_image_path")
#     if source_path and Path(str(source_path)).exists():
#         path = str(source_path)
#         return _load_pil_from_path(path), path, None

#     last_input_path = state.get("last_image_iteration_input_path")
#     if last_input_path and Path(str(last_input_path)).exists():
#         path = str(last_input_path)
#         return _load_pil_from_path(path), path, None

#     previous_visual = _as_dict(
#         state.get("last_visual_preference_output")
#         or state.get("visual_preference_output")
#     )
#     visual_path = previous_visual.get("input_image_local_path")

#     if visual_path and Path(str(visual_path)).exists():
#         path = str(visual_path)
#         return _load_pil_from_path(path), path, None

#     return None, None, selected


# # ============================================================
# # 10. Visual preference extraction runner
# # ============================================================

# async def _extract_visual_preferences_async(
#     image_input: ImageInput,
#     user_query: str,
# ) -> Dict[str, Any]:
#     def _run() -> Dict[str, Any]:
#         extractor = GeminiEmbedding2VisualPreferenceExtractor()
#         return extractor.extract_preferences(
#             image_input=image_input,
#             user_query=user_query,
#         )

#     return await asyncio.to_thread(_run)


# # ============================================================
# # 11. VisualPreferenceAgent
# # ============================================================

# class VisualPreferenceAgent(BaseAgent):
#     @override
#     async def _run_async_impl(
#         self,
#         ctx: InvocationContext,
#     ) -> AsyncGenerator[Event, None]:
#         logger.info("[%s] Starting visual preference / image iteration.", self.name)

#         state = getattr(getattr(ctx, "session", None), "state", {}) or {}

#         previous_visual = _as_dict(
#             state.get("last_visual_preference_output")
#             or state.get("visual_preference_output")
#         )

#         image_input = find_image_input(ctx)
#         user_query = extract_text_from_content(getattr(ctx, "user_content", None))
#         user_intent = _make_generation_intent(user_query)

#         current_turn_output: Dict[str, Any]
#         effective_output: Dict[str, Any]
#         last_visual_output: Optional[Dict[str, Any]] = (
#             previous_visual if _has_successful_visual(previous_visual) else None
#         )

#         image_iteration_requested_this_turn = False
#         image_iteration_output = _empty_image_iteration_output(
#             "No image iteration requested this turn."
#         )

#         selected_iteration_option = _detect_selected_option(user_query, state)

#         source_image_for_generation: Optional[Image.Image] = None
#         source_image_path: Optional[str] = None
#         image_generation_trigger: Optional[str] = None

#         if image_input is not None:
#             conversion_error: Optional[str] = None

#             try:
#                 pil_img, local_input_path = _image_input_to_pil_and_path(
#                     image_input=image_input,
#                     run_id=uuid.uuid4().hex[:12],
#                 )
#                 source_image_for_generation = pil_img
#                 source_image_path = local_input_path
#             except Exception as exc:
#                 conversion_error = (
#                     f"Image was usable for visual extraction but not for A/B/D "
#                     f"image iteration: {type(exc).__name__}: {exc}"
#                 )
#                 logger.warning(conversion_error)

#             extraction_task = asyncio.create_task(
#                 _extract_visual_preferences_async(
#                     image_input=image_input,
#                     user_query=user_query,
#                 )
#             )

#             image_iteration_task: Optional[asyncio.Task[Dict[str, Any]]] = None

#             if ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE and source_image_for_generation is not None:
#                 image_iteration_requested_this_turn = True
#                 image_generation_trigger = "new_image"
#                 image_iteration_task = asyncio.create_task(
#                     _run_abd_image_iteration(
#                         image=source_image_for_generation,
#                         source_image_path=source_image_path,
#                         user_intent=user_intent,
#                         trigger=image_generation_trigger,
#                     )
#                 )

#             try:
#                 extracted = await extraction_task
#                 extracted["image_available_this_turn"] = True
#                 extracted["reused_from_previous_turn"] = False
#                 extracted["visual_extraction_failed_this_turn"] = False

#                 if source_image_path:
#                     extracted["input_image_local_path"] = source_image_path

#                 if conversion_error:
#                     extracted.setdefault("notes", []).append(conversion_error)

#                 current_turn_output = extracted
#                 effective_output = extracted
#                 last_visual_output = extracted

#             except Exception as exc:
#                 reason = (
#                     f"Visual preference extraction failed: "
#                     f"{type(exc).__name__}: {exc}"
#                 )
#                 logger.exception("[%s] Visual extraction failed.", self.name)

#                 failed_output = empty_visual_preference_output(reason=reason)
#                 failed_output["image_available_this_turn"] = True
#                 failed_output["reused_from_previous_turn"] = False
#                 failed_output["visual_extraction_failed_this_turn"] = True

#                 if source_image_path:
#                     failed_output["input_image_local_path"] = source_image_path

#                 if conversion_error:
#                     failed_output.setdefault("notes", []).append(conversion_error)

#                 current_turn_output = failed_output
#                 effective_output = failed_output
#                 last_visual_output = previous_visual if _has_successful_visual(previous_visual) else None

#             if image_iteration_task is not None:
#                 image_iteration_output = await image_iteration_task

#         else:
#             current_turn_output = empty_visual_preference_output(
#                 reason="No image input provided this turn."
#             )
#             current_turn_output["image_available_this_turn"] = False
#             current_turn_output["reused_from_previous_turn"] = False
#             current_turn_output["visual_extraction_failed_this_turn"] = False

#             if _has_successful_visual(previous_visual):
#                 effective_output = _reuse_previous_visual(previous_visual)
#                 last_visual_output = previous_visual
#             else:
#                 effective_output = empty_visual_preference_output(
#                     reason="No image input provided."
#                 )
#                 effective_output["image_available_this_turn"] = False
#                 effective_output["reused_from_previous_turn"] = False
#                 effective_output["visual_extraction_failed_this_turn"] = False
#                 last_visual_output = None

#             if _is_modify_image_intent(user_query):
#                 prev_img, prev_path, selected_for_edit = _resolve_previous_iteration_base_image(
#                     user_query=user_query,
#                     state=state,
#                 )

#                 if prev_img is not None:
#                     source_image_for_generation = prev_img
#                     source_image_path = prev_path
#                     image_iteration_requested_this_turn = True
#                     image_generation_trigger = (
#                         "modify_selected_option"
#                         if selected_for_edit
#                         else "modify_previous_image"
#                     )

#                     image_iteration_output = await _run_abd_image_iteration(
#                         image=source_image_for_generation,
#                         source_image_path=source_image_path,
#                         user_intent=user_intent,
#                         trigger=image_generation_trigger,
#                     )
#                 else:
#                     image_iteration_output = {
#                         "ok": False,
#                         "requested": True,
#                         "pending_selection": False,
#                         "options": [],
#                         "errors": [
#                             "User asked to modify/iterate an image, but no previous "
#                             "local image or generated option was available."
#                         ],
#                         "notes": [
#                             "Ask the user to upload an image or choose an existing option."
#                         ],
#                     }

#         state_delta: Dict[str, Any] = {
#             "visual_preference_output": effective_output,
#             "visual_preference_output_json": _json_dumps(effective_output),
#             "current_turn_visual_preference_output": current_turn_output,
#             "current_turn_visual_preference_output_json": _json_dumps(current_turn_output),
#             "image_iteration_requested_this_turn": image_iteration_requested_this_turn,
#             "current_turn_image_iteration_output": image_iteration_output,
#             "current_turn_image_iteration_output_json": _json_dumps(image_iteration_output),
#             "selected_image_iteration_option": selected_iteration_option or {},
#             "selected_image_iteration_option_json": _json_dumps(selected_iteration_option or {}),
#         }

#         if last_visual_output is not None:
#             state_delta["last_visual_preference_output"] = last_visual_output
#             state_delta["last_visual_preference_output_json"] = _json_dumps(last_visual_output)

#         if image_iteration_output.get("requested") is True:
#             state_delta["image_iteration_output"] = image_iteration_output
#             state_delta["image_iteration_output_json"] = _json_dumps(image_iteration_output)

#             if image_iteration_output.get("ok") is True:
#                 state_delta["last_image_iteration_output"] = image_iteration_output
#                 state_delta["last_image_iteration_output_json"] = _json_dumps(image_iteration_output)
#                 state_delta["image_iteration_pending_selection"] = bool(
#                     image_iteration_output.get("pending_selection")
#                 )

#                 if source_image_path:
#                     state_delta["last_image_iteration_input_path"] = source_image_path

#         if selected_iteration_option:
#             selected_path = selected_iteration_option.get("selected_image_path")
#             selected_pipeline = selected_iteration_option.get("selected_pipeline")

#             if selected_path:
#                 state_delta["selected_image_iteration_image_path"] = selected_path
#                 state_delta["selected_image_iteration_pipeline"] = selected_pipeline
#                 state_delta["active_design_reference_image_path"] = selected_path
#                 state_delta["image_iteration_pending_selection"] = False

#         yield Event(
#             author=self.name,
#             actions=EventActions(state_delta=state_delta),
#         )


# visual_preference_agent = VisualPreferenceAgent(
#     name="VisualPreferenceAgent",
#     description=(
#         "Silently extracts/reuses visual preferences and runs inlined A/B/D image "
#         "iteration pipelines in parallel when an image is provided or the user "
#         "asks to modify an image. Does not import facet_pipeline, live_pipeline, "
#         "or evaluation_pipeline."
#     ),
# )

# shopping_agent/visual_preference_agent.py
# ============================================================
# VisualPreferenceAgent
#
# Silent multimodal state agent.
#
# Responsibilities:
#   1. Extract visual preferences from optional image input.
#   2. Preserve previous visual context across conversational turns.
#   3. Phase 1: when a user uploads a room image, generate selectable
#      design variants through design_session via visual_iteration_ops.
#   4. Phase 2: when a user asks to edit/iterate the selected design,
#      route the request to self-contained visual_iteration_ops
#      instead of rerunning broad A/B/D generation.
#   5. For multi-reference/product placement, use Gemini scene-level merge
#      through visual_iteration_ops.place_product behind local apply_reliable.
#
# Important:
#   This agent is silent. It yields state_delta only.
# ============================================================

# shopping_agent/visual_preference_agent.py
# ============================================================
# VisualPreferenceAgent
#
# Silent multimodal state agent.
#
# Responsibilities:
#   1. Extract visual preferences from optional image input.
#   2. Preserve previous visual context across conversational turns.
#   3. Phase 1: when a user uploads a room image, generate selectable
#      design variants through design_session via visual_iteration_ops.
#   4. Phase 2: when a user asks to edit/iterate the selected design,
#      route the request to self-contained visual_iteration_ops
#      instead of rerunning broad A/B/E generation.
#   5. For multi-reference/product placement, use Gemini scene-level merge
#      through visual_iteration_ops.place_product behind local apply_reliable.
#
# Important:
#   This agent is silent. It yields state_delta only.
# ============================================================

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from PIL import Image
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from shopping_agent.tools import visual_iteration_ops_fixed as vio
from shopping_agent.tools.visual_preference_extractor import (
    GeminiEmbedding2VisualPreferenceExtractor,
    ImageInput,
    empty_visual_preference_output,
    extract_text_from_content,
    find_image_input,
    safe_json_loads,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Runtime config
# ============================================================

ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE = (
    os.getenv("ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE", "true").lower()
    in {"1", "true", "yes", "y"}
)

IMAGE_ITERATION_OUTPUT_DIR = Path(
    os.getenv(
        "IMAGE_ITERATION_OUTPUT_DIR",
        "shopping_agent/data/generated/image_iterations",
    )
)


# ============================================================
# 2. User intent detection
# ============================================================

MODIFY_IMAGE_PATTERNS = [
    r"\bmodify\b",
    r"\biterate\b",
    r"\biteration\b",
    r"\bvariation\b",
    r"\bvariations\b",
    r"\bredo\b",
    r"\bremake\b",
    r"\bretry\b",
    r"\bchange\b",
    r"\bedit\b",
    r"\brevise\b",
    r"\bmake it\b",
    r"\bmake this\b",
    r"\bmake the room\b",
    r"\bdecorate\b",
    r"\bfurnish\b",
    r"\bredesign\b",
    r"\brestyle\b",
    r"\brenovate\b",
    r"\badd\b",
    r"\bremove\b",
    r"\breplace\b",
    r"\bswap\b",
    r"\bmove\b",
    r"\brearrange\b",
    r"\brelight\b",
    r"\blighting\b",
    r"\bplace this\b",
    r"\bput this\b",
    r"\bmerge\b",
    r"\bcombine\b",
    r"\bcozier\b",
    r"\bmore cozy\b",
    r"\bpet friendly\b",
    r"\bpet-friendly\b",
    r"\bbrighter\b",
    r"\bdarker\b",
    r"\bwarmer\b",
    r"\bmore minimal\b",
    r"\bmore modern\b",
]


def _is_modify_image_intent(user_query: str) -> bool:
    text = (user_query or "").lower()
    return any(re.search(pattern, text) for pattern in MODIFY_IMAGE_PATTERNS)


def _make_generation_intent(user_query: str) -> str:
    text = (user_query or "").strip()

    if text:
        return text

    return (
        "Furnish this room as a comfortable residential living room with sofa, "
        "coffee table, chairs, lighting, textiles, and home decor. Preserve the "
        "original architecture exactly."
    )


# ============================================================
# 3. Generic helpers
# ============================================================

def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    parsed = safe_json_loads(value)

    if isinstance(parsed, dict):
        return parsed

    return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _has_successful_visual(value: Any) -> bool:
    data = _as_dict(value)
    return bool(data.get("has_image") is True)


def _clean_path(value: Any) -> str:
    text = str(value or "").strip()

    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()

    text = os.path.expandvars(text)
    text = os.path.expanduser(text)
    return text


def _reuse_previous_visual(previous_visual: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(previous_visual)
    notes = list(output.get("notes") or [])
    note = "No new image provided; reused previous visual preferences for conversational continuity."

    if note not in notes:
        notes.append(note)

    output["notes"] = notes
    output["reused_from_previous_turn"] = True
    output["image_available_this_turn"] = False
    output["visual_extraction_failed_this_turn"] = False
    return output


def _save_pil_image(img: Image.Image, run_id: str, name: str) -> str:
    IMAGE_ITERATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:160] or "image"
    path = IMAGE_ITERATION_OUTPUT_DIR / f"{run_id}_{safe}.png"
    img.convert("RGB").save(path)
    return str(path)


def _load_pil_from_path(path: str) -> Image.Image:
    return Image.open(_clean_path(path)).convert("RGB")


def _decode_possible_base64(value: Any) -> Optional[bytes]:
    if not value:
        return None

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    text = str(value).strip()

    if not text:
        return None

    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1].strip()

    try:
        return base64.b64decode(text)
    except Exception:
        return None


def _part_inline_bytes(part: Any) -> Optional[bytes]:
    inline_data = getattr(part, "inline_data", None)

    if inline_data is None:
        return None

    data = getattr(inline_data, "data", None)
    return _decode_possible_base64(data)


def _get_image_input_attr(image_input: ImageInput, *names: str) -> Any:
    for name in names:
        value = getattr(image_input, name, None)
        if value:
            return value
    return None


def _image_input_to_pil_and_path(
    image_input: ImageInput,
    run_id: str,
) -> Tuple[Image.Image, Optional[str]]:
    """
    Converts ImageInput to PIL image.

    Local paths are preserved. Inline/base64 images are saved locally because
    image_ops/reliable_ops/design_session expect PIL/local image handling.
    """
    image_path = _get_image_input_attr(
        image_input,
        "image_path",
        "path",
        "local_path",
        "input_image_path",
    )

    if image_path:
        path = _clean_path(image_path)
        return _load_pil_from_path(path), path

    image_bytes = _get_image_input_attr(
        image_input,
        "image_bytes",
        "bytes",
        "data",
    )

    if image_bytes:
        raw = _decode_possible_base64(image_bytes)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    image_base64 = _get_image_input_attr(
        image_input,
        "image_base64",
        "base64",
        "input_image_base64",
    )

    if image_base64:
        raw = _decode_possible_base64(image_base64)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    provided_part = _get_image_input_attr(
        image_input,
        "provided_part",
        "part",
        "image_part",
    )

    if provided_part is not None:
        raw = _part_inline_bytes(provided_part)
        if raw:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            saved_path = _save_pil_image(img, run_id, "input_image")
            return img, saved_path

    gcs_uri = _get_image_input_attr(
        image_input,
        "gcs_uri",
        "image_gcs_uri",
        "input_image_gcs_uri",
        "uri",
    )

    if gcs_uri:
        raise ValueError(
            "Image iteration requires a local or inline image. "
            f"Got GCS URI: {gcs_uri}. Download it locally or pass image_path / input_image_base64."
        )

    raise ValueError("Could not convert ImageInput to PIL image for image iteration.")


def _empty_image_iteration_output(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "requested": False,
        "mode": "none",
        "options": [],
        "pending_selection": False,
        "notes": [reason],
    }


# ============================================================
# 4. Selection / previous image helpers
# ============================================================

def _get_last_iteration_output(state: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(state.get("last_image_iteration_output") or state.get("image_iteration_output"))


def _get_option_by_letter(iteration_output: Dict[str, Any], option_letter: str) -> Optional[Dict[str, Any]]:
    wanted = option_letter.upper()

    for option in iteration_output.get("options", []) or []:
        if str(option.get("option_id", "")).upper() == wanted:
            return option

    return None


def _detect_selected_option(user_query: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detects user selection of A/B/E without treating article "a" as option A.
    """
    raw = (user_query or "").strip()
    lower = raw.lower()
    option_letter: Optional[str] = None

    full = re.fullmatch(r"(?:option\s+|pipeline\s+)?([abe])(?:\s+please)?[.!]?", lower)

    if full:
        option_letter = full.group(1).upper()

    if option_letter is None:
        explicit = re.search(
            r"\b(?:choose|pick|select|use|go\s+with)\s+(?:option\s+|pipeline\s+)?([abe])\b",
            lower,
        )
        if explicit:
            option_letter = explicit.group(1).upper()

    if option_letter is None:
        explicit_option = re.search(r"\b(?:option|pipeline)\s+([abe])\b", lower)
        if explicit_option:
            option_letter = explicit_option.group(1).upper()

    if option_letter is None:
        uppercase = re.search(r"\b([ABE])\b", raw)
        if uppercase:
            option_letter = uppercase.group(1).upper()

    if option_letter is None:
        return None

    last_iter = _get_last_iteration_output(state)
    option = _get_option_by_letter(last_iter, option_letter)

    if not option:
        return None

    return {
        "selected_option_id": option_letter,
        "selected_pipeline": option.get("pipeline"),
        "selected_image_path": option.get("image_path"),
        "selected_option": option,
    }


def _resolve_previous_iteration_base_image(
    user_query: str,
    state: Dict[str, Any],
) -> Tuple[Optional[Image.Image], Optional[str], Optional[Dict[str, Any]]]:
    """
    Base image priority for edits:
      1. Explicit selected option A/B/E in the current user text.
      2. Last selected/generated active design.
      3. Last image-iteration source.
      4. Last successful visual input path.
    """
    selected = _detect_selected_option(user_query, state)

    if selected and selected.get("selected_image_path"):
        path = str(selected["selected_image_path"])
        if Path(path).exists():
            return _load_pil_from_path(path), path, selected

    for key in (
        "selected_image_iteration_image_path",
        "active_design_reference_image_path",
    ):
        selected_path = state.get(key)
        if selected_path and Path(str(selected_path)).exists():
            path = str(selected_path)
            return _load_pil_from_path(path), path, None

    last_iter = _get_last_iteration_output(state)
    for key in ("selected_output_image_path", "source_image_path"):
        source_path = last_iter.get(key)
        if source_path and Path(str(source_path)).exists():
            path = str(source_path)
            return _load_pil_from_path(path), path, None

    last_input_path = state.get("last_image_iteration_input_path")
    if last_input_path and Path(str(last_input_path)).exists():
        path = str(last_input_path)
        return _load_pil_from_path(path), path, None

    previous_visual = _as_dict(state.get("last_visual_preference_output") or state.get("visual_preference_output"))
    visual_path = previous_visual.get("input_image_local_path")

    if visual_path and Path(str(visual_path)).exists():
        path = str(visual_path)
        return _load_pil_from_path(path), path, None

    return None, None, selected


# ============================================================
# 5. Visual preference extraction runner
# ============================================================

async def _extract_visual_preferences_async(
    image_input: ImageInput,
    user_query: str,
) -> Dict[str, Any]:
    def _run() -> Dict[str, Any]:
        extractor = GeminiEmbedding2VisualPreferenceExtractor()
        return extractor.extract_preferences(
            image_input=image_input,
            user_query=user_query,
        )

    return await asyncio.to_thread(_run)


# ============================================================
# 6. VisualPreferenceAgent
# ============================================================

class VisualPreferenceAgent(BaseAgent):
    """
    Silent visual context + image iteration agent.

    It writes:
      - visual_preference_output for PlannerAgent
      - image_iteration_output for design variants or reliable edits
      - selected_image_iteration_* state when user chooses a variant or an edit completes
    """

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        logger.info("[%s] Starting visual preference / image iteration.", self.name)

        state = dict(getattr(getattr(ctx, "session", None), "state", {}) or {})
        previous_visual = _as_dict(state.get("last_visual_preference_output") or state.get("visual_preference_output"))

        image_input = find_image_input(ctx)
        user_query = extract_text_from_content(getattr(ctx, "user_content", None))
        user_intent = _make_generation_intent(user_query)

        current_turn_output: Dict[str, Any]
        effective_output: Dict[str, Any]
        last_visual_output: Optional[Dict[str, Any]] = previous_visual if _has_successful_visual(previous_visual) else None

        image_iteration_requested_this_turn = False
        image_iteration_output = _empty_image_iteration_output("No image iteration requested this turn.")
        selected_iteration_option = _detect_selected_option(user_query, state)

        source_image_for_generation: Optional[Image.Image] = None
        source_image_path: Optional[str] = None
        image_generation_trigger: Optional[str] = None

        # ------------------------------------------------------------
        # Case 1: new image provided this turn.
        # Usually this starts Phase 1 variant generation. If the user is
        # explicitly merging/placing a reference product into the previous
        # design, the new image is treated as product_image instead.
        # ------------------------------------------------------------
        if image_input is not None:
            conversion_error: Optional[str] = None
            uploaded_pil: Optional[Image.Image] = None
            uploaded_path: Optional[str] = None

            try:
                uploaded_pil, uploaded_path = _image_input_to_pil_and_path(
                    image_input=image_input,
                    run_id=uuid.uuid4().hex[:12],
                )
                source_image_for_generation = uploaded_pil
                source_image_path = uploaded_path
            except Exception as exc:  # noqa: BLE001
                conversion_error = (
                    "Image was usable for visual extraction but not for image iteration: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.warning(conversion_error)

            extraction_task = asyncio.create_task(
                _extract_visual_preferences_async(
                    image_input=image_input,
                    user_query=user_query,
                )
            )

            image_iteration_task: Optional[asyncio.Task[Dict[str, Any]]] = None

            is_merge_with_reference = (
                uploaded_pil is not None
                and _is_modify_image_intent(user_query)
                and vio.is_merge_or_product_placement(user_query)
            )

            if is_merge_with_reference:
                base_img, base_path, selected_for_edit = _resolve_previous_iteration_base_image(user_query, state)
                if base_img is not None:
                    selected_iteration_option = selected_iteration_option or selected_for_edit
                    source_image_for_generation = base_img
                    source_image_path = base_path
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "merge_reference_image"
                    image_iteration_task = asyncio.create_task(
                        asyncio.to_thread(
                            vio.run_reliable_edit,
                            base_img,
                            base_path,
                            user_query,
                            uploaded_pil,
                        )
                    )
                else:
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "merge_reference_image_missing_base"
                    image_iteration_output = {
                        "ok": False,
                        "requested": True,
                        "mode": "reliable_edit",
                        "pending_selection": False,
                        "options": [],
                        "errors": [
                            "The user uploaded a reference/product image to merge, but no active room/design image was available."
                        ],
                        "notes": [
                            "Ask the user to first upload a room image or choose a generated design option."
                        ],
                    }

            elif ENABLE_IMAGE_ITERATION_ON_NEW_IMAGE and source_image_for_generation is not None:
                image_iteration_requested_this_turn = True
                image_generation_trigger = "new_image"
                image_iteration_task = asyncio.create_task(
                    asyncio.to_thread(
                        vio.run_design_variants,
                        source_image_for_generation,
                        source_image_path,
                        user_intent,
                        image_generation_trigger,
                    )
                )

            try:
                extracted = await extraction_task
                extracted["image_available_this_turn"] = True
                extracted["reused_from_previous_turn"] = False
                extracted["visual_extraction_failed_this_turn"] = False

                if uploaded_path:
                    extracted["input_image_local_path"] = uploaded_path

                if conversion_error:
                    extracted.setdefault("notes", []).append(conversion_error)

                if is_merge_with_reference:
                    extracted.setdefault("notes", []).append(
                        "This uploaded image was interpreted as a reference/product image for an edit."
                    )

                current_turn_output = extracted
                effective_output = extracted
                last_visual_output = extracted

            except Exception as exc:  # noqa: BLE001
                reason = f"Visual preference extraction failed: {type(exc).__name__}: {exc}"
                logger.exception("[%s] Visual extraction failed.", self.name)

                failed_output = empty_visual_preference_output(reason=reason)
                failed_output["image_available_this_turn"] = True
                failed_output["reused_from_previous_turn"] = False
                failed_output["visual_extraction_failed_this_turn"] = True

                if uploaded_path:
                    failed_output["input_image_local_path"] = uploaded_path

                if conversion_error:
                    failed_output.setdefault("notes", []).append(conversion_error)

                current_turn_output = failed_output
                effective_output = failed_output
                last_visual_output = previous_visual if _has_successful_visual(previous_visual) else None

            if image_iteration_task is not None:
                image_iteration_output = await image_iteration_task

        # ------------------------------------------------------------
        # Case 2: no new image this turn.
        # This is where selected-option edits become reliable_ops edits.
        # ------------------------------------------------------------
        else:
            current_turn_output = empty_visual_preference_output(reason="No image input provided this turn.")
            current_turn_output["image_available_this_turn"] = False
            current_turn_output["reused_from_previous_turn"] = False
            current_turn_output["visual_extraction_failed_this_turn"] = False

            if _has_successful_visual(previous_visual):
                effective_output = _reuse_previous_visual(previous_visual)
                last_visual_output = previous_visual
            else:
                effective_output = empty_visual_preference_output(reason="No image input provided.")
                effective_output["image_available_this_turn"] = False
                effective_output["reused_from_previous_turn"] = False
                effective_output["visual_extraction_failed_this_turn"] = False
                last_visual_output = None

            if _is_modify_image_intent(user_query):
                prev_img, prev_path, selected_for_edit = _resolve_previous_iteration_base_image(user_query, state)

                if prev_img is not None:
                    selected_iteration_option = selected_iteration_option or selected_for_edit
                    source_image_for_generation = prev_img
                    source_image_path = prev_path
                    image_iteration_requested_this_turn = True
                    image_generation_trigger = "modify_selected_option" if selected_for_edit else "modify_previous_image"

                    image_iteration_output = await asyncio.to_thread(
                        vio.run_reliable_edit,
                        source_image_for_generation,
                        source_image_path,
                        user_query,
                        None,
                    )
                else:
                    image_iteration_output = {
                        "ok": False,
                        "requested": True,
                        "mode": "reliable_edit",
                        "pending_selection": False,
                        "options": [],
                        "errors": [
                            "User asked to modify/iterate an image, but no previous local image or generated option was available."
                        ],
                        "notes": ["Ask the user to upload an image or choose an existing option first."],
                    }

        edited_path = image_iteration_output.get("selected_output_image_path")
        if edited_path:
            selected_iteration_option = {
                "selected_option_id": "EDIT",
                "selected_pipeline": image_iteration_output.get("mode"),
                "selected_image_path": edited_path,
                "selected_option": (image_iteration_output.get("options") or [{}])[0],
            }

        logger.info(
            "[%s] visual_has_image=%s reused=%s image_iter_requested=%s mode=%s options=%s selected=%s",
            self.name,
            effective_output.get("has_image"),
            effective_output.get("reused_from_previous_turn"),
            image_iteration_requested_this_turn,
            image_iteration_output.get("mode"),
            len(image_iteration_output.get("options", []) or []),
            bool(selected_iteration_option),
        )

        # ------------------------------------------------------------
        # State update
        # ------------------------------------------------------------
        state_delta: Dict[str, Any] = {
            "visual_preference_output": effective_output,
            "visual_preference_output_json": _json_dumps(effective_output),
            "current_turn_visual_preference_output": current_turn_output,
            "current_turn_visual_preference_output_json": _json_dumps(current_turn_output),
            "image_iteration_requested_this_turn": image_iteration_requested_this_turn,
            "current_turn_image_iteration_output": image_iteration_output,
            "current_turn_image_iteration_output_json": _json_dumps(image_iteration_output),
            "selected_image_iteration_option": selected_iteration_option or {},
            "selected_image_iteration_option_json": _json_dumps(selected_iteration_option or {}),
        }

        if last_visual_output is not None:
            state_delta["last_visual_preference_output"] = last_visual_output
            state_delta["last_visual_preference_output_json"] = _json_dumps(last_visual_output)

        if image_iteration_output.get("requested") is True:
            state_delta["image_iteration_output"] = image_iteration_output
            state_delta["image_iteration_output_json"] = _json_dumps(image_iteration_output)

            if image_iteration_output.get("ok") is True:
                state_delta["last_image_iteration_output"] = image_iteration_output
                state_delta["last_image_iteration_output_json"] = _json_dumps(image_iteration_output)
                state_delta["image_iteration_pending_selection"] = bool(image_iteration_output.get("pending_selection"))

                if source_image_path:
                    state_delta["last_image_iteration_input_path"] = source_image_path

                if edited_path:
                    state_delta["selected_image_iteration_image_path"] = edited_path
                    state_delta["active_design_reference_image_path"] = edited_path
                    state_delta["last_reliable_image_edit_output"] = image_iteration_output
                    state_delta["last_reliable_image_edit_output_json"] = _json_dumps(image_iteration_output)
                    state_delta["image_iteration_pending_selection"] = False

        if selected_iteration_option:
            selected_path = selected_iteration_option.get("selected_image_path")
            selected_pipeline = selected_iteration_option.get("selected_pipeline")

            if selected_path:
                state_delta["selected_image_iteration_image_path"] = selected_path
                state_delta["selected_image_iteration_pipeline"] = selected_pipeline
                state_delta["active_design_reference_image_path"] = selected_path
                state_delta["image_iteration_pending_selection"] = False

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )


visual_preference_agent = VisualPreferenceAgent(
    name="VisualPreferenceAgent",
    description=(
        "Silently extracts/reuses visual preferences, generates initial design variants, "
        "and applies verified Imagen/Gemini image edits through self-contained visual_iteration_ops."
    ),
)

