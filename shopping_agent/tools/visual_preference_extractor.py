# # shopping_agent/tools/visual_preference_extractor.py
# # ============================================================
# # Visual Preference Extraction using Gemini Embedding 2.
# #
# # Purpose:
# #   Convert an uploaded room/product inspiration image into structured
# #   shopping preferences for the PlannerAgent.
# #
# # Method:
# #   1. Embed the image using Vertex AI Gemini Embedding 2.
# #   2. Embed curated interior-design preference labels/prompts.
# #   3. Rank labels by cosine similarity to the image embedding.
# #   4. Return structured preferences:
# #      styles, colors, materials, room/use case, product clues.
# #
# # This is embedding-based preference extraction, not generative captioning.
# # It does not invent products or URLs.
# #
# # Env:
# #   GOOGLE_GENAI_USE_VERTEXAI=TRUE
# #   GOOGLE_CLOUD_PROJECT=...
# #   GOOGLE_CLOUD_LOCATION=us-central1 or us
# #   GOOGLE_CLOUD_EMBEDDING_LOCATION=us  # optional override
# #   VISUAL_EMBEDDING_MODEL=gemini-embedding-2
# #   VISUAL_EMBEDDING_DIM=256
# # ============================================================

# from __future__ import annotations

# import base64
# import json
# import mimetypes
# import os
# import re
# from dataclasses import dataclass
# from functools import lru_cache
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types


# # ============================================================
# # 1. Preference taxonomy
# # ============================================================

# PREFERENCE_TAXONOMY: Dict[str, Dict[str, str]] = {
#     "styles": {
#         "minimalist": "minimalist interior design, clean lines, sparse decor, uncluttered room",
#         "modern": "modern interior design, contemporary furniture, clean silhouettes",
#         "mid century modern": "mid century modern furniture, warm wood, tapered legs, retro modern room",
#         "scandinavian": "Scandinavian interior design, light wood, cozy minimalism, bright neutral room",
#         "japandi": "Japandi interior design, Japanese Scandinavian fusion, natural wood, low contrast, calm neutral room",
#         "boho": "bohemian interior design, layered textiles, woven decor, plants, eclectic warm room",
#         "industrial": "industrial interior design, metal, black accents, exposed materials, urban loft",
#         "farmhouse": "farmhouse interior design, rustic wood, cozy traditional, natural textures",
#         "coastal": "coastal interior design, airy light colors, linen, rattan, beach inspired decor",
#         "traditional": "traditional interior design, classic furniture, ornate details, formal room",
#         "glam": "glam interior design, gold accents, velvet, polished surfaces, dramatic decor",
#         "organic modern": "organic modern interior design, natural stone, wood, curved furniture, earthy neutral palette",
#     },
#     "colors": {
#         "warm neutrals": "warm neutral color palette, beige, cream, ivory, tan, soft brown",
#         "cool neutrals": "cool neutral color palette, white, gray, charcoal, black accents",
#         "earth tones": "earth tone palette, terracotta, clay, olive, brown, sand, rust",
#         "black accents": "black accent decor, black metal, contrast details",
#         "white and cream": "white and cream room palette, light bright airy decor",
#         "beige and tan": "beige tan room palette, warm soft neutral furniture",
#         "green accents": "green accents, sage green, olive green, plants in room",
#         "blue accents": "blue accents, navy, light blue, coastal blue decor",
#         "gold accents": "gold brass metallic accents in decor",
#         "colorful eclectic": "colorful eclectic room, multiple saturated colors, playful decor",
#     },
#     "materials": {
#         "natural wood": "natural wood furniture, oak, walnut, ash, warm wood grain",
#         "rattan and cane": "rattan cane wicker furniture, woven natural fibers",
#         "metal": "metal furniture or decor, black metal, steel, iron, brass",
#         "marble or stone": "marble stone travertine surfaces, stone coffee table or decor",
#         "linen fabric": "linen upholstery, soft woven natural fabric furniture",
#         "boucle fabric": "boucle upholstery, nubby textured fabric, cozy white chair or sofa",
#         "leather": "leather upholstery, leather sofa or chair",
#         "glass": "glass table or transparent glass decor",
#         "ceramic": "ceramic decor, ceramic vase, pottery, handmade clay accessories",
#         "jute or sisal": "jute sisal natural fiber rug, woven rug texture",
#     },
#     "room_or_use_case": {
#         "living room": "living room with sofa, coffee table, rug, side table, media console",
#         "dining room": "dining room with dining table, dining chairs, pendant light, table decor",
#         "bedroom": "bedroom with bed, nightstand, dresser, bedside lighting",
#         "home office": "home office with desk, office chair, shelves, task lighting",
#         "entryway": "entryway foyer with console table, mirror, bench, shoe storage",
#         "small apartment": "small apartment room, compact furniture, storage constraints, small space",
#         "nursery or kids room": "nursery kids room, playful decor, soft colors, child friendly furniture",
#         "pet friendly home": "pet friendly room with durable furniture, washable rug, pet bed, scratch resistant surfaces",
#     },
#     "product_clues": {
#         "area rug": "area rug visible, living room rug, floor covering",
#         "washable rug": "washable rug, practical rug, pet friendly easy clean floor covering",
#         "coffee table": "coffee table visible or needed, living room center table",
#         "side table": "side table or end table beside sofa or chair",
#         "floor lamp": "floor lamp, standing lamp, ambient lighting",
#         "table lamp": "table lamp, bedside lamp, side table lamp",
#         "wall art": "wall art, framed art, prints, canvas, gallery wall",
#         "mirror": "wall mirror, round mirror, decorative mirror",
#         "storage basket": "storage basket, woven basket, blanket basket, toy storage",
#         "throw pillows": "throw pillows, cushions, sofa pillows",
#         "throw blanket": "throw blanket, sofa blanket, cozy textile",
#         "accent chair": "accent chair, lounge chair, reading chair",
#         "media console": "media console, tv stand, storage cabinet",
#         "plants and planters": "indoor plants, planters, plant stands",
#         "pet bed": "pet bed, dog bed, cat bed, pet furniture",
#         "sofa cover": "sofa cover, slipcover, pet protective couch cover",
#     },
# }


# # ============================================================
# # 2. Data classes
# # ============================================================

# @dataclass
# class ImageInput:
#     source_type: str
#     mime_type: Optional[str] = None
#     image_path: Optional[str] = None
#     gcs_uri: Optional[str] = None
#     image_bytes: Optional[bytes] = None
#     provided_part: Optional[Any] = None


# # ============================================================
# # 3. Generic helpers
# # ============================================================

# def clean_text(value: Optional[str], max_len: int = 1000) -> str:
#     if value is None:
#         return ""
#     text = re.sub(r"\s+", " ", str(value)).strip()
#     if len(text) > max_len:
#         return text[: max_len - 3] + "..."
#     return text


# def safe_json_loads(value: Any) -> Any:
#     if isinstance(value, (dict, list)):
#         return value

#     if value is None:
#         return {}

#     text = str(value).strip()
#     if not text:
#         return {}

#     if text.startswith("```"):
#         text = text.replace("```json", "").replace("```", "").strip()

#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         return {}


# def guess_mime_type(path_or_uri: str, fallback: str = "image/jpeg") -> str:
#     guessed, _ = mimetypes.guess_type(path_or_uri)
#     return guessed or fallback


# def l2_normalize(vec: np.ndarray) -> np.ndarray:
#     norm = np.linalg.norm(vec)
#     if norm == 0:
#         return vec
#     return vec / norm


# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     a = l2_normalize(a.astype(float))
#     b = l2_normalize(b.astype(float))
#     return float(np.dot(a, b))


# # ============================================================
# # 4. Extract image input from ADK/user state
# # ============================================================

# def extract_image_input_from_state(state: Dict[str, Any]) -> Optional[ImageInput]:
#     """
#     Supported state keys:
#       - input_image_path / image_path / uploaded_image_path
#       - input_image_gcs_uri / image_gcs_uri / gcs_uri
#       - input_image_base64 / image_base64
#       - input_image_mime_type / image_mime_type
#     """
#     image_path = (
#         state.get("input_image_path")
#         or state.get("image_path")
#         or state.get("uploaded_image_path")
#     )

#     if image_path:
#         image_path = str(image_path)
#         return ImageInput(
#             source_type="path",
#             image_path=image_path,
#             mime_type=guess_mime_type(image_path),
#         )

#     gcs_uri = (
#         state.get("input_image_gcs_uri")
#         or state.get("image_gcs_uri")
#         or state.get("gcs_uri")
#     )

#     if gcs_uri:
#         gcs_uri = str(gcs_uri)
#         return ImageInput(
#             source_type="gcs_uri",
#             gcs_uri=gcs_uri,
#             mime_type=state.get("input_image_mime_type")
#             or state.get("image_mime_type")
#             or guess_mime_type(gcs_uri),
#         )

#     b64_value = state.get("input_image_base64") or state.get("image_base64")
#     if b64_value:
#         mime_type = (
#             state.get("input_image_mime_type")
#             or state.get("image_mime_type")
#             or "image/jpeg"
#         )
#         return ImageInput(
#             source_type="base64",
#             image_bytes=base64.b64decode(str(b64_value)),
#             mime_type=mime_type,
#         )

#     return None


# def extract_image_input_from_content(user_content: Any) -> Optional[ImageInput]:
#     """
#     Best-effort support for ADK user content parts.

#     If ADK gives image attachments as Part.inline_data or Part.file_data, we reuse
#     the original Part directly. This avoids guessing local file handling.
#     """
#     if user_content is None:
#         return None

#     parts = getattr(user_content, "parts", None) or []

#     for part in parts:
#         inline_data = getattr(part, "inline_data", None)
#         if inline_data is not None:
#             mime_type = getattr(inline_data, "mime_type", None) or "image/jpeg"
#             return ImageInput(
#                 source_type="inline_part",
#                 mime_type=mime_type,
#                 provided_part=part,
#             )

#         file_data = getattr(part, "file_data", None)
#         if file_data is not None:
#             file_uri = getattr(file_data, "file_uri", None)
#             mime_type = getattr(file_data, "mime_type", None) or guess_mime_type(str(file_uri))
#             if file_uri:
#                 return ImageInput(
#                     source_type="file_part",
#                     gcs_uri=str(file_uri),
#                     mime_type=mime_type,
#                     provided_part=part,
#                 )

#     return None


# def extract_image_input_from_text(text: str) -> Optional[ImageInput]:
#     """
#     Useful for local debugging:
#       image_path=/tmp/room.jpg
#       image_gcs_uri=gs://bucket/room.jpg
#     """
#     if not text:
#         return None

#     path_match = re.search(r"image_path\s*=\s*([^\s]+)", text)
#     if path_match:
#         image_path = path_match.group(1).strip()
#         return ImageInput(
#             source_type="path",
#             image_path=image_path,
#             mime_type=guess_mime_type(image_path),
#         )

#     gcs_match = re.search(r"image_gcs_uri\s*=\s*(gs://[^\s]+)", text)
#     if gcs_match:
#         gcs_uri = gcs_match.group(1).strip()
#         return ImageInput(
#             source_type="gcs_uri",
#             gcs_uri=gcs_uri,
#             mime_type=guess_mime_type(gcs_uri),
#         )

#     return None


# def extract_text_from_content(user_content: Any) -> str:
#     if user_content is None:
#         return ""

#     parts = getattr(user_content, "parts", None) or []
#     texts: List[str] = []

#     for part in parts:
#         text = getattr(part, "text", None)
#         if text:
#             texts.append(str(text))

#     return "\n".join(texts).strip()


# def find_image_input(ctx: Any) -> Optional[ImageInput]:
#     state = getattr(getattr(ctx, "session", None), "state", {}) or {}

#     from_state = extract_image_input_from_state(state)
#     if from_state:
#         return from_state

#     user_content = getattr(ctx, "user_content", None)
#     from_content = extract_image_input_from_content(user_content)
#     if from_content:
#         return from_content

#     user_text = extract_text_from_content(user_content)
#     from_text = extract_image_input_from_text(user_text)
#     if from_text:
#         return from_text

#     # Last fallback: state may contain raw user text.
#     for key in ["user_query", "input_text", "raw_user_input"]:
#         maybe_text = state.get(key)
#         if maybe_text:
#             from_text = extract_image_input_from_text(str(maybe_text))
#             if from_text:
#                 return from_text

#     return None


# # ============================================================
# # 5. Gemini Embedding 2 client
# # ============================================================

# class GeminiEmbedding2VisualPreferenceExtractor:
#     def __init__(
#         self,
#         model: Optional[str] = None,
#         output_dimensionality: Optional[int] = None,
#     ):
#         load_dotenv()

#         os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

#         project = os.environ.get("GOOGLE_CLOUD_PROJECT")
#         location = (
#             os.environ.get("GOOGLE_CLOUD_EMBEDDING_LOCATION")
#             or os.environ.get("GOOGLE_CLOUD_LOCATION")
#             or "us"
#         )

#         if not project:
#             raise RuntimeError(
#                 "Missing GOOGLE_CLOUD_PROJECT. Set it in .env or environment."
#             )

#         self.model = model or os.environ.get("VISUAL_EMBEDDING_MODEL", "gemini-embedding-2")
#         self.output_dimensionality = int(
#             output_dimensionality
#             or os.environ.get("VISUAL_EMBEDDING_DIM", "256")
#         )

#         self.client = genai.Client(
#             vertexai=True,
#             project=project,
#             location=location,
#         )

#     def _image_part(self, image_input: ImageInput) -> Any:
#         if image_input.provided_part is not None:
#             return image_input.provided_part

#         if image_input.gcs_uri:
#             return types.Part.from_uri(
#                 file_uri=image_input.gcs_uri,
#                 mime_type=image_input.mime_type or guess_mime_type(image_input.gcs_uri),
#             )

#         image_bytes = image_input.image_bytes

#         if image_input.image_path:
#             path = Path(image_input.image_path)
#             image_bytes = path.read_bytes()

#         if image_bytes is None:
#             raise ValueError("ImageInput has no bytes, local path, GCS URI, or provided Part.")

#         mime_type = image_input.mime_type or "image/jpeg"

#         # google-genai has Part.from_bytes in current SDKs.
#         # Blob fallback keeps this robust across minor SDK versions.
#         if hasattr(types.Part, "from_bytes"):
#             return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

#         return types.Part(
#             inline_data=types.Blob(
#                 data=image_bytes,
#                 mime_type=mime_type,
#             )
#         )

#     def embed_content_parts(self, parts: List[Any]) -> np.ndarray:
#         content = types.Content(parts=parts)

#         response = self.client.models.embed_content(
#             model=self.model,
#             contents=[content],
#             config=types.EmbedContentConfig(
#                 output_dimensionality=self.output_dimensionality
#             ),
#         )

#         values = response.embeddings[0].values
#         return np.array(values, dtype=float)

#     def embed_text_label(self, label_prompt: str) -> np.ndarray:
#         # For gemini-embedding-2, the docs recommend putting task instructions
#         # directly in the text prompt rather than using task_type.
#         text = f"task: classification | query: {label_prompt}"

#         return self.embed_content_parts(
#             [
#                 types.Part.from_text(text=text),
#             ]
#         )

#     def embed_room_image(self, image_input: ImageInput, user_query: str = "") -> np.ndarray:
#         instruction = (
#             "task: classification | query: interior design preference extraction "
#             "for ecommerce home furniture and decor recommendations. "
#             "Focus on room style, color palette, materials, furniture types, "
#             "layout/use case, durability and pet-friendly clues. "
#         )

#         if user_query:
#             instruction += f"User shopping request context: {user_query}"

#         return self.embed_content_parts(
#             [
#                 types.Part.from_text(text=instruction),
#                 self._image_part(image_input),
#             ]
#         )

#     @lru_cache(maxsize=1)
#     def taxonomy_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
#         out: Dict[str, Dict[str, np.ndarray]] = {}

#         for section, labels in PREFERENCE_TAXONOMY.items():
#             out[section] = {}
#             for label, prompt in labels.items():
#                 out[section][label] = self.embed_text_label(prompt)

#         return out

#     def rank_taxonomy(
#         self,
#         image_embedding: np.ndarray,
#         top_k_by_section: Optional[Dict[str, int]] = None,
#     ) -> Dict[str, List[Dict[str, Any]]]:
#         top_k_by_section = top_k_by_section or {
#             "styles": 4,
#             "colors": 4,
#             "materials": 5,
#             "room_or_use_case": 3,
#             "product_clues": 8,
#         }

#         taxonomy = self.taxonomy_embeddings()
#         ranked: Dict[str, List[Dict[str, Any]]] = {}

#         for section, label_embeddings in taxonomy.items():
#             rows: List[Dict[str, Any]] = []

#             for label, emb in label_embeddings.items():
#                 score = cosine_similarity(image_embedding, emb)
#                 rows.append(
#                     {
#                         "label": label,
#                         "score": round(score, 4),
#                         "prompt": PREFERENCE_TAXONOMY[section][label],
#                     }
#                 )

#             rows.sort(key=lambda x: x["score"], reverse=True)
#             ranked[section] = rows[: top_k_by_section.get(section, 5)]

#         return ranked

#     def extract_preferences(
#         self,
#         image_input: ImageInput,
#         user_query: str = "",
#     ) -> Dict[str, Any]:
#         image_embedding = self.embed_room_image(
#             image_input=image_input,
#             user_query=user_query,
#         )

#         ranked = self.rank_taxonomy(image_embedding)

#         styles = [x["label"] for x in ranked.get("styles", [])[:3]]
#         colors = [x["label"] for x in ranked.get("colors", [])[:3]]
#         materials = [x["label"] for x in ranked.get("materials", [])[:4]]
#         room_matches = ranked.get("room_or_use_case", [])
#         product_clues = [x["label"] for x in ranked.get("product_clues", [])[:8]]

#         room_or_use_case = room_matches[0]["label"] if room_matches else ""

#         visual_nice_to_have = [
#             *styles[:2],
#             *colors[:2],
#             *materials[:3],
#             *product_clues[:5],
#         ]

#         # Deduplicate preserving order.
#         seen = set()
#         visual_nice_to_have = [
#             x for x in visual_nice_to_have
#             if not (x in seen or seen.add(x))
#         ]

#         return {
#             "has_image": True,
#             "image_source_type": image_input.source_type,
#             "embedding_model": self.model,
#             "embedding_dimensionality": self.output_dimensionality,
#             "room_or_use_case": room_or_use_case,
#             "styles": styles,
#             "colors": colors,
#             "materials": materials,
#             "product_clues": product_clues,
#             "visual_must_have": [],
#             "visual_nice_to_have": visual_nice_to_have,
#             "avoid": [],
#             "raw_matches": ranked,
#             "notes": [
#                 "Visual preferences were inferred using Gemini Embedding 2 similarity against an interior-design taxonomy.",
#                 "Scores are ranking signals, not calibrated probabilities.",
#             ],
#         }


# def empty_visual_preference_output(reason: str = "No image input provided.") -> Dict[str, Any]:
#     return {
#         "has_image": False,
#         "image_source_type": None,
#         "embedding_model": os.environ.get("VISUAL_EMBEDDING_MODEL", "gemini-embedding-2"),
#         "embedding_dimensionality": int(os.environ.get("VISUAL_EMBEDDING_DIM", "256")),
#         "room_or_use_case": "",
#         "styles": [],
#         "colors": [],
#         "materials": [],
#         "product_clues": [],
#         "visual_must_have": [],
#         "visual_nice_to_have": [],
#         "avoid": [],
#         "raw_matches": {},
#         "notes": [reason],
#     }



# shopping_agent/tools/visual_preference_extractor.py
# ============================================================
# Visual Preference Extraction using Gemini Embedding 2.
#
# Purpose:
#   Convert an uploaded room/product inspiration image into structured
#   shopping preferences for the PlannerAgent.
#
# Method:
#   1. Embed the image using Vertex AI Gemini Embedding 2.
#   2. Embed curated interior-design preference labels/prompts.
#   3. Rank labels by cosine similarity to the image embedding.
#   4. Return structured preferences:
#      styles, colors, materials, room/use case, product clues.
#
# This is embedding-based preference extraction, not generative captioning.
# It does not invent products or URLs.
#
# Env:
#   GOOGLE_GENAI_USE_VERTEXAI=TRUE
#   GOOGLE_CLOUD_PROJECT=...
#   GOOGLE_CLOUD_LOCATION=us-central1 or us
#   GOOGLE_CLOUD_EMBEDDING_LOCATION=us  # optional override
#   VISUAL_EMBEDDING_MODEL=gemini-embedding-2
#   VISUAL_EMBEDDING_DIM=256
# ============================================================

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. Preference taxonomy
# ============================================================

PREFERENCE_TAXONOMY: Dict[str, Dict[str, str]] = {
    "styles": {
        "minimalist": "minimalist interior design, clean lines, sparse decor, uncluttered room",
        "modern": "modern interior design, contemporary furniture, clean silhouettes",
        "mid century modern": "mid century modern furniture, warm wood, tapered legs, retro modern room",
        "scandinavian": "Scandinavian interior design, light wood, cozy minimalism, bright neutral room",
        "japandi": "Japandi interior design, Japanese Scandinavian fusion, natural wood, low contrast, calm neutral room",
        "boho": "bohemian interior design, layered textiles, woven decor, plants, eclectic warm room",
        "industrial": "industrial interior design, metal, black accents, exposed materials, urban loft",
        "farmhouse": "farmhouse interior design, rustic wood, cozy traditional, natural textures",
        "coastal": "coastal interior design, airy light colors, linen, rattan, beach inspired decor",
        "traditional": "traditional interior design, classic furniture, ornate details, formal room",
        "glam": "glam interior design, gold accents, velvet, polished surfaces, dramatic decor",
        "organic modern": "organic modern interior design, natural stone, wood, curved furniture, earthy neutral palette",
    },
    "colors": {
        "warm neutrals": "warm neutral color palette, beige, cream, ivory, tan, soft brown",
        "cool neutrals": "cool neutral color palette, white, gray, charcoal, black accents",
        "earth tones": "earth tone palette, terracotta, clay, olive, brown, sand, rust",
        "black accents": "black accent decor, black metal, contrast details",
        "white and cream": "white and cream room palette, light bright airy decor",
        "beige and tan": "beige tan room palette, warm soft neutral furniture",
        "green accents": "green accents, sage green, olive green, plants in room",
        "blue accents": "blue accents, navy, light blue, coastal blue decor",
        "gold accents": "gold brass metallic accents in decor",
        "colorful eclectic": "colorful eclectic room, multiple saturated colors, playful decor",
    },
    "materials": {
        "natural wood": "natural wood furniture, oak, walnut, ash, warm wood grain",
        "rattan and cane": "rattan cane wicker furniture, woven natural fibers",
        "metal": "metal furniture or decor, black metal, steel, iron, brass",
        "marble or stone": "marble stone travertine surfaces, stone coffee table or decor",
        "linen fabric": "linen upholstery, soft woven natural fabric furniture",
        "boucle fabric": "boucle upholstery, nubby textured fabric, cozy white chair or sofa",
        "leather": "leather upholstery, leather sofa or chair",
        "glass": "glass table or transparent glass decor",
        "ceramic": "ceramic decor, ceramic vase, pottery, handmade clay accessories",
        "jute or sisal": "jute sisal natural fiber rug, woven rug texture",
    },
    "room_or_use_case": {
        "living room": "living room with sofa, coffee table, rug, side table, media console",
        "dining room": "dining room with dining table, dining chairs, pendant light, table decor",
        "bedroom": "bedroom with bed, nightstand, dresser, bedside lighting",
        "home office": "home office with desk, office chair, shelves, task lighting",
        "entryway": "entryway foyer with console table, mirror, bench, shoe storage",
        "small apartment": "small apartment room, compact furniture, storage constraints, small space",
        "nursery or kids room": "nursery kids room, playful decor, soft colors, child friendly furniture",
        "pet friendly home": "pet friendly room with durable furniture, washable rug, pet bed, scratch resistant surfaces",
    },
    "product_clues": {
        "area rug": "area rug visible, living room rug, floor covering",
        "washable rug": "washable rug, practical rug, pet friendly easy clean floor covering",
        "coffee table": "coffee table visible or needed, living room center table",
        "side table": "side table or end table beside sofa or chair",
        "floor lamp": "floor lamp, standing lamp, ambient lighting",
        "table lamp": "table lamp, bedside lamp, side table lamp",
        "wall art": "wall art, framed art, prints, canvas, gallery wall",
        "mirror": "wall mirror, round mirror, decorative mirror",
        "storage basket": "storage basket, woven basket, blanket basket, toy storage",
        "throw pillows": "throw pillows, cushions, sofa pillows",
        "throw blanket": "throw blanket, sofa blanket, cozy textile",
        "accent chair": "accent chair, lounge chair, reading chair",
        "media console": "media console, tv stand, storage cabinet",
        "plants and planters": "indoor plants, planters, plant stands",
        "pet bed": "pet bed, dog bed, cat bed, pet furniture",
        "sofa cover": "sofa cover, slipcover, pet protective couch cover",
    },
}


# ============================================================
# 2. Data classes
# ============================================================

@dataclass
class ImageInput:
    source_type: str
    mime_type: Optional[str] = None
    image_path: Optional[str] = None
    gcs_uri: Optional[str] = None
    image_bytes: Optional[bytes] = None
    provided_part: Optional[Any] = None


# ============================================================
# 3. Generic helpers
# ============================================================

def clean_text(value: Optional[str], max_len: int = 1000) -> str:
    if value is None:
        return ""

    text = re.sub(r"\s+", " ", str(value)).strip()

    if len(text) > max_len:
        return text[: max_len - 3] + "..."

    return text


def clean_path_like_value(value: Any) -> str:
    """
    Normalizes values like:
      '"data/images/room.jpg"' -> 'data/images/room.jpg'
      "'data/images/room.jpg'" -> 'data/images/room.jpg'
      " data/images/room.jpg " -> 'data/images/room.jpg'

    Also expands:
      ~/image.jpg
      $HOME/image.jpg
    """
    text = str(value).strip()

    # Remove one layer of matching surrounding quotes.
    if (
        len(text) >= 2
        and (
            (text[0] == text[-1] == '"')
            or (text[0] == text[-1] == "'")
        )
    ):
        text = text[1:-1].strip()

    text = os.path.expandvars(text)
    text = os.path.expanduser(text)

    return text


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value

    if value is None:
        return {}

    text = str(value).strip()

    if not text:
        return {}

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def guess_mime_type(path_or_uri: str, fallback: str = "image/jpeg") -> str:
    path_or_uri = clean_path_like_value(path_or_uri)
    guessed, _ = mimetypes.guess_type(path_or_uri)
    return guessed or fallback


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(float))
    b = l2_normalize(b.astype(float))
    return float(np.dot(a, b))


# ============================================================
# 4. Extract image input from ADK/user state
# ============================================================

def extract_image_input_from_state(state: Dict[str, Any]) -> Optional[ImageInput]:
    """
    Supported state keys:
      - input_image_path / image_path / uploaded_image_path
      - input_image_gcs_uri / image_gcs_uri / gcs_uri
      - input_image_base64 / image_base64
      - input_image_mime_type / image_mime_type
    """
    image_path = (
        state.get("input_image_path")
        or state.get("image_path")
        or state.get("uploaded_image_path")
    )

    if image_path:
        image_path = clean_path_like_value(image_path)

        return ImageInput(
            source_type="path",
            image_path=image_path,
            mime_type=guess_mime_type(image_path),
        )

    gcs_uri = (
        state.get("input_image_gcs_uri")
        or state.get("image_gcs_uri")
        or state.get("gcs_uri")
    )

    if gcs_uri:
        gcs_uri = clean_path_like_value(gcs_uri)

        return ImageInput(
            source_type="gcs_uri",
            gcs_uri=gcs_uri,
            mime_type=(
                state.get("input_image_mime_type")
                or state.get("image_mime_type")
                or guess_mime_type(gcs_uri)
            ),
        )

    b64_value = state.get("input_image_base64") or state.get("image_base64")

    if b64_value:
        mime_type = (
            state.get("input_image_mime_type")
            or state.get("image_mime_type")
            or "image/jpeg"
        )

        return ImageInput(
            source_type="base64",
            image_bytes=base64.b64decode(str(b64_value)),
            mime_type=mime_type,
        )

    return None


def extract_image_input_from_content(user_content: Any) -> Optional[ImageInput]:
    """
    Best-effort support for ADK user content parts.

    If ADK gives image attachments as Part.inline_data or Part.file_data, we reuse
    the original Part directly. This avoids guessing local file handling.
    """
    if user_content is None:
        return None

    parts = getattr(user_content, "parts", None) or []

    for part in parts:
        inline_data = getattr(part, "inline_data", None)

        if inline_data is not None:
            mime_type = getattr(inline_data, "mime_type", None) or "image/jpeg"

            return ImageInput(
                source_type="inline_part",
                mime_type=mime_type,
                provided_part=part,
            )

        file_data = getattr(part, "file_data", None)

        if file_data is not None:
            file_uri = getattr(file_data, "file_uri", None)
            mime_type = (
                getattr(file_data, "mime_type", None)
                or guess_mime_type(str(file_uri))
            )

            if file_uri:
                return ImageInput(
                    source_type="file_part",
                    gcs_uri=clean_path_like_value(file_uri),
                    mime_type=mime_type,
                    provided_part=part,
                )

    return None


def extract_image_input_from_text(text: str) -> Optional[ImageInput]:
    """
    Useful for local debugging.

    Supported:
      image_path=data/images/room.jpg
      image_path="data/images/room.jpg"
      image_path='data/images/room.jpg'

      image_gcs_uri=gs://bucket/room.jpg
      image_gcs_uri="gs://bucket/room.jpg"
      image_gcs_uri='gs://bucket/room.jpg'
    """
    if not text:
        return None

    # Supports quoted paths with spaces and unquoted paths without spaces.
    path_match = re.search(
        r"image_path\s*=\s*(\"[^\"]+\"|'[^']+'|[^\s]+)",
        text,
    )

    if path_match:
        image_path = clean_path_like_value(path_match.group(1))

        return ImageInput(
            source_type="path",
            image_path=image_path,
            mime_type=guess_mime_type(image_path),
        )

    gcs_match = re.search(
        r"image_gcs_uri\s*=\s*(\"gs://[^\"]+\"|'gs://[^']+'|gs://[^\s]+)",
        text,
    )

    if gcs_match:
        gcs_uri = clean_path_like_value(gcs_match.group(1))

        return ImageInput(
            source_type="gcs_uri",
            gcs_uri=gcs_uri,
            mime_type=guess_mime_type(gcs_uri),
        )

    return None


def extract_text_from_content(user_content: Any) -> str:
    if user_content is None:
        return ""

    parts = getattr(user_content, "parts", None) or []
    texts: List[str] = []

    for part in parts:
        text = getattr(part, "text", None)

        if text:
            texts.append(str(text))

    return "\n".join(texts).strip()


def find_image_input(ctx: Any) -> Optional[ImageInput]:
    state = getattr(getattr(ctx, "session", None), "state", {}) or {}

    from_state = extract_image_input_from_state(state)

    if from_state:
        return from_state

    user_content = getattr(ctx, "user_content", None)

    from_content = extract_image_input_from_content(user_content)

    if from_content:
        return from_content

    user_text = extract_text_from_content(user_content)

    from_text = extract_image_input_from_text(user_text)

    if from_text:
        return from_text

    # Last fallback: state may contain raw user text.
    for key in ["user_query", "input_text", "raw_user_input"]:
        maybe_text = state.get(key)

        if maybe_text:
            from_text = extract_image_input_from_text(str(maybe_text))

            if from_text:
                return from_text

    return None


# ============================================================
# 5. Gemini Embedding 2 client
# ============================================================

class GeminiEmbedding2VisualPreferenceExtractor:
    def __init__(
        self,
        model: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ):
        load_dotenv()

        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = (
            os.environ.get("GOOGLE_CLOUD_EMBEDDING_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or "us"
        )

        if not project:
            raise RuntimeError(
                "Missing GOOGLE_CLOUD_PROJECT. Set it in .env or environment."
            )

        self.model = model or os.environ.get(
            "VISUAL_EMBEDDING_MODEL",
            "gemini-embedding-2",
        )

        self.output_dimensionality = int(
            output_dimensionality
            or os.environ.get("VISUAL_EMBEDDING_DIM", "256")
        )

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def _image_part(self, image_input: ImageInput) -> Any:
        if image_input.provided_part is not None:
            return image_input.provided_part

        if image_input.gcs_uri:
            return types.Part.from_uri(
                file_uri=clean_path_like_value(image_input.gcs_uri),
                mime_type=(
                    image_input.mime_type
                    or guess_mime_type(image_input.gcs_uri)
                ),
            )

        image_bytes = image_input.image_bytes

        if image_input.image_path:
            image_path = clean_path_like_value(image_input.image_path)
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image file not found: {image_path}. "
                    f"Current working directory: {Path.cwd()}"
                )

            image_bytes = path.read_bytes()

        if image_bytes is None:
            raise ValueError(
                "ImageInput has no bytes, local path, GCS URI, or provided Part."
            )

        mime_type = image_input.mime_type or "image/jpeg"

        # google-genai has Part.from_bytes in current SDKs.
        # Blob fallback keeps this robust across minor SDK versions.
        if hasattr(types.Part, "from_bytes"):
            return types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )

        return types.Part(
            inline_data=types.Blob(
                data=image_bytes,
                mime_type=mime_type,
            )
        )

    def embed_content_parts(self, parts: List[Any]) -> np.ndarray:
        content = types.Content(parts=parts)

        response = self.client.models.embed_content(
            model=self.model,
            contents=[content],
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            ),
        )

        values = response.embeddings[0].values

        return np.array(values, dtype=float)

    def embed_text_label(self, label_prompt: str) -> np.ndarray:
        # For gemini-embedding-2, task instructions are placed directly
        # in the text prompt.
        text = f"task: classification | query: {label_prompt}"

        return self.embed_content_parts(
            [
                types.Part.from_text(text=text),
            ]
        )

    def embed_room_image(
        self,
        image_input: ImageInput,
        user_query: str = "",
    ) -> np.ndarray:
        instruction = (
            "task: classification | query: interior design preference extraction "
            "for ecommerce home furniture and decor recommendations. "
            "Focus on room style, color palette, materials, furniture types, "
            "layout/use case, durability and pet-friendly clues. "
        )

        user_query = clean_text(user_query, max_len=1000)

        if user_query:
            instruction += f" User shopping request context: {user_query}"

        return self.embed_content_parts(
            [
                types.Part.from_text(text=instruction),
                self._image_part(image_input),
            ]
        )

    @lru_cache(maxsize=1)
    def taxonomy_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}

        for section, labels in PREFERENCE_TAXONOMY.items():
            out[section] = {}

            for label, prompt in labels.items():
                out[section][label] = self.embed_text_label(prompt)

        return out

    def rank_taxonomy(
        self,
        image_embedding: np.ndarray,
        top_k_by_section: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        top_k_by_section = top_k_by_section or {
            "styles": 4,
            "colors": 4,
            "materials": 5,
            "room_or_use_case": 3,
            "product_clues": 8,
        }

        taxonomy = self.taxonomy_embeddings()
        ranked: Dict[str, List[Dict[str, Any]]] = {}

        for section, label_embeddings in taxonomy.items():
            rows: List[Dict[str, Any]] = []

            for label, emb in label_embeddings.items():
                score = cosine_similarity(image_embedding, emb)

                rows.append(
                    {
                        "label": label,
                        "score": round(score, 4),
                        "prompt": PREFERENCE_TAXONOMY[section][label],
                    }
                )

            rows.sort(key=lambda x: x["score"], reverse=True)
            ranked[section] = rows[: top_k_by_section.get(section, 5)]

        return ranked

    def extract_preferences(
        self,
        image_input: ImageInput,
        user_query: str = "",
    ) -> Dict[str, Any]:
        image_embedding = self.embed_room_image(
            image_input=image_input,
            user_query=user_query,
        )

        ranked = self.rank_taxonomy(image_embedding)

        styles = [x["label"] for x in ranked.get("styles", [])[:3]]
        colors = [x["label"] for x in ranked.get("colors", [])[:3]]
        materials = [x["label"] for x in ranked.get("materials", [])[:4]]

        room_matches = ranked.get("room_or_use_case", [])
        room_or_use_case = room_matches[0]["label"] if room_matches else ""

        product_clues = [
            x["label"]
            for x in ranked.get("product_clues", [])[:8]
        ]

        visual_nice_to_have = [
            *styles[:2],
            *colors[:2],
            *materials[:3],
            *product_clues[:5],
        ]

        # Deduplicate preserving order.
        seen = set()
        visual_nice_to_have = [
            x
            for x in visual_nice_to_have
            if not (x in seen or seen.add(x))
        ]

        return {
            "has_image": True,
            "image_source_type": image_input.source_type,
            "embedding_model": self.model,
            "embedding_dimensionality": self.output_dimensionality,
            "room_or_use_case": room_or_use_case,
            "styles": styles,
            "colors": colors,
            "materials": materials,
            "product_clues": product_clues,
            "visual_must_have": [],
            "visual_nice_to_have": visual_nice_to_have,
            "avoid": [],
            "raw_matches": ranked,
            "notes": [
                "Visual preferences were inferred using Gemini Embedding 2 similarity against an interior-design taxonomy.",
                "Scores are ranking signals, not calibrated probabilities.",
            ],
        }


def empty_visual_preference_output(
    reason: str = "No image input provided.",
) -> Dict[str, Any]:
    return {
        "has_image": False,
        "image_source_type": None,
        "embedding_model": os.environ.get(
            "VISUAL_EMBEDDING_MODEL",
            "gemini-embedding-2",
        ),
        "embedding_dimensionality": int(
            os.environ.get("VISUAL_EMBEDDING_DIM", "256")
        ),
        "room_or_use_case": "",
        "styles": [],
        "colors": [],
        "materials": [],
        "product_clues": [],
        "visual_must_have": [],
        "visual_nice_to_have": [],
        "avoid": [],
        "raw_matches": {},
        "notes": [reason],
    }