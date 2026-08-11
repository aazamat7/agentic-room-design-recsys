from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from google.cloud import storage
from PIL import Image


@dataclass(frozen=True)
class ImagePayload:
    data: bytes
    mime_type: str
    filename: str
    source: str


_IMAGE_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    no_scheme = uri[5:]
    bucket, sep, object_name = no_scheme.partition("/")
    if not bucket or not sep or not object_name:
        raise ValueError(f"GCS URI must include bucket and object: {uri}")
    return bucket, object_name


def load_image(
    source: str,
    *,
    project_id: str | None = None,
    max_bytes: int = 25_000_000,
    timeout_seconds: int = 60,
) -> ImagePayload:
    """Read an image from a local path, gs:// URI, or HTTPS URL."""
    cleaned = source.strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Image source is empty")

    if cleaned.startswith("gs://"):
        bucket_name, object_name = parse_gs_uri(cleaned)
        client = storage.Client(project=project_id)
        blob = client.bucket(bucket_name).blob(object_name)
        if not blob.exists(client=client):
            raise FileNotFoundError(f"GCS image not found: {cleaned}")
        if blob.size and blob.size > max_bytes:
            raise ValueError(f"Image is larger than {max_bytes:,} bytes: {cleaned}")
        data = blob.download_as_bytes()
        filename = Path(object_name).name or "image"
        content_type = blob.content_type
    elif cleaned.startswith(("http://", "https://")):
        response = requests.get(cleaned, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.content
        filename = Path(urlparse(cleaned).path).name or "image"
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
    else:
        path = Path(cleaned).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Local image not found: {path}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Image is larger than {max_bytes:,} bytes: {path}")
        data = path.read_bytes()
        filename = path.name
        content_type = mimetypes.guess_type(path.name)[0]

    if len(data) > max_bytes:
        raise ValueError(f"Image is larger than {max_bytes:,} bytes: {cleaned}")

    mime_type = _detect_image_mime(data, content_type, filename)
    return ImagePayload(data=data, mime_type=mime_type, filename=filename, source=cleaned)


def _detect_image_mime(data: bytes, declared: str | None, filename: str) -> str:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
            detected = _IMAGE_FORMAT_TO_MIME.get((image.format or "").upper())
    except Exception as exc:
        raise ValueError(f"Unsupported or corrupt image: {filename}") from exc

    mime = detected or declared or mimetypes.guess_type(filename)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(
            f"Unsupported image type {mime!r}. Use JPEG, PNG, or WEBP."
        )
    return mime


def extension_for_mime(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime_type, ".png")
