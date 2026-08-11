from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

from google.cloud import storage

from renovation_agent.config import Settings
from renovation_agent.services.image_io import parse_gs_uri


_ID_KEYS = (
    "id",
    "pair_id",  # catalog.json and Vector Search datapoints use pair_id
    "datapoint_id",
    "reference_id",
    "image_id",
    "item_id",
)
_IMAGE_KEYS = (
    "after_gcs_uri",
    "after_image_gcs_uri",
    "gcs_uri",
    "image_uri",
    "after_image_url",
    "image_url",
    "url",
    "local_path",
)


class MetadataCatalog:
    """Loads a sidecar mapping from Vector Search datapoint IDs to image metadata."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = storage.Client(project=settings.project_id)
        self._records: dict[str, dict[str, Any]] | None = None
        self.source_uri: str | None = None

    def get(self, datapoint_id: str) -> dict[str, Any]:
        if self._records is None:
            self._records = self._load()
        return dict(self._records.get(str(datapoint_id), {}))

    def size(self) -> int:
        if self._records is None:
            self._records = self._load()
        return len(self._records)

    def _load(self) -> dict[str, dict[str, Any]]:
        uri = (
            self.settings.vector_metadata_uri
            or self._configured_catalog_uri_if_present()
            or self._discover_metadata_uri()
        )
        if not uri:
            return {}
        self.source_uri = uri
        text = self._read_text(uri)
        rows = list(self._parse_rows(uri, text))
        output: dict[str, dict[str, Any]] = {}
        for raw in rows:
            canonical = self._canonicalize(raw)
            if canonical is not None:
                output[canonical["id"]] = canonical
        return output

    def _configured_catalog_uri_if_present(self) -> str | None:
        object_name = str(self.settings.catalog_object_name or "").strip().lstrip("/")
        if not object_name:
            return None
        bucket_name = self.settings.output_bucket
        blob = self.client.bucket(bucket_name).blob(object_name)
        if blob.exists():
            return f"gs://{bucket_name}/{object_name}"
        return None

    def _discover_metadata_uri(self) -> str | None:
        prefix = self.settings.vector_data_prefix
        bucket_name, object_prefix = parse_gs_uri(prefix.rstrip("/") + "/placeholder")
        object_prefix = object_prefix.rsplit("/", 1)[0].rstrip("/")

        # Search both the ingestion subfolder (for example after_orig/) and its
        # parent index folder, where catalog.json commonly lives.
        prefixes = [object_prefix + "/"]
        parent_prefix = object_prefix.rsplit("/", 1)[0] if "/" in object_prefix else ""
        if parent_prefix:
            prefixes.append(parent_prefix.rstrip("/") + "/")

        blobs_by_name = {}
        for search_prefix in prefixes:
            for blob in self.client.list_blobs(bucket_name, prefix=search_prefix):
                blobs_by_name[blob.name] = blob
        blobs = list(blobs_by_name.values())
        candidates: list[tuple[int, str]] = []
        for blob in blobs:
            lower = blob.name.lower()
            if not lower.endswith((".jsonl", ".ndjson", ".json", ".csv")):
                continue
            score = 0
            for token, weight in (
                ("metadata", 100),
                ("catalog", 90),
                ("mapping", 80),
                ("references", 70),
                ("items", 50),
            ):
                if token in lower:
                    score += weight
            # Avoid choosing the large embedding ingestion file unless its name
            # explicitly identifies it as a metadata sidecar.
            if score == 0 and any(
                token in lower for token in ("embedding", "vectors", "index_data")
            ):
                continue
            if score:
                candidates.append((score, f"gs://{bucket_name}/{blob.name}"))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _read_text(self, uri: str) -> str:
        if uri.startswith("gs://"):
            bucket_name, object_name = parse_gs_uri(uri)
            return self.client.bucket(bucket_name).blob(object_name).download_as_text()
        from pathlib import Path
        from urllib.request import urlopen

        if uri.startswith(("http://", "https://")):
            with urlopen(uri, timeout=60) as response:  # nosec B310 - user-configured URI
                return response.read().decode("utf-8")
        return Path(uri).expanduser().read_text(encoding="utf-8")

    def _parse_rows(self, uri: str, text: str) -> Iterable[dict[str, Any]]:
        lower = uri.lower()
        if lower.endswith(".csv"):
            yield from csv.DictReader(io.StringIO(text))
            return
        if lower.endswith((".jsonl", ".ndjson")):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    yield parsed
            return

        parsed = json.loads(text)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict):
                    yield row
            return

        if isinstance(parsed, dict):
            # Support wrapper formats such as {"items": [...]}, {"records": [...]},
            # or {"catalog": [...]}, in addition to an ID-keyed object.
            for wrapper_key in ("items", "records", "catalog", "data", "references"):
                wrapped = parsed.get(wrapper_key)
                if isinstance(wrapped, list):
                    for row in wrapped:
                        if isinstance(row, dict):
                            yield row
                    return

            for key, value in parsed.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("id", key)
                    yield row

    def _canonicalize(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        datapoint_id = _first_nonempty(raw, _ID_KEYS)
        if datapoint_id is None:
            return None
        image_uri = _first_nonempty(raw, _IMAGE_KEYS)
        return {
            "id": str(datapoint_id),
            "image_uri": str(image_uri) if image_uri is not None else None,
            "style": _string_or_none(raw.get("style") or raw.get("design_style")),
            "room_type": _string_or_none(raw.get("room_type") or raw.get("room")),
            "caption": _string_or_none(
                raw.get("caption") or raw.get("description") or raw.get("prompt")
            ),
            "raw": raw,
        }


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
