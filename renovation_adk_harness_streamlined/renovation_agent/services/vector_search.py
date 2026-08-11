from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google import genai
from google.cloud import aiplatform, aiplatform_v1
from google.genai import types
from google.protobuf.json_format import MessageToDict

from renovation_agent.config import Settings
from renovation_agent.schemas import ReferenceCandidate
from renovation_agent.services.gcs_store import GCSImageStore
from renovation_agent.services.image_io import load_image
from renovation_agent.services.metadata_catalog import MetadataCatalog


@dataclass(frozen=True)
class VectorResources:
    index_name: str
    endpoint_name: str
    deployed_index_id: str
    dimension: int
    dimension_source: str


class VectorSearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.catalog = MetadataCatalog(settings)
        self.store = GCSImageStore(settings)
        self.genai_client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.embedding_location,
        )
        self._resources: VectorResources | None = None

    def resolve_resources(self) -> VectorResources:
        if self._resources is not None:
            return self._resources

        location = self.settings.vector_location
        parent = f"projects/{self.settings.project_id}/locations/{location}"
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}

        index_client = aiplatform_v1.IndexServiceClient(client_options=client_options)
        endpoint_client = aiplatform_v1.IndexEndpointServiceClient(
            client_options=client_options
        )

        index = None
        if self.settings.vector_index_name:
            full_name = _canonical_resource_name(
                self.settings.vector_index_name,
                parent=parent,
                collection="indexes",
            )
            index = index_client.get_index(name=full_name)
        else:
            indexes = list(index_client.list_indexes(parent=parent))
            index = _choose_index(
                indexes=indexes,
                display_name=self.settings.vector_index_display_name,
                data_prefix=self.settings.vector_data_prefix,
            )
        if index is None:
            available = [
                {
                    "display_name": getattr(candidate, "display_name", None),
                    "name": getattr(candidate, "name", None),
                    "metadata_uris": _gcs_values(
                        _to_dict(getattr(candidate, "metadata", {}))
                    ),
                }
                for candidate in indexes
            ]
            raise RuntimeError(
                "Could not find a Vertex AI Vector Search index matching "
                f"display name {self.settings.vector_index_display_name!r} or data prefix "
                f"{self.settings.vector_data_prefix!r}. Available indexes: {available}. "
                "Set VECTOR_INDEX_NAME explicitly to the matching resource name."
            )

        configured_dimension = self.settings.embedding_dimension
        metadata_dimension = _extract_dimension(index)
        dimension = configured_dimension or metadata_dimension
        dimension_source = (
            "EMBEDDING_DIMENSION" if configured_dimension else "index_metadata"
        )
        if not dimension:
            raise RuntimeError(
                "Could not infer the Vector Search index dimension. This project index "
                "was built with 768-dimensional gemini-embedding-2 vectors, so set "
                "EMBEDDING_DIMENSION=768."
            )

        endpoint = None
        if self.settings.vector_index_endpoint_name:
            endpoint_name = _canonical_resource_name(
                self.settings.vector_index_endpoint_name,
                parent=parent,
                collection="indexEndpoints",
            )
            endpoint = endpoint_client.get_index_endpoint(name=endpoint_name)
        else:
            endpoints = list(endpoint_client.list_index_endpoints(parent=parent))
            endpoint = _choose_endpoint(endpoints=endpoints, index_name=index.name)
        if endpoint is None:
            raise RuntimeError(
                "The index exists but is not deployed to an IndexEndpoint. Deploy it first, "
                "or set VECTOR_INDEX_ENDPOINT_NAME and DEPLOYED_INDEX_ID."
            )

        deployed_id = self.settings.deployed_index_id or _choose_deployed_index_id(
            endpoint=endpoint, index_name=index.name
        )
        if not deployed_id:
            raise RuntimeError(
                f"Index endpoint {endpoint.name} does not contain a deployed index for {index.name}."
            )

        self._resources = VectorResources(
            index_name=index.name,
            endpoint_name=endpoint.name,
            deployed_index_id=deployed_id,
            dimension=int(dimension),
            dimension_source=dimension_source,
        )
        return self._resources

    def search(self, source_image_uri: str, num_neighbors: int) -> list[ReferenceCandidate]:
        resources = self.resolve_resources()
        payload = load_image(
            source_image_uri,
            project_id=self.settings.project_id,
            max_bytes=self.settings.max_image_bytes,
        )
        embedding_result = self.genai_client.models.embed_content(
            model=self.settings.embedding_model,
            contents=[
                types.Part.from_bytes(data=payload.data, mime_type=payload.mime_type)
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=resources.dimension
            ),
        )
        if not embedding_result.embeddings:
            raise RuntimeError("Gemini embedding call returned no embedding")
        query = embedding_result.embeddings[0].values
        if len(query) != resources.dimension:
            raise RuntimeError(
                f"Embedding dimension {len(query)} does not match index dimension "
                f"{resources.dimension}. Rebuild or query with the same embedding configuration."
            )

        aiplatform.init(project=self.settings.project_id, location=self.settings.vector_location)
        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=resources.endpoint_name
        )
        matches = endpoint.find_neighbors(
            deployed_index_id=resources.deployed_index_id,
            queries=[query],
            num_neighbors=max(1, min(num_neighbors, 20)),
        )
        neighbors = matches[0] if matches else []

        candidates: list[ReferenceCandidate] = []
        for rank, neighbor in enumerate(neighbors, start=1):
            point_id = _neighbor_id(neighbor)
            metadata = self.catalog.get(point_id)
            image_uri = metadata.get("image_uri")
            candidates.append(
                ReferenceCandidate(
                    rank=rank,
                    reference_id=point_id,
                    distance=_neighbor_distance(neighbor),
                    image_uri=image_uri,
                    preview_url=self.store.preview_url(image_uri),
                    style=metadata.get("style"),
                    room_type=metadata.get("room_type"),
                    caption=metadata.get("caption"),
                    raw_metadata=metadata.get("raw", {}),
                )
            )
        return candidates


def _canonical_resource_name(value: str, *, parent: str, collection: str) -> str:
    if value.startswith("projects/"):
        return value
    return f"{parent}/{collection}/{value}"


def _choose_index(indexes: list[Any], display_name: str, data_prefix: str) -> Any | None:
    """Choose the most likely index while remaining deterministic.

    Vertex AI index metadata may store the exact ingestion folder, a parent folder,
    or a child shard path. Match both parent/child directions after normalizing the
    GCS URI. If no configured identifier matches and the project contains exactly
    one index, use it as a safe development fallback.
    """
    exact = [idx for idx in indexes if getattr(idx, "display_name", None) == display_name]
    normalized_prefix = _normalize_gcs_prefix(data_prefix)

    for idx in exact:
        values = _gcs_values(_to_dict(getattr(idx, "metadata", {})))
        if any(_gcs_prefixes_overlap(normalized_prefix, value) for value in values):
            return idx

    if len(exact) == 1:
        return exact[0]

    prefix_matches: list[Any] = []
    for idx in indexes:
        values = _gcs_values(_to_dict(getattr(idx, "metadata", {})))
        if any(_gcs_prefixes_overlap(normalized_prefix, value) for value in values):
            prefix_matches.append(idx)

    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # Development convenience: a project with one index is unambiguous even when
    # the API omits or rewrites the original contentsDeltaUri in returned metadata.
    if len(indexes) == 1:
        return indexes[0]

    return None



def _normalize_gcs_prefix(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _gcs_values(value: Any) -> list[str]:
    return [
        _normalize_gcs_prefix(item)
        for item in _all_string_values(value)
        if str(item).startswith("gs://")
    ]


def _gcs_prefixes_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left = _normalize_gcs_prefix(left)
    right = _normalize_gcs_prefix(right)
    return (
        left == right
        or left.startswith(right + "/")
        or right.startswith(left + "/")
    )

def _choose_endpoint(endpoints: list[Any], index_name: str) -> Any | None:
    for endpoint in endpoints:
        for deployed in getattr(endpoint, "deployed_indexes", []):
            if getattr(deployed, "index", None) == index_name:
                return endpoint
    return None


def _choose_deployed_index_id(endpoint: Any, index_name: str) -> str | None:
    for deployed in getattr(endpoint, "deployed_indexes", []):
        if getattr(deployed, "index", None) == index_name:
            return str(getattr(deployed, "id", "")) or None
    deployed = list(getattr(endpoint, "deployed_indexes", []))
    if len(deployed) == 1:
        return str(getattr(deployed[0], "id", "")) or None
    return None


def _extract_dimension(index: Any) -> int | None:
    metadata = _to_dict(getattr(index, "metadata", {}))
    for key, value in _walk_items(metadata):
        if key.lower() in {"dimensions", "dimension"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return MessageToDict(value)
    except Exception:
        try:
            return dict(value)
        except Exception:
            return {}


def _walk_items(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)


def _all_string_values(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            output.extend(_all_string_values(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_all_string_values(child))
    elif isinstance(value, str):
        output.append(value)
    return output


def _neighbor_id(neighbor: Any) -> str:
    direct = getattr(neighbor, "id", None)
    if direct:
        return str(direct)
    datapoint = getattr(neighbor, "datapoint", None)
    value = getattr(datapoint, "datapoint_id", None) if datapoint else None
    if value:
        return str(value)
    raise RuntimeError(f"Vector Search returned a neighbor without a datapoint ID: {neighbor!r}")


def _neighbor_distance(neighbor: Any) -> float | None:
    value = getattr(neighbor, "distance", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
