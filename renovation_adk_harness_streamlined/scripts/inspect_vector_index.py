from __future__ import annotations

import json

from renovation_agent.config import get_settings
from renovation_agent.services.vector_search import VectorSearchService


def main() -> None:
    settings = get_settings()
    service = VectorSearchService(settings)
    resources = service.resolve_resources()
    payload = {
        "project": settings.project_id,
        "vector_location": settings.vector_location,
        "embedding_location": settings.embedding_location,
        "vector_data_prefix": settings.vector_data_prefix,
        "index_name": resources.index_name,
        "endpoint_name": resources.endpoint_name,
        "deployed_index_id": resources.deployed_index_id,
        "dimension": resources.dimension,
        "dimension_source": resources.dimension_source,
        "metadata_uri": service.catalog.source_uri,
        "metadata_records": service.catalog.size(),
    }
    payload["metadata_uri"] = service.catalog.source_uri
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
