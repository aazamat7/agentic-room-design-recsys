from __future__ import annotations

import json

from google.cloud import aiplatform_v1
from google.protobuf.json_format import MessageToDict

from renovation_agent.config import get_settings


def _to_dict(value):
    try:
        return MessageToDict(value)
    except Exception:
        try:
            return dict(value)
        except Exception:
            return {}


def main() -> None:
    settings = get_settings()
    location = settings.vector_location
    parent = f"projects/{settings.project_id}/locations/{location}"
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}

    index_client = aiplatform_v1.IndexServiceClient(client_options=client_options)
    endpoint_client = aiplatform_v1.IndexEndpointServiceClient(
        client_options=client_options
    )

    indexes = []
    for index in index_client.list_indexes(parent=parent):
        indexes.append(
            {
                "name": index.name,
                "display_name": index.display_name,
                "metadata": _to_dict(index.metadata),
            }
        )

    endpoints = []
    for endpoint in endpoint_client.list_index_endpoints(parent=parent):
        endpoints.append(
            {
                "name": endpoint.name,
                "display_name": endpoint.display_name,
                "public_endpoint_domain_name": endpoint.public_endpoint_domain_name,
                "deployed_indexes": [
                    {
                        "id": deployed.id,
                        "index": deployed.index,
                        "display_name": deployed.display_name,
                    }
                    for deployed in endpoint.deployed_indexes
                ],
            }
        )

    print(
        json.dumps(
            {
                "project": settings.project_id,
                "location": location,
                "configured_data_prefix": settings.vector_data_prefix,
                "configured_display_name": settings.vector_index_display_name,
                "indexes": indexes,
                "index_endpoints": endpoints,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
