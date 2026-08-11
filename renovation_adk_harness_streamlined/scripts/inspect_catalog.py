from __future__ import annotations

import json

from renovation_agent.bootstrap import bootstrap_environment

bootstrap_environment()

from renovation_agent.config import get_settings
from renovation_agent.services.metadata_catalog import MetadataCatalog


def main() -> None:
    settings = get_settings()
    catalog = MetadataCatalog(settings)
    records = catalog._load()
    sample = list(records.values())[:5]
    print(
        json.dumps(
            {
                "configured_vector_metadata_uri": settings.vector_metadata_uri,
                "catalog_object_name": settings.catalog_object_name,
                "resolved_catalog_uri": catalog.source_uri,
                "catalog_record_count": len(records),
                "sample_records": sample,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    if not records:
        raise SystemExit(
            "Catalog loaded zero usable records. Confirm the JSON contains pair_id/id and gcs_uri/image_uri fields."
        )


if __name__ == "__main__":
    main()
