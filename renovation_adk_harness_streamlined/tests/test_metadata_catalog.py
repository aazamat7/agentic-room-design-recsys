from renovation_agent.config import Settings
from renovation_agent.services.metadata_catalog import MetadataCatalog


def test_canonicalizes_common_vector_metadata_fields():
    catalog = MetadataCatalog(Settings())
    row = {
        "datapoint_id": "abc123",
        "after_image_url": "https://example.com/after.jpg",
        "style": "Japandi",
        "room_type": "living_room",
    }
    result = catalog._canonicalize(row)
    assert result is not None
    assert result["id"] == "abc123"
    assert result["image_uri"] == "https://example.com/after.jpg"
    assert result["style"] == "Japandi"


def test_canonicalizes_pair_id_catalog_schema():
    catalog = MetadataCatalog(Settings())
    row = {
        "pair_id": "4443ccf802ba2c73b3",
        "gcs_uri": "gs://adsp-s26-reccys-bucket/living-room-renovation-index/after_orig/4443ccf802ba2c73b3.jpg",
        "room_type": "living_room",
    }
    result = catalog._canonicalize(row)
    assert result is not None
    assert result["id"] == "4443ccf802ba2c73b3"
    assert result["image_uri"].endswith("4443ccf802ba2c73b3.jpg")


def test_parses_wrapped_catalog_records():
    catalog = MetadataCatalog(Settings())
    rows = list(
        catalog._parse_rows(
            "catalog.json",
            '{"items":[{"pair_id":"abc","gcs_uri":"gs://bucket/abc.jpg"}]}',
        )
    )
    assert rows == [{"pair_id": "abc", "gcs_uri": "gs://bucket/abc.jpg"}]
