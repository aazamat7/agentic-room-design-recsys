import base64

from renovation_agent.services.qwen_backends import _generated_from_value


def test_normalizes_url_output():
    output = _generated_from_value(
        {"output_url": "https://example.com/result.png"}, model="test"
    )
    assert output.uri == "https://example.com/result.png"
    assert output.model == "test"


def test_normalizes_base64_output():
    encoded = base64.b64encode(b"image-bytes").decode("ascii")
    output = _generated_from_value(
        {"output_base64": encoded, "mime_type": "image/png"}, model="test"
    )
    assert output.data == b"image-bytes"
    assert output.mime_type == "image/png"
