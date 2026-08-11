from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def bootstrap_environment() -> None:
    """Load runtime configuration before importing ADK or google-genai.

    ADK decides between the Gemini Developer API and Vertex AI while creating
    its Google GenAI client. These variables must therefore be present before
    ``google.adk`` is imported.
    """

    package_root = Path(__file__).resolve().parent.parent
    env_path = package_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # Also support launching from a directory that already contains .env.
        load_dotenv(override=True)

    project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or "adsp-s26-reccys").strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GEMINI_LOCATION")
        or "global"
    ).strip()

    # Current ADK/google-genai compatibility variables. Both are set to the
    # same value so old and new SDK versions choose the Google Cloud backend.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "TRUE"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    # Do not require or inject a Gemini API key. Authentication is provided by
    # Application Default Credentials (ADC) for all Gemini and Vertex calls.

# Backward-compatible name used by earlier package revisions.
configure_vertex_adc_environment = bootstrap_environment
