from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import google.auth
from google.auth.transport.requests import Request

from renovation_agent.bootstrap import bootstrap_environment


def main() -> None:
    bootstrap_environment()

    credentials, detected_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    print("ADC authentication succeeded.")
    print(f"Configured project: {os.environ['GOOGLE_CLOUD_PROJECT']}")
    print(f"ADC detected project: {detected_project or '(not returned)'}")
    print(f"Google Cloud location: {os.environ['GOOGLE_CLOUD_LOCATION']}")
    print(
        "Vertex backend enabled: "
        f"{os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')}"
    )
    print(f"Credential type: {type(credentials).__name__}")


if __name__ == "__main__":
    main()
