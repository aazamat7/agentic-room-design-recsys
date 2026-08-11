from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from renovation_agent.bootstrap import configure_vertex_adc_environment

configure_vertex_adc_environment()

import google.auth
from google.auth.transport.requests import Request


credentials, detected_project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())

print("ADC_OK=true")
print(f"credential_type={type(credentials).__name__}")
print(f"detected_project={detected_project}")
print(f"configured_project={os.environ.get('GOOGLE_CLOUD_PROJECT')}")
print(f"configured_location={os.environ.get('GOOGLE_CLOUD_LOCATION')}")
print(f"use_vertexai={os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')}")
print(f"use_enterprise={os.environ.get('GOOGLE_GENAI_USE_ENTERPRISE')}")
print(f"token_present={bool(getattr(credentials, 'token', None))}")
