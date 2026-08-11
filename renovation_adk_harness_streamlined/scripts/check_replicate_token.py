from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from renovation_agent.bootstrap import bootstrap_environment


def main() -> None:
    bootstrap_environment()
    token = (os.getenv("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is missing. Add it to .env or inject it from a secret store."
        )
    print("Replicate token is configured.")
    print(f"Token prefix: {token[:3]}... (value intentionally hidden)")


if __name__ == "__main__":
    main()
