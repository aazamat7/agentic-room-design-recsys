from __future__ import annotations

import argparse
import json

from renovation_agent.config import get_settings
from renovation_agent.services.vector_search import VectorSearchService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Local path, gs:// URI, or HTTPS URL")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    service = VectorSearchService(get_settings())
    matches = service.search(args.image, args.top_k)
    print(json.dumps([m.model_dump(mode="json") for m in matches], indent=2))


if __name__ == "__main__":
    main()
