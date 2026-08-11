from __future__ import annotations

from typing import Any

# Must run before importing renovation_agent.agent / google.adk.
from renovation_agent.bootstrap import bootstrap_environment

bootstrap_environment()

__all__ = ["root_agent"]


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        from renovation_agent.agent import root_agent

        return root_agent
    raise AttributeError(name)
