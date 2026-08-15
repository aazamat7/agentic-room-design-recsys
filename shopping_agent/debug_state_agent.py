from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List

from pydantic import Field
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from shopping_agent.debug_utils import (
    build_debug_event_payload,
    debug_enabled,
    print_state_snapshot,
    write_snapshot,
)


class DebugStatePrinterAgent(BaseAgent):
    """
    Prints and saves selected ADK session state keys.

    Example:
        DebugStatePrinterAgent(
            name="DebugAfterPlanner",
            label="after_planner",
            keys_to_print=["planner_output"],
        )
    """

    label: str = "debug"
    keys_to_print: List[str] = Field(default_factory=list)
    print_full_json: bool = True

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        saved_files: List[str] = []

        if not debug_enabled():
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=json.dumps(
                                {
                                    "debug_status": "disabled",
                                    "label": self.label,
                                },
                                ensure_ascii=False,
                            )
                        )
                    ],
                ),
            )
            return

        for key in self.keys_to_print:
            value: Any = ctx.session.state.get(key, None)

            filename = f"{self.label}__{key}.json"
            path = write_snapshot(ctx=ctx, filename=filename, payload=value)
            saved_files.append(str(path))

            print_state_snapshot(
                label=self.label,
                key=key,
                value=value,
                snapshot_path=path,
                print_full_json=self.print_full_json,
            )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=json.dumps(
                            build_debug_event_payload(
                                label=self.label,
                                saved_files=saved_files,
                            ),
                            ensure_ascii=False,
                        )
                    )
                ],
            ),
        )


debug_after_planner_agent = DebugStatePrinterAgent(
    name="DebugAfterPlanner",
    description="Prints and saves planner_output.",
    label="01_after_planner",
    keys_to_print=["planner_output"],
    print_full_json=True,
)

debug_after_web_discovery_agent = DebugStatePrinterAgent(
    name="DebugAfterWebDiscovery",
    description="Prints and saves web_discovery_output.",
    label="02_after_web_discovery",
    keys_to_print=["web_discovery_output"],
    print_full_json=True,
)

debug_after_reranking_agent = DebugStatePrinterAgent(
    name="DebugAfterReranking",
    description="Prints and saves reranking_output.",
    label="03_after_reranking",
    keys_to_print=["reranking_output"],
    print_full_json=True,
)

debug_after_final_output_agent = DebugStatePrinterAgent(
    name="DebugAfterFinalOutput",
    description="Prints and saves final_output.",
    label="04_after_final_output",
    keys_to_print=["final_output"],
    print_full_json=True,
)
