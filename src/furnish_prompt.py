"""
furnish_prompt.py — rewrite the generation prompt so the model furnishes the
room instead of preserving it empty.

Two things in the original prompt keep the rooms bare:

  1. "while retaining the recognizable identity and perspective of the
     original room" reads as an instruction to leave the room as it is.
     An empty input room stays empty. Replacing it with a narrow constraint
     on architecture and camera keeps the comparison fair (same viewpoint,
     same walls and windows) without asking the model to preserve emptiness.

  2. The design direction is mostly surfaces — walls, flooring, shelving,
     lighting. An explicit staging clause is added so furniture is requested
     directly rather than left implicit.

Usage inside the generation script, right after the prompt is read from the
metadata record:

    from furnish_prompt import furnish
    prompt = furnish(prompt)

Set FURNISH=0 in the environment to disable the rewrite and reproduce the
original behaviour, which is what the control arm of the comparison needs.
"""

from __future__ import annotations

import os
import re

# The clause that tells the model to leave the room alone. Matched loosely
# because the wording varies slightly between records.
RETAIN_CLAUSE = re.compile(
    r"\s*while\s+retaining\s+the\s+recognizable\s+identity\s+and\s+perspective"
    r"\s+of\s+the\s+original\s+room\s*\.?",
    re.IGNORECASE,
)

# Constraint that replaces it: keeps the geometry fixed so base and LoRA
# outputs remain directly comparable, without implying an empty result.
GEOMETRY_CLAUSE = (
    "Keep the original room architecture, window and door positions, "
    "ceiling shape and camera viewpoint unchanged."
)

STAGING_CLAUSE = (
    "Fully furnish and stage the room as a finished, magazine-ready interior: "
    "include appropriate seating, tables, soft furnishings, rugs, window "
    "treatments, lighting fixtures, artwork and styling accessories suited to "
    "the room type. The room must not be left empty or unfurnished."
)

FURNITURE_HINTS = (
    "sofa", "couch", "chair", "seating", "table", "bed", "desk", "bench",
    "stool", "cabinet", "dresser", "furnish", "furniture", "furnishings",
)


def mentions_furniture(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in FURNITURE_HINTS)


def furnish(prompt: str, force: bool = False) -> str:
    """Return the prompt rewritten to request a furnished room.

    `force` adds the staging clause even when the design direction already
    names furniture, which is useful when a design direction mentions one
    item in passing but the room still comes back bare.
    """
    if not force and os.environ.get("FURNISH", "1") == "0":
        return prompt

    text = (prompt or "").strip()
    if not text:
        return text

    had_retain = bool(RETAIN_CLAUSE.search(text))
    text = RETAIN_CLAUSE.sub("", text).strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and not text.endswith((".", "!", "?")):
        text += "."

    parts = [text]
    if force or not mentions_furniture(text):
        parts.append(STAGING_CLAUSE)
    if had_retain:
        parts.append(GEOMETRY_CLAUSE)

    return " ".join(p.strip() for p in parts if p.strip())


if __name__ == "__main__":
    import glob
    import sys

    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/new21_outputs/*/prompt.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no prompt files matched {pattern}")
        raise SystemExit(1)

    bare = sum(1 for f in files if not mentions_furniture(open(f).read()))
    print(f"{len(files)} prompts, {bare} never mention furniture\n")

    for f in files[:2]:
        original = open(f, encoding="utf-8").read().strip()
        print("=" * 70)
        print(f"FILE: {f}")
        print("-" * 70)
        print("BEFORE:\n" + original)
        print("-" * 70)
        print("AFTER:\n" + furnish(original))
        print()
