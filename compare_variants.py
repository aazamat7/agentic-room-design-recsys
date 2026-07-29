"""
Score prompt variants on whether rooms actually got furnished.

An empty room is visually smooth: large flat wall and floor areas. Furniture
adds edges, occlusions and texture, so mean gradient magnitude separates bare
from furnished. The designer target sets the reference level for a properly
staged room, giving a scale-free ratio comparable across room types.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOTS = {
    "A_control": [Path.home() / "promptA_control",
                  Path("/tmp/prompt_tests/promptA_control")],
    "B_furnish": [Path.home() / "promptB_furnish"],
    "C_cfg6.5":  [Path.home() / "promptC_cfg65"],
}
PANELS = ["input.png", "expected.png", "base_generated.png", "lora_generated.png"]

def energy(p, size=384):
    try:
        im = Image.open(p).convert("L").resize((size, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32) / 255.0
    gy, gx = np.gradient(a)
    return float(np.hypot(gx, gy).mean())

rows = []
for label, cands in ROOTS.items():
    root = next((c for c in cands if c.is_dir()), None)
    if root is None:
        continue
    for s in sorted(d for d in root.iterdir() if d.is_dir()):
        e = {n: energy(s / n) for n in PANELS}
        if not e["expected.png"]:
            continue
        ref = e["expected.png"]
        rows.append((label, s.name[:28],
                     e["input.png"] / ref if e["input.png"] else float("nan"),
                     e["base_generated.png"] / ref if e["base_generated.png"] else float("nan"),
                     e["lora_generated.png"] / ref if e["lora_generated.png"] else float("nan")))

if not rows:
    print("no samples found"); sys.exit(1)

hdr = f"{'variant':<10} {'pair':<28} {'input':>7} {'base':>7} {'lora':>7}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:<10} {r[1]:<28} {r[2]:>7.2f} {r[3]:>7.2f} {r[4]:>7.2f}")
print("-" * len(hdr))
print("\nmean fill_ratio vs designer target (1.00 = comparably furnished):")
for label in ROOTS:
    sub = [r for r in rows if r[0] == label]
    if sub:
        mi, mb, ml = (np.nanmean([r[i] for r in sub]) for i in (2, 3, 4))
        print(f"  {label:<10} input {mi:.2f} | base {mb:.2f} | lora {ml:.2f}"
              f" | lora gain over input {ml - mi:+.2f}")
