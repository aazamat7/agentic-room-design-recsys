"""
Turn the result tables into shareable artifacts.

Produces, in ~/report/:
    RESULTS.md              GitHub-ready summary with markdown tables
    figures/<pair>.jpg      labelled side-by-side strips
                            (input / designer target / base / fine-tuned)
    figures/contact_sheet.jpg   all selected pairs stacked in one image

The figures are small enough to commit; the full output folders are not.

CPU only, no extra packages.
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PANEL = 420
HEADER = 40
GAP = 6
LABELS = ["INPUT (BEFORE)", "DESIGNER TARGET", "BASE MODEL", "FINE-TUNED (LoRA)"]
KEYS = ["input", "expected", "base", "lora"]


def _font(size=15):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit(img, size):
    canvas = Image.new("RGB", size, "white")
    c = img.convert("RGB")
    c.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(c, ((size[0] - c.width) // 2, (size[1] - c.height) // 2))
    return canvas


def strip(row, caption):
    """One labelled row: input / target / base / lora."""
    n = len(KEYS)
    w = PANEL * n + GAP * (n - 1)
    canvas = Image.new("RGB", (w, HEADER + PANEL + 30), "white")
    d = ImageDraw.Draw(canvas)
    f, fs = _font(15), _font(13)

    for i, (key, label) in enumerate(zip(KEYS, LABELS)):
        x = i * (PANEL + GAP)
        d.text((x + 8, 12), label, fill="black", font=f)
        path = row.get(key)
        if path and Path(path).exists():
            canvas.paste(fit(Image.open(path), (PANEL, PANEL)), (x, HEADER))
        else:
            d.rectangle([x, HEADER, x + PANEL, HEADER + PANEL], outline="#cccccc")
            d.text((x + 12, HEADER + PANEL // 2), "missing", fill="#999999", font=fs)

    d.text((8, HEADER + PANEL + 8), caption, fill="#333333", font=fs)
    return canvas


def md_table(df, cols=None, floatfmt="{:.3f}"):
    """Render a dataframe as a markdown table."""
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    lines = [header, sep]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_image_paths(base_dir):
    """
    Image paths live in the generation manifests, not in the metric tables.
    Scans for eval_manifest.json files and maps pair_id -> panel paths.
    """
    import json
    base = Path(base_dir).expanduser()
    mapping = {}
    for man in sorted(base.rglob("eval_manifest.json")):
        try:
            records = json.loads(man.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  could not read {man}: {type(e).__name__}")
            continue
        for r in records:
            pid = r.get("pair_id")
            if not pid:
                continue
            mapping[pid] = {k: r.get(k, "") for k in KEYS}
        print(f"  paths from {man.parent.name}: {len(records)} records")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="~/final_tables")
    ap.add_argument("--outputs", default="~",
                    help="Directory to scan for eval_manifest.json files "
                         "(the generation output folders live here).")
    ap.add_argument("--out", default="~/report")
    ap.add_argument("--top", type=int, default=6,
                    help="How many top-ranked pairs to render as figures.")
    ap.add_argument("--prefer-holdout", action="store_true", default=True,
                    help="Rank holdout pairs first among equally strong ones.")
    args = ap.parse_args()

    tdir = Path(args.tables).expanduser()
    out = Path(args.out).expanduser()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(tdir / "summary_by_split.csv")
    per_pair = pd.read_csv(tdir / "per_pair_full.csv")
    cand = pd.read_csv(tdir / "demo_candidates.csv")
    by_room = pd.read_csv(tdir / "by_room_type.csv") if (tdir / "by_room_type.csv").exists() else None
    by_style = pd.read_csv(tdir / "by_style.csv") if (tdir / "by_style.csv").exists() else None

    print("Locating image paths:")
    paths = load_image_paths(args.outputs)
    if not paths:
        print("  WARNING: no eval_manifest.json found under "
              f"{args.outputs} - figures will be blank.")

    # ---- pick the pairs to illustrate ----
    cand = cand.copy()
    cand["_holdout"] = (cand["split"] == "holdout").astype(int)
    order = ["signals_for_lora"] + (["_holdout"] if args.prefer_holdout else [])
    cand = cand.sort_values(order, ascending=False)
    # one per room type first, so the figures cover different spaces
    picked, seen = [], set()
    for _, r in cand.iterrows():
        if r["room_type"] in seen:
            continue
        picked.append(r)
        seen.add(r["room_type"])
        if len(picked) >= args.top:
            break
    for _, r in cand.iterrows():   # top up if not enough distinct room types
        if len(picked) >= args.top:
            break
        if r["pair_id"] not in [p["pair_id"] for p in picked]:
            picked.append(r)

    print(f"Rendering {len(picked)} figures...")
    strips = []
    missing_paths = 0
    for r in picked:
        row = per_pair[per_pair["pair_id"] == r["pair_id"]]
        if row.empty:
            continue
        row = row.iloc[0].to_dict()
        # Panel paths come from the generation manifest.
        panels = paths.get(r["pair_id"])
        if panels:
            row.update(panels)
        else:
            missing_paths += 1
        style = r.get("style_primary", "")
        caption = (f"{r['pair_id']}  |  {r['room_type']}  |  split={r['split']}"
                   f"{'  |  ' + str(style) if style and str(style) != 'nan' else ''}"
                   f"  |  signals favouring LoRA: {r['signals_for_lora']}/{r['signals_total']}")
        img = strip(row, caption)
        path = figs / f"{r['pair_id']}.jpg"
        img.save(path, quality=92)
        strips.append(img)
        print(f"  {path.name}")
    if missing_paths:
        print(f"  WARNING: {missing_paths} pair(s) had no manifest entry; "
              "their panels are blank.")

    if strips:
        w = max(s.width for s in strips)
        h = sum(s.height for s in strips) + GAP * (len(strips) - 1)
        sheet = Image.new("RGB", (w, h), "white")
        y = 0
        for s in strips:
            sheet.paste(s, (0, y))
            y += s.height + GAP
        sheet.save(figs / "contact_sheet.jpg", quality=88)
        print(f"  contact_sheet.jpg ({w}x{h})")

    # ---- markdown report ----
    rate_cols = ["split", "n"] + [c for c in summary.columns if c.endswith("_rate")]
    p_cols = ["split", "n"] + [c for c in summary.columns if c.endswith("_p")]

    md = []
    md.append("# Evaluation Results: Base vs Fine-Tuned (LoRA)\n")
    md.append("Comparison of the base image-editing model against the LoRA "
              "fine-tuned on paired before/after interior renovations.\n")
    md.append("Both versions were generated in a single run on identical inputs, "
              "with the same seed, step count and guidance settings, so any "
              "difference is attributable to the adapter rather than to sampling "
              "noise.\n")

    md.append("\n## Sets\n")
    md.append("| set | n | notes |")
    md.append("|---|---|---|")
    md.append("| holdout | 10 | never seen during fine-tuning: evidence of generalization |")
    md.append("| train | 21 | seen during fine-tuning: larger sample, expected to be optimistic |")
    md.append("\nThe two sets are reported separately rather than pooled into a "
              "single figure, because they support different claims.\n")

    md.append("\n## Headline: how often the fine-tuned output wins\n")
    md.append("Win rate = share of pairs where the fine-tuned output is closer to "
              "the designer target (or preferred by the judge).\n")
    md.append(md_table(summary[rate_cols]))
    md.append("\n### One-sided binomial p-values (against a 50% null)\n")
    md.append(md_table(summary[p_cols], floatfmt="{:.4f}"))

    md.append("\n## Metrics used\n")
    md.append("| signal | what it measures |")
    md.append("|---|---|")
    md.append("| DINOv2 | structural / semantic similarity to the designer target |")
    md.append("| CLIP-I | semantic similarity to the designer target |")
    md.append("| LPIPS | perceptual distance to the target (lower is better) |")
    md.append("| palette | dominant-colour similarity to the target, in CIELAB |")
    md.append("| Claude / Gemini | blind pairwise preference, randomized A/B order |")
    md.append("\nThe two LLM judges were run blind: neither was told which image "
              "came from which model, and the A/B position was randomized per item.\n")

    if by_room is not None:
        md.append("\n## By room type\n")
        cols = ["split", "room_type", "n"] + [c for c in by_room.columns if c.endswith("_rate")]
        md.append(md_table(by_room[[c for c in cols if c in by_room.columns]]))

    if by_style is not None:
        md.append("\n## By design style\n")
        cols = ["split", "style_primary", "n"] + [c for c in by_style.columns if c.endswith("_rate")]
        md.append(md_table(by_style[[c for c in cols if c in by_style.columns]]))

    md.append("\n## Example comparisons\n")
    md.append("Each strip shows, left to right: the original room, the professional "
              "designer result, the base model output, and the fine-tuned output.\n")
    for r in picked:
        md.append(f"\n**{r['pair_id']}** — {r['room_type']}, `{r['split']}`, "
                  f"{r['signals_for_lora']}/{r['signals_total']} signals favour the fine-tuned output\n")
        md.append(f"![{r['pair_id']}](figures/{r['pair_id']}.jpg)\n")

    md.append("\n## Files\n")
    md.append("| file | contents |")
    md.append("|---|---|")
    md.append("| `summary_by_split.csv` | headline win rates and p-values |")
    md.append("| `by_room_type.csv` | win rates per room type |")
    md.append("| `by_style.csv` | win rates per design style |")
    md.append("| `per_pair_full.csv` | every metric and verdict, one row per pair |")
    md.append("| `demo_candidates.csv` | pairs ranked by agreement across signals |")

    (out / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")

    # copy the tables next to the report so the folder is self-contained
    tables_out = out / "tables"
    tables_out.mkdir(exist_ok=True)
    for csv in tdir.glob("*.csv"):
        (tables_out / csv.name).write_text(csv.read_text(encoding="utf-8"),
                                           encoding="utf-8")

    print(f"\nReport written to {out / 'RESULTS.md'}")
    print(f"Figures in {figs}")
    print(f"Tables copied to {tables_out}")


if __name__ == "__main__":
    main()
