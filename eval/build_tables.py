"""
Assemble the final result tables from the metric and judge CSVs.
 
Reads (missing inputs are skipped with a warning):
    ~/metrics_train21/metrics.csv
    ~/metrics_holdout10/metrics.csv
    ~/judges_train21/judges.csv
    ~/judges_holdout10/judges.csv
    ~/pairs_with_style.csv          (optional, adds the style breakdown)
 
Writes into ~/final_tables/:
    per_pair_full.csv      one row per comparison, every metric and verdict
    summary_by_split.csv   headline table: win rates for train vs holdout
    by_room_type.csv       win rates per room type, split-aware
    by_style.csv           win rates per design style (needs the style CSV)
    demo_candidates.csv    pairs ranked by how many signals favour the LoRA
 
CPU only, runs in seconds.
"""
 
import argparse
from math import comb
from pathlib import Path
 
import pandas as pd
 
 
WIN_COLS = ["lora_closer_clip_i", "lora_closer_dino",
            "lora_closer_lpips", "lora_closer_palette"]
 
 
def binom_p(k, n):
    """One-sided binomial p-value against a 50% null."""
    if n == 0:
        return None
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
 
 
def read_csv(path, label):
    p = Path(path).expanduser()
    if not p.exists():
        print(f"  missing, skipped: {p}")
        return None
    df = pd.read_csv(p)
    print(f"  {label}: {len(df)} rows")
    return df
 
 
def load_side(metrics_path, judges_path, split_label):
    """Merge the metric and judge tables for one split."""
    m = read_csv(metrics_path, f"metrics [{split_label}]")
    j = read_csv(judges_path, f"judges  [{split_label}]")
    if m is None and j is None:
        return None
    if m is None:
        j["split"] = split_label
        return j
    if "split" not in m.columns or m["split"].isna().all():
        m["split"] = split_label
    m["split"] = m["split"].fillna(split_label).replace("", split_label)
    if j is None:
        return m
    keep = [c for c in ("pair_id", "claude_pick", "claude_reason",
                        "gemini_pick", "gemini_reason") if c in j.columns]
    return m.merge(j[keep], on="pair_id", how="left")
 
 
def attach_styles(df, style_csv):
    """
    Join the style labels. The metric tables use ids like
    'kitchen__8e4a18cdf683b7367c' while the style CSV stores only the hash,
    so the room-type prefix is stripped to form the join key.
    """
    s = read_csv(style_csv, "styles")
    if s is None:
        return df
    cols = [c for c in ("pair_id", "style_primary", "style_secondary",
                        "style_desc", "style_conf") if c in s.columns]
    s = s[cols].copy()
    s["_key"] = s["pair_id"].astype(str)
    df = df.copy()
    df["_key"] = df["pair_id"].astype(str).str.split("__", n=1).str[-1]
    merged = df.merge(s.drop(columns=["pair_id"]), on="_key", how="left")
    unmatched = merged["style_primary"].isna().sum() if "style_primary" in merged else len(merged)
    print(f"  styles joined, unmatched rows: {unmatched}")
    return merged.drop(columns=["_key"])
 
 
def win_summary(df, group_cols=None):
    """Win counts, rates and p-values for every available signal."""
    signals = [c for c in WIN_COLS if c in df.columns]
    judge_cols = [c for c in ("claude_pick", "gemini_pick") if c in df.columns]
 
    def one(sub):
        out = {"n": len(sub)}
        for c in signals:
            v = sub[c].dropna().astype(bool)
            if len(v):
                k = int(v.sum())
                out[c.replace("lora_closer_", "") + "_wins"] = k
                out[c.replace("lora_closer_", "") + "_rate"] = round(k / len(v), 3)
                out[c.replace("lora_closer_", "") + "_p"] = round(binom_p(k, len(v)), 4)
        for c in judge_cols:
            v = sub[c].dropna()
            if len(v):
                k = int((v == "lora").sum())
                name = c.replace("_pick", "")
                out[name + "_wins"] = k
                out[name + "_rate"] = round(k / len(v), 3)
                out[name + "_p"] = round(binom_p(k, len(v)), 4)
        if judge_cols == ["claude_pick", "gemini_pick"]:
            both = sub.dropna(subset=judge_cols)
            if len(both):
                out["judges_agree_rate"] = round(
                    (both["claude_pick"] == both["gemini_pick"]).mean(), 3)
                out["both_chose_lora"] = int(
                    ((both["claude_pick"] == "lora") &
                     (both["gemini_pick"] == "lora")).sum())
        return pd.Series(out)
 
    if group_cols:
        return df.groupby(group_cols, dropna=False).apply(
            one, include_groups=False).reset_index()
    return one(df).to_frame().T
 
 
def demo_ranking(df):
    """
    Rank pairs by how many independent signals favour the LoRA.
    The top entries are the safest picks for the demo slides.
    """
    d = df.copy()
    score = pd.Series(0, index=d.index, dtype=int)
    used = []
    for c in WIN_COLS:
        if c in d.columns:
            score += d[c].fillna(False).astype(bool).astype(int)
            used.append(c)
    for c in ("claude_pick", "gemini_pick"):
        if c in d.columns:
            score += (d[c] == "lora").astype(int)
            used.append(c)
    d["signals_for_lora"] = score
    d["signals_total"] = len(used)
    cols = [c for c in ("pair_id", "room_type", "split", "style_primary",
                        "signals_for_lora", "signals_total",
                        "claude_pick", "gemini_pick") if c in d.columns]
    return d.sort_values(["signals_for_lora", "split"],
                         ascending=[False, True])[cols]
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="~", help="Directory holding the result folders")
    ap.add_argument("--style-csv", default="~/pairs_with_style.csv")
    ap.add_argument("--out", default="~/final_tables")
    args = ap.parse_args()
 
    base = Path(args.base).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
 
    print("Loading inputs:")
    train = load_side(base / "metrics_train21" / "metrics.csv",
                      base / "judges_train21" / "judges.csv", "train")
    hold = load_side(base / "metrics_holdout10" / "metrics.csv",
                     base / "judges_holdout10" / "judges.csv", "holdout")
 
    parts = [p for p in (train, hold) if p is not None]
    if not parts:
        raise SystemExit("No input tables found. Check --base.")
    df = pd.concat(parts, ignore_index=True)
    df = attach_styles(df, args.style_csv)
 
    df.to_csv(out / "per_pair_full.csv", index=False)
 
    # 1) headline table
    summary = win_summary(df, ["split"])
    summary.to_csv(out / "summary_by_split.csv", index=False)
 
    pooled = win_summary(df)
    pooled.insert(0, "split", "POOLED")
    pd.concat([summary, pooled], ignore_index=True).to_csv(
        out / "summary_by_split.csv", index=False)
 
    # 2) room types
    win_summary(df, ["split", "room_type"]).to_csv(
        out / "by_room_type.csv", index=False)
 
    # 3) styles
    if "style_primary" in df.columns:
        win_summary(df, ["split", "style_primary"]).to_csv(
            out / "by_style.csv", index=False)
 
    # 4) demo candidates
    ranking = demo_ranking(df)
    ranking.to_csv(out / "demo_candidates.csv", index=False)
 
    # ---- console report ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
 
    print("\n" + "=" * 70)
    print("HEADLINE: LoRA win rate by split")
    print("=" * 70)
    show = [c for c in pd.concat([summary, pooled]).columns
            if c in ("split", "n") or c.endswith("_rate") or c.endswith("_p")]
    print(pd.concat([summary, pooled], ignore_index=True)[show].to_string(index=False))
 
    if "style_primary" in df.columns:
        print("\n" + "=" * 70)
        print("BY STYLE")
        print("=" * 70)
        bs = win_summary(df, ["split", "style_primary"])
        cols = [c for c in bs.columns
                if c in ("split", "style_primary", "n") or c.endswith("_rate")]
        print(bs[cols].to_string(index=False))
 
    print("\n" + "=" * 70)
    print("TOP DEMO CANDIDATES (most signals favouring the LoRA)")
    print("=" * 70)
    print(ranking.head(12).to_string(index=False))
 
    print(f"\nAll tables written to {out}")
 
 
if __name__ == "__main__":
    main()
