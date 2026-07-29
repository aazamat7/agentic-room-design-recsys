"""
Generate BASE and LoRA outputs for N training-set pairs using DiffSynth-Studio.

This mirrors the validation script that already runs successfully on this
machine: same pipeline class, same loading procedure, same generation settings
(seed / steps / cfg_scale / zero_cond_t / max pixels). Matching those settings
is what makes the new images directly comparable with the existing holdout
outputs.

Both passes run in ONE process:
  pass 1 -> base model, no LoRA
  pass 2 -> same pipeline after load_lora()
Identical seed and settings in both passes, so any visible difference is
attributable to the LoRA rather than to sampling noise.

------------------------------------------------------------------------
RUN FROM THE DiffSynth-Studio DIRECTORY:

    cd ~/DiffSynth-Studio
    python ~/generate_base_and_lora_diffsynth.py --n 21

DiffSynth resolves model files against ./models in the current working
directory. Running from ~/DiffSynth-Studio reuses the 54 GB of weights that
are already downloaded there, so nothing is fetched over the network.
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from furnish_prompt import furnish
import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

# Cache locations must be configured before importing HF or DiffSynth packages.
os.environ.setdefault("HF_HOME", "/home/jupyter/data/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/home/jupyter/data/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE",
                      "/home/jupyter/data/.cache/huggingface/transformers")
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "HuggingFace")

import torch
from PIL import Image

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


DATASET_ROOT = Path(os.environ.get(
    "DATASET_ROOT", "/home/jupyter/qwen2511_hgtv_augmented_lora_dataset"))

DEFAULT_LORA = ("/home/jupyter/recovered/qwen2511_lora_output/"
                "step-2880.safetensors")

# Generation settings copied from the working validation run so that the new
# images are directly comparable with the existing holdout outputs.
SEED = 42
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 4.0
MAX_PIXELS = 1048576


# ----------------------------------------------------------------------
# Manifest handling
# ----------------------------------------------------------------------
def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL metadata file."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Metadata is empty: {path}")
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of records in {path}")
    return data


def normalized_path(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").lstrip("./")


def pair_group_id(record: dict[str, Any]) -> str:
    """Collapse the original and its augmented variants into one pair id."""
    for key in ("pair_id", "image_id", "source_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    stem = Path(normalized_path(record.get("edit_image"))).stem
    return re.sub(r"_(?:orig|v[0-9]+)$", "", stem, flags=re.IGNORECASE)


def is_original_variant(record: dict[str, Any]) -> bool:
    stem = Path(normalized_path(record.get("edit_image"))).stem.lower()
    return stem.endswith(("_orig", "-orig", "_original", "-original"))


def room_type_of(group_id: str) -> str:
    """Pair ids look like 'living_room__e21f1bafadc124b825'."""
    return group_id.split("__")[0] if "__" in group_id else "unknown"


def resolve_image_path(value: Any) -> Path:
    path = Path(str(value))
    resolved = path if path.is_absolute() else DATASET_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved


def unique_pairs(metadata_path: Path) -> list[dict[str, Any]]:
    """One record per original pair, augmented duplicates removed."""
    records = load_metadata(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[pair_group_id(record)].append(record)

    selected = []
    for group_id in sorted(grouped):
        variants = grouped[group_id]
        originals = [r for r in variants if is_original_variant(r)]
        chosen = dict(originals[0] if originals else sorted(
            variants, key=lambda r: normalized_path(r.get("edit_image")))[0])
        chosen["_group_id"] = group_id
        chosen["_room_type"] = room_type_of(group_id)
        selected.append(chosen)
    return selected



def record_key(record):
    """Identity of a record, used to separate holdout from training rows."""
    return (normalized_path(record.get("edit_image")),
            normalized_path(record.get("image")))


def holdout_pairs(all_metadata: Path, train_metadata: Path):
    """
    Holdout pairs are the records present in metadata_all.json but absent from
    the training metadata. Augmented variants are collapsed to one record per
    pair, preferring the original.
    """
    train_records = load_metadata(train_metadata)
    all_records = load_metadata(all_metadata)
    train_keys = {record_key(r) for r in train_records}
    held = [r for r in all_records if record_key(r) not in train_keys]

    print(f"Training records: {len(train_records)}")
    print(f"All records: {len(all_records)}")
    print(f"Holdout augmented records: {len(held)}")
    if not held:
        raise RuntimeError("No holdout records found. Check the metadata files.")

    grouped = defaultdict(list)
    for r in held:
        grouped[pair_group_id(r)].append(r)

    selected = []
    for group_id in sorted(grouped):
        variants = grouped[group_id]
        originals = [r for r in variants if is_original_variant(r)]
        chosen = dict(originals[0] if originals else sorted(
            variants, key=lambda r: normalized_path(r.get("edit_image")))[0])
        chosen["_group_id"] = group_id
        chosen["_room_type"] = room_type_of(group_id)
        selected.append(chosen)

    print(f"Unique holdout pairs: {len(selected)}")
    return selected


def stratified_sample(records, n, exclude_ids=None, seed=42, per_type_cap=None):
    """Pick n pairs spread across room types, deterministically."""
    exclude_ids = exclude_ids or set()
    random.seed(seed)
    pool = [r for r in records if r["_group_id"] not in exclude_ids]

    by_type = defaultdict(list)
    for r in pool:
        by_type[r["_room_type"]].append(r)
    for t in by_type:
        by_type[t].sort(key=lambda r: r["_group_id"])
        random.shuffle(by_type[t])

    selected, idx = [], {t: 0 for t in by_type}
    types = sorted(by_type, key=lambda t: -len(by_type[t]))
    while len(selected) < n and any(idx[t] < len(by_type[t]) for t in types):
        for t in types:
            if len(selected) >= n:
                break
            if per_type_cap is not None and \
               sum(1 for s in selected if s["_room_type"] == t) >= per_type_cap:
                continue
            if idx[t] < len(by_type[t]):
                selected.append(by_type[t][idx[t]])
                idx[t] += 1
    return selected[:n]


# ----------------------------------------------------------------------
# Image + pipeline helpers (identical behaviour to the validation script)
# ----------------------------------------------------------------------
def fit_to_model(image: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Preserve aspect ratio, stay under max_pixels, force multiples of 16."""
    image = image.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width, height = max(16, int(width * scale)), max(16, int(height * scale))
    width = max(16, width - width % 16)
    height = max(16, height - height % 16)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def load_pipeline() -> QwenImagePipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the PyTorch installation.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading Qwen-Image-Edit-2511 (base, no LoRA)...")

    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="processor/"),
    )
    print("Base model loaded.")
    return pipeline


def run_one(pipeline, source_for_model, prompt, steps, cfg_scale, seed):
    width, height = source_for_model.size
    return pipeline(
        prompt=prompt,
        edit_image=[source_for_model],
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=steps,
        cfg_scale=cfg_scale,
        zero_cond_t=True,
    )


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(DATASET_ROOT / "metadata_train.json"),
                    help="Training metadata (augmented variants are collapsed).")
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--exclude-ids", default="",
                    help="Comma-separated pair ids, or a path to a JSON list.")
    ap.add_argument("--per-type-cap", type=int, default=None)
    ap.add_argument("--lora-path", default=DEFAULT_LORA)
    ap.add_argument("--output-dir", default="/home/jupyter/new21_outputs")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--cfg-scale", type=float, default=CFG_SCALE)
    ap.add_argument("--split-label", default="train",
                    help="Recorded in the eval manifest for reporting.")
    ap.add_argument("--holdout", action="store_true",
                    help="Generate for the HOLDOUT pairs instead of training pairs. "
                         "Holdout is derived by diffing metadata_all.json against "
                         "the training metadata, matching the earlier validation run.")
    ap.add_argument("--all-metadata",
                    default=str(DATASET_ROOT / "metadata_all.json"))
    ap.add_argument("--train-metadata-for-diff",
                    default=str(DATASET_ROOT / "metadata.json"),
                    help="Training metadata used to identify holdout rows.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate selection and paths without loading the model.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_ids = set()
    if args.exclude_ids:
        if os.path.exists(args.exclude_ids):
            exclude_ids = set(json.load(open(args.exclude_ids)))
        else:
            exclude_ids = {s.strip() for s in args.exclude_ids.split(",") if s.strip()}

    if args.holdout:
        pairs = holdout_pairs(Path(args.all_metadata),
                              Path(args.train_metadata_for_diff))
    else:
        pairs = unique_pairs(Path(args.metadata))
    print(f"Unique pairs available: {len(pairs)}")
    print("Room types:", dict(Counter(p["_room_type"] for p in pairs)))

    available = [p for p in pairs if p["_group_id"] not in exclude_ids]
    if len(available) < args.n:
        raise SystemExit(f"Only {len(available)} pairs available, but --n is {args.n}.")

    if args.holdout:
        # Deterministic order, same as the earlier validation run.
        chosen = [p for p in pairs if p["_group_id"] not in exclude_ids][:args.n]
    else:
        chosen = stratified_sample(pairs, args.n, exclude_ids,
                                   args.seed, args.per_type_cap)
    print(f"Selected {len(chosen)} pairs")
    print("Selection spread:", dict(Counter(p["_room_type"] for p in chosen)))

    # Verify every image exists before spending GPU time.
    prepared = []
    for i, rec in enumerate(chosen, 1):
        src = resolve_image_path(rec["edit_image"])
        tgt = resolve_image_path(rec["image"])
        prompt = str(rec.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Missing prompt for {rec['_group_id']}")
        prompt = furnish(prompt)
        prepared.append((i, rec, src, tgt, prompt))
    print(f"All {len(prepared)} input/target images found.")

    if args.dry_run:
        print("\n[dry-run] Selection and paths verified. "
              "Model not loaded, nothing generated.")
        for i, rec, src, tgt, prompt in prepared[:3]:
            print(f"  {i:02d} {rec['_group_id']}")
            print(f"     input : {src}")
            print(f"     target: {tgt}")
        return

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint missing: {lora_path}")

    pipeline = load_pipeline()

    # ---------------- PASS 1: BASE (no LoRA) ----------------
    print("\n=== PASS 1/2: base model ===")
    for i, rec, src, tgt, prompt in prepared:
        gid = rec["_group_id"]
        sample_dir = out_dir / f"{i:02d}_{gid}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_original = Image.open(src).convert("RGB")
        target = Image.open(tgt).convert("RGB")
        source_for_model = fit_to_model(source_original)

        source_original.save(sample_dir / "input.png")
        target.save(sample_dir / "expected.png")
        (sample_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        print(f"[base {i}/{len(prepared)}] {gid} "
              f"({source_for_model.width}x{source_for_model.height})", flush=True)

        generated = run_one(pipeline, source_for_model, prompt,
                            args.steps, args.cfg_scale, args.seed)
        generated.save(sample_dir / "base_generated.png")
        generated.close()
        torch.cuda.empty_cache()

    # ---------------- PASS 2: LoRA ----------------
    print(f"\n=== PASS 2/2: loading LoRA {lora_path} ===")
    pipeline.load_lora(pipeline.dit, str(lora_path))
    print("LoRA loaded.")

    for i, rec, src, tgt, prompt in prepared:
        gid = rec["_group_id"]
        sample_dir = out_dir / f"{i:02d}_{gid}"
        source_for_model = fit_to_model(Image.open(src).convert("RGB"))

        print(f"[lora {i}/{len(prepared)}] {gid}", flush=True)

        generated = run_one(pipeline, source_for_model, prompt,
                            args.steps, args.cfg_scale, args.seed)
        generated.save(sample_dir / "lora_generated.png")
        generated.close()
        torch.cuda.empty_cache()

    # ---------------- manifest for the evaluation step ----------------
    eval_manifest = []
    for i, rec, src, tgt, prompt in prepared:
        gid = rec["_group_id"]
        sample_dir = out_dir / f"{i:02d}_{gid}"
        eval_manifest.append({
            "index": i,
            "pair_id": gid,
            "room_type": rec["_room_type"],
            "split": "holdout" if args.holdout else args.split_label,
            "prompt": prompt,
            "input": str(sample_dir / "input.png"),
            "expected": str(sample_dir / "expected.png"),
            "base": str(sample_dir / "base_generated.png"),
            "lora": str(sample_dir / "lora_generated.png"),
            "seed": args.seed,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
        })
    (out_dir / "eval_manifest.json").write_text(
        json.dumps(eval_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. {len(eval_manifest)} pairs generated (base + LoRA).")
    print(f"Eval manifest: {out_dir / 'eval_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
