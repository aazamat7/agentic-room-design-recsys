"""
Evaluate base vs LoRA outputs with a PANEL of complementary metrics, so the
result does not rest on LLM judges alone.

Two independent families of signal:

 A) Deterministic / embedding metrics (reproducible, no LLM bias)
      - CLIP-I : cosine(generated, HGTV target)      -> design closeness
      - CLIP-T : cosine(generated, design prompt)     -> prompt adherence
      - DINOv2 : cosine(generated, target)            -> design closeness
                 cosine(generated, input)             -> identity/layout kept
      - LPIPS  : perceptual distance(generated,target) (lower = closer)
      - SSIM   : structural(generated, input)          (moderate is good)

 B) LLM-as-judge, done to be as trustworthy as possible
      - PAIRWISE A/B (not 1-10 absolute scoring)
      - BLIND (judge is not told which is base / LoRA)
      - RANDOMIZED A/B position per item (kills position bias)
      - TWO judges (Claude + Gemini) + inter-judge agreement (Cohen's kappa)

The headline claim you want to support ("LoRA output is closer to the real
HGTV designer image than base") is directly measured by CLIP-I / DINOv2 /
LPIPS-vs-target AND cross-checked by the two judges. When both families agree,
the result is defensible without human raters.

Run this on cpu-workbench (fine on CPU/GPU). Metrics need:
    pip install torch torchvision open_clip_torch lpips scikit-image pillow pandas numpy
LLM judges need API access:
    - Claude:  ANTHROPIC_API_KEY  (pip install anthropic)
    - Gemini:  Vertex AI creds    (pip install google-cloud-aiplatform)  OR
               GOOGLE_API_KEY      (pip install google-generativeai)
Judges are OPTIONAL: pass --no-judges to run metrics only.
"""

import os
import io
import json
import base64
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch


# ----------------------------------------------------------------------
# Embedding / perceptual metrics
# ----------------------------------------------------------------------
class MetricBank:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip = None
        self._clip_pre = None
        self._clip_tok = None
        self._dino = None
        self._lpips = None
        self._aesthetic = None

    # ---- CLIP (open_clip) ----
    def _ensure_clip(self):
        if self._clip is None:
            import open_clip
            self._clip, _, self._clip_pre = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._clip_tok = open_clip.get_tokenizer("ViT-B-32")
            self._clip.to(self.device).eval()

    def clip_image_embed(self, img: Image.Image):
        self._ensure_clip()
        x = self._clip_pre(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self._clip.encode_image(x)
        return torch.nn.functional.normalize(e, dim=-1)

    def clip_text_embed(self, text: str):
        self._ensure_clip()
        tok = self._clip_tok([text]).to(self.device)
        with torch.no_grad():
            e = self._clip.encode_text(tok)
        return torch.nn.functional.normalize(e, dim=-1)

    def clip_i(self, gen, target):
        return float((self.clip_image_embed(gen) * self.clip_image_embed(target)).sum())

    def clip_t(self, gen, prompt):
        return float((self.clip_image_embed(gen) * self.clip_text_embed(prompt)).sum())

    # ---- DINOv2 (torch.hub) ----
    def _ensure_dino(self):
        if self._dino is None:
            self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self._dino.to(self.device).eval()

    def dino_embed(self, img: Image.Image):
        self._ensure_dino()
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = t(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self._dino(x)
        return torch.nn.functional.normalize(e, dim=-1)

    def dino_sim(self, a, b):
        return float((self.dino_embed(a) * self.dino_embed(b)).sum())

    # ---- LPIPS ----
    def _ensure_lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()

    def lpips_dist(self, a, b):
        self._ensure_lpips()
        import torchvision.transforms as T
        t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        xa = (t(a.convert("RGB")) * 2 - 1).unsqueeze(0).to(self.device)
        xb = (t(b.convert("RGB")) * 2 - 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            d = self._lpips(xa, xb)
        return float(d.item())

    # ---- SSIM (structure preservation vs input) ----
    # ---- LAION aesthetic predictor (small head on top of CLIP embeddings) ----
    def _ensure_aesthetic(self):
        """
        Loads a ~4 MB linear head trained on human aesthetic ratings.
        Returns False (and disables the metric) if the weights cannot be
        fetched, so the run never fails because of an optional metric.
        """
        if self._aesthetic is not None:
            return self._aesthetic is not False
        try:
            import urllib.request
            self._ensure_clip()
            url = ("https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
                   "sa_0_4_vit_b_32_linear.pth")
            cache = Path.home() / ".cache" / "aesthetic"
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / "sa_0_4_vit_b_32_linear.pth"
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            head = torch.nn.Linear(512, 1)
            head.load_state_dict(torch.load(path, map_location="cpu"))
            self._aesthetic = head.to(self.device).eval()
            return True
        except Exception as e:
            print(f"  (aesthetic score unavailable, skipping: {type(e).__name__})")
            self._aesthetic = False
            return False

    def aesthetic(self, img):
        """LAION aesthetic score, roughly 1-10. None if unavailable."""
        if not self._ensure_aesthetic():
            return None
        try:
            with torch.no_grad():
                return float(self._aesthetic(self.clip_image_embed(img).float()).item())
        except Exception:
            return None

    # ---- CLIP-IQA style quality probes (reuses the already-loaded CLIP) ----
    _IQA_PAIRS = {
        "quality": ("a sharp, high quality photograph",
                    "a blurry, low quality photograph"),
        "realism": ("a real photograph of a room",
                    "a computer generated fake looking image"),
    }

    def clip_iqa(self, img, axis="quality"):
        """
        No-reference quality probe: relative affinity to a positive vs negative
        text prompt. Returns 0-1, higher is better. Needs no extra packages.
        """
        try:
            pos, neg = self._IQA_PAIRS[axis]
            e = self.clip_image_embed(img)
            sp = float((e * self.clip_text_embed(pos)).sum())
            sn = float((e * self.clip_text_embed(neg)).sum())
            return float(np.exp(sp * 100) / (np.exp(sp * 100) + np.exp(sn * 100)))
        except Exception:
            return None

    @staticmethod
    def ssim(a, b):
        from skimage.metrics import structural_similarity as sk_ssim
        a = np.array(a.convert("L").resize((256, 256)))
        b = np.array(b.convert("L").resize((256, 256)))
        return float(sk_ssim(a, b))


# ----------------------------------------------------------------------
# Palette metrics (dependency-free: PIL + numpy + scikit-image)
# ----------------------------------------------------------------------
# Colour vocabulary covering common interior-design terms, including the
# palette names reported in current designer trend surveys.
NAMED_COLORS = {
    "white": (245, 245, 245), "off-white": (240, 235, 225), "cream": (245, 238, 220),
    "beige": (222, 205, 180), "ivory": (255, 250, 235), "greige": (200, 190, 178),
    "grey": (140, 140, 140), "gray": (140, 140, 140), "charcoal": (60, 60, 62),
    "black": (25, 25, 25), "taupe": (150, 135, 120),
    "brown": (120, 85, 60), "chocolate": (75, 52, 40), "walnut": (95, 70, 50),
    "oak": (190, 155, 110), "wood": (160, 120, 80), "tan": (200, 170, 130),
    "terracotta": (190, 105, 75), "rust": (165, 85, 50), "clay": (185, 120, 95),
    "sage": (155, 170, 145), "green": (85, 120, 85), "olive": (120, 125, 75),
    "emerald": (45, 120, 90), "forest": (50, 80, 55),
    "blue": (75, 110, 155), "navy": (35, 50, 85), "teal": (55, 120, 125),
    "cornflower": (120, 150, 215), "powder blue": (175, 200, 220),
    "burgundy": (110, 35, 50), "red": (165, 55, 50), "maroon": (95, 40, 45),
    "pink": (225, 175, 180), "powder pink": (235, 200, 200), "blush": (230, 190, 185),
    "butter yellow": (240, 225, 160), "yellow": (225, 200, 90), "mustard": (200, 160, 60),
    "gold": (200, 165, 90), "brass": (185, 150, 95), "bronze": (150, 110, 70),
    "copper": (185, 115, 80), "silver": (190, 190, 195), "chrome": (200, 205, 210),
    "cognac": (150, 95, 55), "camel": (195, 155, 110), "pistachio": (190, 205, 150),
}


def dominant_palette(img, n_colors=8):
    """Dominant colours via PIL median-cut quantisation (no extra deps).

    Images with few distinct colours yield fewer palette entries than
    requested, so the actual palette length is used and empty bins dropped.
    """
    small = img.convert("RGB").resize((200, 200))
    q = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
    pal = q.getpalette() or []
    idx = np.array(q)
    n_avail = max(1, min(n_colors, len(pal) // 3))
    counts = np.bincount(idx.ravel(), minlength=n_avail).astype(float)[:n_avail]
    colors, weights = [], []
    for i in range(n_avail):
        if counts[i] <= 0:
            continue
        colors.append(tuple(int(v) for v in pal[i * 3:i * 3 + 3]))
        weights.append(counts[i])
    if not colors:
        colors, weights = [(0, 0, 0)], [1.0]
    weights = np.array(weights, dtype=float)
    return colors, weights / weights.sum()


def _to_lab(rgb_list):
    from skimage.color import rgb2lab
    arr = np.array(rgb_list, dtype=np.float64).reshape(-1, 1, 3) / 255.0
    return rgb2lab(arr).reshape(-1, 3)


def palette_similarity(img_a, img_b, n_colors=8):
    """
    Weighted palette distance in CIELAB, mapped to a 0-1 similarity.
    Symmetric: averages best-match distance in both directions.
    Returns (similarity, mean_delta_e).
    """
    ca, wa = dominant_palette(img_a, n_colors)
    cb, wb = dominant_palette(img_b, n_colors)
    la, lb = _to_lab(ca), _to_lab(cb)
    d = np.linalg.norm(la[:, None, :] - lb[None, :, :], axis=2)
    fwd = (d.min(axis=1) * wa).sum()
    bwd = (d.min(axis=0) * wb).sum()
    mean_delta_e = float((fwd + bwd) / 2.0)
    return float(np.exp(-mean_delta_e / 25.0)), mean_delta_e


def colors_in_prompt(prompt):
    p = (prompt or "").lower()
    return [name for name in NAMED_COLORS if name in p]


def palette_prompt_adherence(img, prompt, n_colors=8, threshold=25.0):
    """Fraction of colours named in the design brief that appear in the image."""
    names = colors_in_prompt(prompt)
    if not names:
        return None, []
    colors, _ = dominant_palette(img, n_colors)
    lab_img = _to_lab(colors)
    hits = [nm for nm in names
            if np.linalg.norm(lab_img - _to_lab([NAMED_COLORS[nm]])[0],
                              axis=1).min() < threshold]
    return len(hits) / len(names), hits


def score_row(bank, rec):
    inp = Image.open(rec["input"])
    base = Image.open(rec["base"])
    lora = Image.open(rec["lora"])
    tgt = Image.open(rec["expected"]) if rec.get("expected") else None
    prompt = rec["prompt"]

    row = {
        "pair_id": rec["pair_id"], "room_type": rec.get("room_type", ""),
        "split": rec.get("split", ""),
        "base_clip_t": bank.clip_t(base, prompt),
        "lora_clip_t": bank.clip_t(lora, prompt),
        # --- no-reference image quality / aesthetics ---
        "base_aesthetic": bank.aesthetic(base),
        "lora_aesthetic": bank.aesthetic(lora),
        "base_clipiqa_quality": bank.clip_iqa(base, "quality"),
        "lora_clipiqa_quality": bank.clip_iqa(lora, "quality"),
        "base_clipiqa_realism": bank.clip_iqa(base, "realism"),
        "lora_clipiqa_realism": bank.clip_iqa(lora, "realism"),
        "base_ssim_input": bank.ssim(base, inp),   # structure kept vs input
        "lora_ssim_input": bank.ssim(lora, inp),
        "base_dino_input": bank.dino_sim(base, inp),  # identity kept
        "lora_dino_input": bank.dino_sim(lora, inp),
    }
    # --- palette adherence to the design brief ---
    try:
        b_adh, b_hits = palette_prompt_adherence(base, prompt)
        l_adh, l_hits = palette_prompt_adherence(lora, prompt)
        row["prompt_colors"] = ";".join(colors_in_prompt(prompt))
        row["base_palette_prompt"] = b_adh
        row["lora_palette_prompt"] = l_adh
        row["base_palette_hits"] = ";".join(b_hits)
        row["lora_palette_hits"] = ";".join(l_hits)
    except Exception as e:
        print(f"  (palette/prompt metric skipped: {type(e).__name__})")

    if tgt is not None:
        # --- palette closeness to the designer target ---
        try:
            b_sim, b_de = palette_similarity(base, tgt)
            l_sim, l_de = palette_similarity(lora, tgt)
            row["base_palette_target"] = b_sim
            row["lora_palette_target"] = l_sim
            row["base_palette_deltaE"] = b_de
            row["lora_palette_deltaE"] = l_de
            row["lora_closer_palette"] = l_sim > b_sim
        except Exception as e:
            print(f"  (palette/target metric skipped: {type(e).__name__})")

        row.update({
            "base_clip_i_target": bank.clip_i(base, tgt),
            "lora_clip_i_target": bank.clip_i(lora, tgt),
            "base_dino_target": bank.dino_sim(base, tgt),
            "lora_dino_target": bank.dino_sim(lora, tgt),
            "base_lpips_target": bank.lpips_dist(base, tgt),  # lower better
            "lora_lpips_target": bank.lpips_dist(lora, tgt),
        })
        # convenience: does LoRA win on closeness-to-HGTV?
        row["lora_closer_clip_i"] = row["lora_clip_i_target"] > row["base_clip_i_target"]
        row["lora_closer_dino"] = row["lora_dino_target"] > row["base_dino_target"]
        row["lora_closer_lpips"] = row["lora_lpips_target"] < row["base_lpips_target"]
    return row


# ----------------------------------------------------------------------
# LLM judges (pairwise, blind, randomized)
# ----------------------------------------------------------------------
JUDGE_RUBRIC = (
    "You are comparing two AI-renovated versions of the SAME room, produced "
    "from the same input photo and the same design brief. Pick which one is "
    "the better renovation, judging: (1) adherence to the design direction, "
    "(2) photorealism / professional quality, (3) preservation of the room's "
    "original layout and perspective. Answer with a strict JSON object only: "
    '{"winner": "A" or "B", "reason": "<one short sentence>"}.'
)


def _img_b64(path, max_side=1536):
    """Downscale before sending so the judge does not resize it itself."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Judge configuration matching the setup already used elsewhere in this project.
# Both judges run through Vertex AI, so no separate API keys are required.
CLAUDE_JUDGE_MODEL = os.environ.get("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-6")
GEMINI_JUDGE_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.1-pro-preview")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "adsp-s26-reccys")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "us-east5")
GEMINI_LOCATION = os.environ.get("GOOGLE_CLOUD_GEN_LOCATION", "global")


def judge_claude(prompt_text, img_a_path, img_b_path, model=None):
    """Claude judge via Vertex AI (no API key needed - uses project credentials)."""
    from anthropic import AnthropicVertex
    client = AnthropicVertex(project_id=VERTEX_PROJECT, region=VERTEX_REGION)
    msg = client.messages.create(
        model=model or CLAUDE_JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"Design brief:\n{prompt_text}\n\n{JUDGE_RUBRIC}"},
            {"type": "text", "text": "Option A:"},
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": _img_b64(img_a_path)}},
            {"type": "text", "text": "Option B:"},
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": _img_b64(img_b_path)}},
        ]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _parse_winner(text)


def judge_gemini(prompt_text, img_a_path, img_b_path, model=None):
    """Gemini judge via Vertex AI (no API key needed - uses project credentials)."""
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=VERTEX_PROJECT,
                          location=GEMINI_LOCATION)
    parts = [
        types.Part.from_text(text=f"Design brief:\n{prompt_text}\n\n{JUDGE_RUBRIC}"),
        types.Part.from_text(text="Option A:"),
        types.Part.from_bytes(data=base64.b64decode(_img_b64(img_a_path)),
                              mime_type="image/png"),
        types.Part.from_text(text="Option B:"),
        types.Part.from_bytes(data=base64.b64decode(_img_b64(img_b_path)),
                              mime_type="image/png"),
    ]
    resp = client.models.generate_content(
        model=model or GEMINI_JUDGE_MODEL,
        contents=[types.Content(role="user", parts=parts)],
    )
    return _parse_winner(resp.text)


def _parse_winner(text):
    try:
        s = text[text.index("{"):text.rindex("}") + 1]
        obj = json.loads(s)
        w = obj.get("winner", "").strip().upper()
        return (w if w in ("A", "B") else None), obj.get("reason", "")
    except Exception:
        return None, text.strip()[:120]


def run_judges(records, use_claude, use_gemini, seed=42):
    random.seed(seed)
    rows = []
    for rec in records:
        if not rec.get("base") or not rec.get("lora"):
            continue
        # randomize which physical image is shown as A vs B (blind)
        flip = random.random() < 0.5
        a_path, b_path = (rec["lora"], rec["base"]) if flip else (rec["base"], rec["lora"])
        a_is_lora = flip

        out = {"pair_id": rec["pair_id"], "room_type": rec.get("room_type", ""),
               "a_is_lora": a_is_lora}

        if use_claude:
            w, why = judge_claude(rec["prompt"], a_path, b_path)
            out["claude_pick"] = _winner_to_model(w, a_is_lora)
            out["claude_reason"] = why
        if use_gemini:
            w, why = judge_gemini(rec["prompt"], a_path, b_path)
            out["gemini_pick"] = _winner_to_model(w, a_is_lora)
            out["gemini_reason"] = why
        rows.append(out)
        print("judged", rec["pair_id"])
    return rows


def _winner_to_model(winner, a_is_lora):
    """Map the blind A/B verdict back to 'lora' or 'base'."""
    if winner is None:
        return None
    if winner == "A":
        return "lora" if a_is_lora else "base"
    return "base" if a_is_lora else "lora"


def cohen_kappa(labels_a, labels_b):
    """Simple 2-rater Cohen's kappa over {'lora','base'} labels."""
    pairs = [(x, y) for x, y in zip(labels_a, labels_b) if x and y]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    from collections import Counter
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def scan_validation_dir(root: Path):
    """
    Build an eval manifest from the folder layout produced by the existing
    qwen_lora_vs_base notebook, i.e. per-sample dirs containing:
        input.png / expected.png / generated.png (LoRA) /
        base_qwen_api_generated.png (base) / prompt.txt
    Samples missing the base output (e.g. the one that hit 402) are skipped.
    """
    records = []
    for gen in sorted(root.rglob("generated.png")):
        d = gen.parent
        base = d / "base_qwen_api_generated.png"
        inp = d / "input.png"
        if not (base.exists() and inp.exists()):
            print(f"  skip {d.name} (no base output yet)")
            continue
        prompt_file = d / "prompt.txt"
        prompt = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.exists() else ""
        name = d.name
        pair_id = name.split("_", 1)[1] if name.split("_", 1)[0].isdigit() else name
        records.append({
            "pair_id": pair_id,
            "room_type": pair_id.split("__")[0] if "__" in pair_id else "unknown",
            "split": "holdout",
            "prompt": prompt,
            "input": str(inp),
            "expected": str(d / "expected.png") if (d / "expected.png").exists() else "",
            "base": str(base),
            "lora": str(gen),
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="eval_manifest.json from the generation step")
    ap.add_argument("--validation-dir",
                    help="Alternative to --manifest: scan the existing notebook's "
                         "validation folder (input.png/expected.png/generated.png/"
                         "base_qwen_api_generated.png) to evaluate the already-made "
                         "holdout comparisons.")
    ap.add_argument("--output-dir", default="./eval_outputs")
    ap.add_argument("--no-judges", action="store_true", help="metrics only, skip LLM judges")
    ap.add_argument("--judges-only", action="store_true",
                    help="Run only the LLM judges, skip the embedding metrics. "
                         "Useful on a machine that has Vertex credentials but not "
                         "the metric packages (CLIP / LPIPS / scikit-image).")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--no-gemini", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest and not args.validation_dir:
        raise SystemExit("Pass either --manifest or --validation-dir.")

    if args.validation_dir:
        print(f"Scanning validation dir: {args.validation_dir}")
        records = scan_validation_dir(Path(args.validation_dir))
        print(f"Found {len(records)} samples with both base and LoRA outputs")
        (out_dir / "scanned_manifest.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        records = json.load(open(args.manifest))

    missing_target = [r["pair_id"] for r in records if not r.get("expected")]
    if missing_target:
        print(f"Note: {len(missing_target)} sample(s) have no HGTV target image; "
              f"target-based metrics will be skipped for those.")

    # ---- A) metric panel ----
    if args.judges_only:
        print("Skipping embedding metrics (--judges-only).")
        mdf = pd.DataFrame()
    else:
        print("=== Computing embedding / perceptual metrics ===")
        bank = MetricBank()
        metric_rows = [score_row(bank, r) for r in records]
        mdf = pd.DataFrame(metric_rows)
        mdf.to_csv(out_dir / "metrics.csv", index=False)

    # per-room-type summary of "LoRA closer to HGTV" win rates
    if not mdf.empty and "lora_closer_clip_i" in mdf.columns:
        summary = mdf.groupby("room_type")[
            [c for c in ("lora_closer_clip_i", "lora_closer_dino", "lora_closer_lpips")
             if c in mdf.columns]
        ].mean()
        summary.to_csv(out_dir / "winrate_by_room_type.csv")
        print("\nLoRA-closer-to-HGTV win rate by room type:")
        print(summary)

    # ---- B) LLM judges ----
    if not args.no_judges:
        print(f"Judges: Claude={CLAUDE_JUDGE_MODEL} (Vertex {VERTEX_REGION}), "
              f"Gemini={GEMINI_JUDGE_MODEL} (Vertex {GEMINI_LOCATION})")
        print("\n=== Running LLM judges (blind, randomized, pairwise) ===")
        jrows = run_judges(records, use_claude=not args.no_claude,
                           use_gemini=not args.no_gemini)
        jdf = pd.DataFrame(jrows)
        jdf.to_csv(out_dir / "judges.csv", index=False)

        # win rates + agreement
        for col in ("claude_pick", "gemini_pick"):
            if col in jdf.columns:
                rate = (jdf[col] == "lora").mean()
                print(f"{col}: LoRA preferred in {rate:.0%} of items")
        if "claude_pick" in jdf.columns and "gemini_pick" in jdf.columns:
            k = cohen_kappa(list(jdf["claude_pick"]), list(jdf["gemini_pick"]))
            agree = (jdf["claude_pick"] == jdf["gemini_pick"]).mean()
            print(f"Inter-judge agreement: {agree:.0%}  (Cohen's kappa = {k})")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
