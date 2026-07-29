# agentic-room-design-recsys

Agentic room design recommender: retrieval-guided style discovery with generative visualization. Capstone project, 2026.

Given a photo of an existing room, the system identifies a target design style and produces a redesigned visualization of that room. The generation model is a LoRA-fine-tuned image editor trained on before/after renovation pairs; a style-labelled index supports retrieval of design references.

## Dataset

The dataset is 106 before/after room pairs scraped from published renovation features (HGTV Property Brothers and similar sources). Each pair is the original room ("before") and the professionally staged redesign ("after"), labelled with room type and design style.

| Split | Pairs | Use |
|---|---|---|
| train | 96 | LoRA training (augmented) |
| holdout | 10 | generalization check (not augmented) |
| **total** | **106** | |

Room types: living room (42), bedroom (26), kitchen (12), bathroom (10), dining room (9), entryway (5), game room (2).
Primary styles: Contemporary (45), Traditional (41), Farmhouse (7), Mid-Century Modern (4), Bohemian (3), Glam (3).

**On evaluation scope.** The minimum evaluation set for the base-vs-LoRA comparison was 10 pairs, chosen with A100 compute cost in mind. We evaluated 31 pairs instead, to select demonstration examples from a larger pool. The 5 demonstration pairs shown in the report are a subset of those 31.

### Data links (Google Drive)

Images are hosted on Drive rather than in the repository, to keep the repo lightweight. The label manifests are versioned here under `data/`.

- Full dataset, 106 before/after pairs: https://drive.google.com/drive/folders/1t1Oe6MwzNRrxxDAaYUJMOi3hS9tNn0ZQ
- Training set, 96 pairs (augmented): https://drive.google.com/drive/folders/1BLyWKW4GkoX47oA9AP0MiZNEc9e9Lx8S
- Holdout set, 10 pairs: https://drive.google.com/drive/folders/1iL_flBHCpt-anCO0Grhpyi2b-HiSuo4E

## Repository structure

```
data/
  pairs_with_style.csv        full labels: room type, split, style, style evidence, source URLs
  train_manifest.csv          training split manifest (96 pairs)
  holdout_manifest.csv        holdout split manifest (10 pairs)
  dataset_augmentation.ipynb  paired before/after augmentation pipeline
  gallery.html                visual sample: 15 pairs with style labels (download to view)
README.md
LICENSE
```

## Method

**Augmentation** (`data/dataset_augmentation.ipynb`). Each training pair is expanded into the original plus 5 variants. Two design choices matter:
- Geometric transforms (crop-zoom, flip, small rotation) use one random state per (pair, variant) and are applied identically to the before and after image. If they diverged, the model would learn "renovation = mirror the room."
- Photometric degradation (jitter, grain, JPEG recompression) is applied to the before side only, matching the production case where a user uploads a noisy phone photo and expects a clean result. The holdout set is left un-augmented.

**Generation.** A LoRA adapter fine-tunes an image-editing model to map a room photo toward the target redesign style.

**Evaluation.** Base and LoRA outputs are compared against the designer reference on 31 pairs, using embedding-similarity metrics (DINOv2, CLIP-I, LPIPS, palette distance) and two LLM judges (Claude and Gemini) in a blind, position-randomized A/B comparison.

## Results and limitations

The LoRA improves finish quality and photorealism, and this is visible to both the judges and the eye. Two limitations are stated honestly, because they shape what the metrics can and cannot claim:

1. **Reference-similarity metrics are weak for this task.** Base and LoRA outputs are compositionally near-identical, so embedding distance to the reference cannot separate them reliably. Across 31 pairs, no single metric is significant on both the train and holdout slices at once. The top of the DINOv2 ranking is dominated by pairs that broke the room type — because drifting toward a generic magazine interior *reduces* distance to the reference, which is exactly what a room-type failure looks like.

2. **Room type is only partially preserved.** The prompt clause that enumerates furniture pulls some room types (notably bathrooms and dining rooms) toward living rooms. Room-type preservation is therefore tracked separately from the aesthetic comparison.

The two evidence families (embedding metrics and LLM judges) agree less often than a single headline number suggests; they are reported side by side rather than merged.
