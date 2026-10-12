# Evaluation report

**Status (2026-10-12): accuracy has not been measured. No labeled data yet.**

`eval.py report` regenerates this file, with confusion matrices, Wilson and bootstrap 95% confidence intervals, cross-validated tuned thresholds, the stacker comparison, edge-case tables, calibration, and every misclassified file. Until you supply `labeled/{safe,review,reject}/`, **no accuracy claim can be made** about this system, including the provisional thresholds in `config.yaml`. Accuracy figures on vendor model cards (e.g. Freepik's) are not ours and are not repeated here.

## What has been verified

| Check | Result |
|---|---|
| ONNX export matches PyTorch (random input) | Freepik FP32: max logit difference 0.019. Falconsai and Marqo FP32: 0.000. INT8 variants: up to 0.25 logit (Freepik). INT8 accuracy has to be measured before use. |
| Our NudeNet adapter matches `nudenet==3.4.2` (18 sample images) | max score difference < 1e-6, with BGR input |
| Finding | `nudenet==3.4.2` feeds the model **BGR**: its `cvtColor(RGBA2BGR)` and `swapRB=True` cancel out. The YOLO model was probably trained on RGB. Both settings are in the bake-off (`nudenet` vs `nudenet-rgb`). The default stays BGR to match your hand-checked prototype until the data decides. |
| Pipeline on 18 unlabeled samples | 13 of 18 settled by stage 1 (45–90 ms). 5 went to stage 2 (360–470 ms). Classifiers and NudeNet disagree on some images (e.g. NudeNet scores breast exposure ≥0.5 where Falconsai's NSFW score is ≈0.01). **These are anecdotes, not metrics.** |

## Per-model latency (Apple M4, CPU, onnxruntime 1.31, 1024×768 input, incl. preprocessing)

| model | params / input | 1 thread p50 | 2 threads p50 | INT8, 2 threads |
|---|---|---|---|---|
| NudeNet 320n | YOLOv8n, 320 | 13 ms | 9 ms | n/a |
| NudeNet 640m | YOLOv8m, 640 | 207 ms | 122 ms | n/a |
| Marqo | ViT-tiny, 384 | 33 ms | 23 ms | 25 ms |
| Falconsai | ViT-base, 224 | 57 ms | 42 ms | 39 ms |
| AdamCodd | ViT-base, 384 | 181 ms | 142 ms | 138 ms |
| Freepik | EVA02-base, 448 | 491 ms | 375 ms | 421 ms |

INT8 dynamic quantization was **not faster on ARM (M4)**. It cuts RAM (whole service: about 0.55 GB vs 0.95 GB) and may be faster on x86 CPUs with VNNI. Measure it on the target instance.

## How much labeled data you need

The 80/20 dev/test split is deterministic by file hash. Numbers are **totals per class**:

| class | minimum to start | for trustworthy targets | why |
|---|---|---|---|
| reject (explicit) | 500 | **≈1,500** | ≈300 in test. With zero misses, the 95% upper bound on "explicit auto-approved" is ≈1% (rule of three). Fewer images can't support the ≤1% goal. |
| safe | 1,000 | **≈2,500** | ≈500 in test, enough to measure a 1–2% false-reject rate to about ±1%. Should mirror real uploads: selfies, swimwear, gym, shirtless men, group photos. |
| review (borderline) | 300 | **≈750** | Lingerie, underboob/sideboob, sheer clothing, suggestive poses, art and statues, breastfeeding, medical. |
| each edge-case subfolder | 30 | **≥100** | At n=50 a rate's CI is about ±10–14 points. The report flags n<50 groups. |

Edge-case subfolders to create (e.g. `labeled/safe/swimwear/`): swimwear, shirtless, gym, lingerie, sheer_wet, underboob_sideboob, breastfeeding, medical, art_statues, anime_drawn, ai_generated, sex_toys, clothed_sexual_acts, censored_pixelated, screenshots, memes_text, low_light_blurry, very_large.

**Never include any image that could depict a minor.**

## Run order

```bash
uv run python eval.py score            # ~1.8 s/image on M4 for all 9 variants; resumable
uv run python eval.py report           # bake-off + CV-tuned thresholds (dev only)
# edit config.yaml (enable the chosen models, paste runs/tuned_rules.yaml), bump policy_version
uv run python eval.py robust           # evasion suite with the production pipeline
uv run python eval.py report --final   # touch the held-out test split ONCE; writes eval_metrics.json
cp eval_metrics.json eval_baseline.json && git add eval_*.json   # CI gate baseline
```
