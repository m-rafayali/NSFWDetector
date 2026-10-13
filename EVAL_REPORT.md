# Evaluation report

policy_version `2026-10-13.3` · generated 2026-10-13 10:46 · 185 images (149 dev / 36 test)

All numbers come from `eval.py` on our labeled set. Rates use 95% Wilson intervals; cost uses a 1000x percentile bootstrap. Cost matrix (true→pred A/RV/RJ): {'safe': [0, 1, 10], 'review': [5, 0, 2], 'reject': [100, 3, 0]}.

## Per-model bake-off (dev split, reject vs safe)

| signal | ROC AUC [95% CI] | recall@1% FPR | p50 ms | p95 ms |
|---|---|---|---|---|
| adamcodd.nsfw | 0.801 [0.710–0.873] | 31.4% | 109 | 140 |
| falconsai-int8.nsfw | 0.920 [0.868–0.958] | 70.0% | 31 | 35 |
| falconsai.nsfw | 0.926 [0.877–0.964] | 70.0% | 36 | 42 |
| freepik-int8.high | 0.975 [0.939–0.997] | 88.6% | 299 | 345 |
| freepik-int8.medium+ | 0.965 [0.927–0.992] | 80.0% | 299 | 345 |
| freepik-int8.nsfw | 0.929 [0.871–0.971] | 0.0% | 299 | 345 |
| freepik.high | 0.975 [0.939–0.997] | 88.6% | 305 | 337 |
| freepik.medium+ | 0.960 [0.917–0.993] | 75.7% | 305 | 337 |
| freepik.nsfw | 0.929 [0.869–0.971] | 0.0% | 305 | 337 |
| marqo.nsfw | 0.856 [0.777–0.916] | 15.7% | 20 | 23 |
| nudenet-640m.max_exposed | 0.957 [0.909–0.985] | 70.0% | 96 | 110 |
| nudenet-bgr.max_exposed | 0.948 [0.901–0.979] | 68.6% | 9 | 11 |
| nudenet.max_exposed | 0.953 [0.905–0.982] | 74.3% | 9 | 12 |

Latency = per-model time on this machine incl. preprocessing, single process.

## Ensemble decisions

### Current config thresholds (dev, untuned)

n = 149  (safe: 65, review: 14, reject: 70)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 51 | 14 | 0 |
| review | 3 | 8 | 3 |
| reject | 1 | 3 | 66 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 1.4% [0.3–7.7] |
| explicit_caught_auto | 94.3% [86.2–97.8] |
| safe_auto_rejected | 0.0% [0.0–5.6] |
| safe_sent_to_review | 21.5% [13.3–33.0] |
| borderline_auto_approved | 21.4% [7.6–47.6] |
| review_rate_this_mix | 16.8% [11.6–23.6] |
| reject_precision | 95.7% [88.0–98.5] |
| mean cost / image | 0.966 [0.201–2.437] (bootstrap) |

> **Too small:** 70 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

### Tuned rules, 5-fold cross-validated (dev)

n = 149  (safe: 65, review: 14, reject: 70)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 51 | 14 | 0 |
| review | 5 | 6 | 3 |
| reject | 1 | 4 | 65 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 1.4% [0.3–7.7] |
| explicit_caught_auto | 92.9% [84.3–96.9] |
| safe_auto_rejected | 0.0% [0.0–5.6] |
| safe_sent_to_review | 21.5% [13.3–33.0] |
| borderline_auto_approved | 35.7% [16.3–61.2] |
| review_rate_this_mix | 16.1% [11.1–22.8] |
| reject_precision | 95.6% [87.8–98.5] |
| mean cost / image | 1.054 [0.262–2.537] (bootstrap) |

> **Too small:** 70 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

### Logistic-regression stacker, 5-fold CV (dev)

n = 149  (safe: 65, review: 14, reject: 70)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 0 | 64 | 1 |
| review | 0 | 12 | 2 |
| reject | 0 | 8 | 62 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 0.0% [0.0–5.2] |
| explicit_caught_auto | 88.6% [79.0–94.1] |
| safe_auto_rejected | 1.5% [0.3–8.2] |
| safe_sent_to_review | 98.5% [91.8–99.7] |
| borderline_auto_approved | 0.0% [0.0–21.5] |
| review_rate_this_mix | 56.4% [48.4–64.1] |
| reject_precision | 95.4% [87.3–98.4] |
| mean cost / image | 0.685 [0.544–0.872] (bootstrap) |

> **Too small:** 70 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

Stacker **beats** tuned rules on CV cost (0.685 vs 1.054). Only adopt it if it wins clearly (outside bootstrap noise).

Suggested thresholds (tuned on all dev) written to `runs/tuned_rules.yaml`. Copy into config.yaml and bump `policy_version` by hand after review.

## Final test

### HELD-OUT TEST (frozen config thresholds, touched once)

n = 36  (safe: 17, review: 9, reject: 10)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 14 | 3 | 0 |
| review | 2 | 6 | 1 |
| reject | 1 | 0 | 9 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 10.0% [1.8–40.4] |
| explicit_caught_auto | 90.0% [59.6–98.2] |
| safe_auto_rejected | 0.0% [0.0–18.4] |
| safe_sent_to_review | 17.6% [6.2–41.0] |
| borderline_auto_approved | 22.2% [6.3–54.7] |
| review_rate_this_mix | 25.0% [13.8–41.1] |
| reject_precision | 90.0% [59.6–98.2] |
| mean cost / image | 3.194 [0.139–8.861] (bootstrap) |

> **Too small:** 10 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

## Edge cases

| class/edge | n | auto-approved | auto-rejected | review |
|---|---|---|---|---|

## Calibration (dev): predicted vs observed explicit rate

| signal | bins (mean score → observed explicit rate, n) |
|---|---|
| freepik.high | 0.01→0.06 (n=66); 0.20→0.67 (n=6); 0.70→1.00 (n=1); 0.76→1.00 (n=2); 0.99→0.98 (n=60) |
| falconsai.nsfw | 0.01→0.10 (n=40); 0.18→0.20 (n=10); 0.40→0.33 (n=3); 0.69→0.00 (n=1); 0.77→0.17 (n=6); 1.00→0.83 (n=75) |
| nudenet.max_exposed | 0.03→0.03 (n=36); 0.19→0.12 (n=16); 0.42→0.38 (n=13); 0.60→0.58 (n=12); 0.82→0.95 (n=58) |

## Misclassified files

Severe = explicit approved or safe rejected.

| severity | file | true | pred | top signals |
|---|---|---|---|---|
| SEVERE | ../labeled/reject/images (48)213.jpeg | reject | APPROVE | freepik.nsfw=1.00, freepik.low+=1.00, falconsai.nsfw=1.00 |
| minor | ../labeled/review/images (37).jpeg | review | APPROVE | falconsai.normal=1.00, freepik.neutral=0.96, marqo.sfw=0.95 |
| minor | ../labeled/review/images (5).jpeg | review | APPROVE | falconsai.normal=0.98, marqo.sfw=0.93, adamcodd.nsfw=0.88 |
| minor | ../labeled/review/images (7).jpeg | review | REJECT | freepik.nsfw=1.00, freepik.low+=1.00, freepik.medium+=1.00 |
| minor | ../labeled/safe/images (20)213.jpeg | safe | REVIEW | freepik.nsfw=1.00, freepik.low+=1.00, freepik.medium+=1.00 |
| minor | ../labeled/safe/images (21).jpeg | safe | REVIEW | falconsai.normal=1.00, adamcodd.sfw=0.99, freepik.neutral=0.87 |
| minor | ../labeled/safe/images (37).jpeg | safe | REVIEW | falconsai.normal=1.00, adamcodd.sfw=0.92, freepik.nsfw=0.91 |

## Robustness (dev reject+review images, production pipeline)

n = 84

| transform | APPROVE | REVIEW | REJECT | explicit escaped to APPROVE |
|---|---|---|---|---|
| original | 4 | 11 | 69 | 0 |
| crop_center_70 | 5 | 12 | 67 | 0 |
| rotate_90 | 5 | 12 | 67 | 1 |
| rotate_15 | 6 | 10 | 68 | 1 |
| mirror | 6 | 9 | 69 | 0 |
| grayscale | 6 | 14 | 64 | 1 |
| blur_r4 | 6 | 25 | 53 | 0 |
| jpeg_q10 | 9 | 26 | 49 | 1 |
| pad_border_40 | 8 | 23 | 53 | 2 |
| stickers_on_regions | 13 | 59 | 12 | 6 |
| meme_text | 5 | 12 | 67 | 0 |
| screenshot_of_screen | 9 | 28 | 47 | 2 |

