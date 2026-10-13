# Evaluation report

policy_version `2026-10-13.3` · generated 2026-10-13 12:45 · 185 images (149 dev / 36 test)

All numbers come from `eval.py` on our labeled set. Rates use 95% Wilson intervals; cost uses a 1000x percentile bootstrap. Cost matrix (true→pred APPROVE/REVIEW/REJECT): {'safe': [0, 1, 10], 'review': [5, 0, 2], 'reject': [100, 3, 0]}.

## Signals (dev, explicit vs safe)

| signal | ROC AUC [95% CI] | recall at 1% false positives |
|---|---|---|
| freepik.high | 0.975 [0.939–0.997] | 88.6% |
| nudenet.ANUS_EXPOSED | 0.844 [0.762–0.902] | 11.4% |
| nudenet.BUTTOCKS_EXPOSED | 0.628 [0.521–0.725] | 2.9% |
| nudenet.FEMALE_BREAST_EXPOSED | 0.899 [0.828–0.958] | 77.1% |
| nudenet.FEMALE_GENITALIA_EXPOSED | 0.896 [0.831–0.943] | 68.6% |
| nudenet.MALE_GENITALIA_EXPOSED | 0.796 [0.714–0.866] | 30.0% |
| nudenet.max_exposed | 0.953 [0.905–0.982] | 74.3% |

## Decisions

### Current config thresholds (dev)

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

### Re-tuned thresholds, 5-fold cross-validated (dev)

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

Suggested thresholds (tuned on all dev) are in `runs/tuned_rules.yaml`. Adopt them only if the CV row beats the current row; copy into config.yaml and bump `policy_version`.

## Final test

### HELD-OUT TEST (frozen config thresholds)

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


## Misclassified files

Severe = explicit approved or safe rejected.

| severity | file | true | pred | top signals |
|---|---|---|---|---|
| SEVERE | ../labeled/reject/images (48)213.jpeg | reject | APPROVE | nudenet.FEMALE_BREAST_EXPOSED=0.02, freepik.high=0.00, nudenet.FEMALE_GENITALIA_EXPOSED=0.00 |
| minor | ../labeled/review/images (37).jpeg | review | APPROVE | nudenet.FEMALE_BREAST_EXPOSED=0.00, freepik.high=0.00, nudenet.FEMALE_GENITALIA_EXPOSED=0.00 |
| minor | ../labeled/review/images (5).jpeg | review | APPROVE | freepik.high=0.08, nudenet.FEMALE_BREAST_EXPOSED=0.05, nudenet.FEMALE_GENITALIA_EXPOSED=0.00 |
| minor | ../labeled/review/images (7).jpeg | review | REJECT | nudenet.FEMALE_BREAST_EXPOSED=0.77, nudenet.BUTTOCKS_EXPOSED=0.34, freepik.high=0.27 |
| minor | ../labeled/safe/images (20)213.jpeg | safe | REVIEW | nudenet.FEMALE_BREAST_EXPOSED=0.63, freepik.high=0.49, nudenet.FEMALE_GENITALIA_EXPOSED=0.02 |
| minor | ../labeled/safe/images (21).jpeg | safe | REVIEW | nudenet.FEMALE_BREAST_EXPOSED=0.42, nudenet.MALE_GENITALIA_EXPOSED=0.04, freepik.high=0.01 |
| minor | ../labeled/safe/images (37).jpeg | safe | REVIEW | nudenet.FEMALE_BREAST_EXPOSED=0.59, freepik.high=0.06, nudenet.FEMALE_GENITALIA_EXPOSED=0.02 |

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

