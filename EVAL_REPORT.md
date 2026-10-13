# Evaluation report

policy_version `2026-10-13.4` · generated 2026-10-13 13:13 · 185 images (149 dev / 36 test)

All numbers come from `eval.py` on our labeled set. Rates use 95% Wilson intervals; cost uses a 1000x percentile bootstrap. Cost matrix (true→pred APPROVE/REVIEW/REJECT): {'safe': [0, 3, 10], 'review': [5, 0, 2], 'reject': [100, 3, 0]}.

## Signals (dev, explicit vs safe)

| signal | ROC AUC [95% CI] | recall at 1% false positives |
|---|---|---|
| freepik.high | 0.975 [0.939–0.997] | 88.6% |
| nudenet.ANUS_EXPOSED | 0.690 [0.606–0.770] | 12.9% |
| nudenet.BUTTOCKS_EXPOSED | 0.658 [0.548–0.748] | 2.9% |
| nudenet.FEMALE_BREAST_EXPOSED | 0.929 [0.882–0.966] | 77.1% |
| nudenet.FEMALE_GENITALIA_EXPOSED | 0.899 [0.838–0.944] | 71.4% |
| nudenet.MALE_GENITALIA_EXPOSED | 0.761 [0.679–0.833] | 28.6% |
| nudenet.max_exposed | 0.958 [0.917–0.983] | 74.3% |

## Decisions

### Current config thresholds (dev)

n = 149  (safe: 65, review: 14, reject: 70)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 57 | 8 | 0 |
| review | 4 | 7 | 3 |
| reject | 1 | 3 | 66 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 1.4% [0.3–7.7] |
| explicit_caught_auto | 94.3% [86.2–97.8] |
| safe_auto_rejected | 0.0% [0.0–5.6] |
| safe_sent_to_review | 12.3% [6.4–22.5] |
| borderline_auto_approved | 28.6% [11.7–54.6] |
| review_rate_this_mix | 12.1% [7.8–18.3] |
| reject_precision | 95.7% [88.0–98.5] |
| mean cost / image | 1.067 [0.262–2.557] (bootstrap) |

> **Too small:** 70 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

### Re-tuned thresholds, 5-fold cross-validated (dev)

n = 149  (safe: 65, review: 14, reject: 70)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 57 | 8 | 0 |
| review | 4 | 7 | 3 |
| reject | 1 | 4 | 65 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 1.4% [0.3–7.7] |
| explicit_caught_auto | 92.9% [84.3–96.9] |
| safe_auto_rejected | 0.0% [0.0–5.6] |
| safe_sent_to_review | 12.3% [6.4–22.5] |
| borderline_auto_approved | 28.6% [11.7–54.6] |
| review_rate_this_mix | 12.8% [8.3–19.1] |
| reject_precision | 95.6% [87.8–98.5] |
| mean cost / image | 1.087 [0.282–2.591] (bootstrap) |

> **Too small:** 70 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.

Suggested thresholds (tuned on all dev) are in `runs/tuned_rules.yaml`. Adopt them only if the CV row beats the current row; copy into config.yaml and bump `policy_version`.

## Final test

### HELD-OUT TEST (frozen config thresholds)

n = 36  (safe: 17, review: 9, reject: 10)

| true \ pred | APPROVE | REVIEW | REJECT |
|---|---|---|---|
| safe | 15 | 2 | 0 |
| review | 2 | 6 | 1 |
| reject | 1 | 0 | 9 |

| metric | value [95% CI] |
|---|---|
| explicit_auto_approved | 10.0% [1.8–40.4] |
| explicit_caught_auto | 90.0% [59.6–98.2] |
| safe_auto_rejected | 0.0% [0.0–18.4] |
| safe_sent_to_review | 11.8% [3.3–34.3] |
| borderline_auto_approved | 22.2% [6.3–54.7] |
| review_rate_this_mix | 22.2% [11.7–38.1] |
| reject_precision | 90.0% [59.6–98.2] |
| mean cost / image | 3.278 [0.194–8.945] (bootstrap) |

> **Too small:** 10 explicit images. Showing ≤1% explicit auto-approval at 95% confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.


## Misclassified files

Severe = explicit approved or safe rejected.

| severity | file | true | pred | top signals |
|---|---|---|---|---|
| SEVERE | ../labeled/reject/images (48)213.jpeg | reject | APPROVE | freepik.high=0.00, nudenet.BUTTOCKS_EXPOSED=0.00, nudenet.FEMALE_BREAST_EXPOSED=0.00 |
| minor | ../labeled/review/images (37).jpeg | review | APPROVE | freepik.high=0.00, nudenet.BUTTOCKS_EXPOSED=0.00, nudenet.MALE_GENITALIA_EXPOSED=0.00 |
| minor | ../labeled/review/images (5).jpeg | review | APPROVE | freepik.high=0.08, nudenet.FEMALE_BREAST_EXPOSED=0.01, nudenet.FEMALE_GENITALIA_EXPOSED=0.00 |
| minor | ../labeled/review/images (7).jpeg | review | REJECT | nudenet.FEMALE_BREAST_EXPOSED=0.77, nudenet.BUTTOCKS_EXPOSED=0.34, freepik.high=0.27 |
| minor | ../labeled/safe/images (20)213.jpeg | safe | REVIEW | nudenet.FEMALE_BREAST_EXPOSED=0.63, freepik.high=0.49, nudenet.BUTTOCKS_EXPOSED=0.00 |
| minor | ../labeled/safe/images (37).jpeg | safe | REVIEW | nudenet.FEMALE_BREAST_EXPOSED=0.59, freepik.high=0.06, nudenet.MALE_GENITALIA_EXPOSED=0.00 |

## Robustness (dev reject+review images, production pipeline)

n = 84

| transform | APPROVE | REVIEW | REJECT | explicit escaped to APPROVE |
|---|---|---|---|---|
| original | 5 | 10 | 69 | 0 |
| crop_center_70 | 7 | 10 | 67 | 0 |
| rotate_90 | 6 | 11 | 67 | 1 |
| rotate_15 | 7 | 9 | 68 | 1 |
| mirror | 6 | 9 | 69 | 0 |
| grayscale | 7 | 13 | 64 | 1 |
| blur_r4 | 8 | 23 | 53 | 0 |
| jpeg_q10 | 10 | 25 | 49 | 2 |
| pad_border_40 | 9 | 22 | 53 | 2 |
| stickers_on_regions | 14 | 58 | 12 | 6 |
| meme_text | 6 | 11 | 67 | 0 |
| screenshot_of_screen | 10 | 27 | 47 | 3 |

