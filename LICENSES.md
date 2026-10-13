# License matrix

This is an engineering risk inventory, **not legal advice**. Licenses were checked on 2026-10-12 from the sources listed. Before going to production, ask counsel to confirm the AGPL item and the model-weight terms in particular.

## Models

| Component | Used as | License (source) | Commercial-use risk | Notes |
|---|---|---|---|---|
| Freepik/nsfw_image_detector (EVA02-base, INT8) | stage-2 classifier (enabled) | MIT (HF model card + `config.json` `pretrained_cfg.license`) | Low | Base model timm `eva02_base_patch14_448` is MIT. Trained on "100k synthetically labeled images"; training-data provenance isn't documented. |
| Falconsai/nsfw_image_detection (ViT-base-224, INT8) | stage-1 cascade gate (enabled) | Apache-2.0 (HF card) | Low | Keep the NOTICE/attribution. |
| AdamCodd/vit-base-nsfw-detector | evaluated, not used | Apache-2.0 (HF card) | Low | |
| Marqo/nsfw-image-detection-384 | evaluated, not used | Apache-2.0 (HF card) | Low | |
| **NudeNet v3.4 weights (320n)** | region detector (enabled, **optional**) | **AGPL-3.0** (github.com/notAI-tech/NudeNet, `LICENSE`) | **High: flag for counsel** | AGPL §13 can require offering source to users who interact with the software over a network. Open questions for counsel: whether calls from our own backend count as that interaction, and whether the weights are a "work based on" the program. We don't import NudeNet's code. We load only its ONNX weights (our own adapter is in `scorers.py`). |
| SafeVision `best.onnx` | **not used** | **AGPL-3.0** declared in the ONNX file's Ultralytics metadata (SafeVision `Models/README.md`). The repository code itself is Apache-2.0. | High | Excluded for that reason. |
| onnx-community/age-gender-prediction-ONNX | **not used** (candidate for the age hook) | Apache-2.0 (HF card, per SafeVision registry) | Low license risk, **high product/privacy risk** | Estimating age from faces may be biometric processing under some laws. Ask counsel before enabling it. |

### Removing the AGPL component

Set `models.nudenet.enabled: false` in `config.yaml`, delete `models/nudenet-*.onnx`, and bump `policy_version`. Rules that reference `nudenet.*` switch off automatically, and the classifier rule (`EXPLICIT_CONTENT`, Freepik) keeps working. Also remove the `nudenet.*` entries from `cascade.clear_if_below`, or Falconsai alone will decide which images skip Freepik. **What you lose:** `regions` boxes, the body-part reason codes, and some recall on explicit images that the classifiers score low. **Re-run `eval.py report` before shipping that change**, because the eval measures how much recall drops. I know of no permissively licensed NSFW region detector of comparable quality as of 2026-10. Most YOLO-based ones inherit AGPL from Ultralytics.

## Libraries (runtime image)

| Library | License | Risk | Notes |
|---|---|---|---|
| onnxruntime | MIT | Low | |
| numpy | BSD-3-Clause | Low | |
| Pillow | MIT-CMU (HPND) | Low | |
| opencv-python-headless | Apache-2.0 | Low–Medium | Wheels bundle third-party native libraries, including LGPL ones (e.g. FFmpeg). Dynamic linking; keep the notices. Used only for one resize. It could be swapped for a numpy resize if needed. |
| pi-heif | BSD-3-Clause | Low–Medium | Bundles **libheif and libde265 (LGPL-3.0)**, dynamically linked. Decode-only: no x265 (GPL) encoder, unlike `pillow-heif`. Keep the notices. |
| fastapi, starlette, pydantic, uvicorn | MIT / BSD-3-Clause | Low | |
| python-multipart | Apache-2.0 | Low | |
| PyYAML | MIT | Low | |
| PyJWT | MIT | Low | |
| Caddy (container) | Apache-2.0 | Low | |
| Podman (runtime on Mac + server) | Apache-2.0 | Low | Not shipped in our image. |
| podman-compose (deploy tool) | GPL-2.0 | Low | A CLI we run, not linked into or shipped with the service. |
| Python | PSF | Low | |

## Dev/eval-only (not shipped)

PyTorch (BSD-3), timm (Apache-2.0), transformers (Apache-2.0), onnx (Apache-2.0), httpx (BSD-3), locust (MIT), pytest (MIT). All low risk.

## Data

There is no data in this repo. The labeled evaluation images you supply have their own rights questions. Make sure you're allowed to use each one, and that **no image could depict a minor**. Never commit images, since `.gitignore` blocks common image formats.
