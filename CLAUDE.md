# CLAUDE.md

CTC (dating app) image-moderation service. Phase 1 = images. Phase 2 = video (later, reuse `Scorer`s as frame scorers).

## Stack
Python 3.12 (pinned), uv, FastAPI, onnxruntime (CPU only), Pillow + pi-heif, opencv-headless (NudeNet resize only), SQLite (decisions and review queue; never images), Podman (podman-compose) + Caddy on one EC2 instance. No Docker anywhere (owner's choice).

## Hard rules
- No paid or external detection APIs, ever. Pretrained open models only, run locally via ONNX.
- License discipline: prefer MIT/Apache/BSD. NudeNet weights are AGPL. Keep it an optional layer (`enabled: false` must keep working). Record new components in LICENSES.md.
- Never commit images, model weights, keys, `.env`, `data/`, `labeled/`, `runs/`. Never reuse `../assests/` samples in tests: tests use synthetic images and the `Fake` scorer.
- Accuracy numbers come only from `eval.py` with CIs. Never quote model-card numbers or claim 100%.
- Any change to `config.yaml` thresholds or rules → bump `policy_version`.
- Keep the codebase small: few files, no new packages/modules unless clearly needed (owner's preference).

## Commands
```bash
uv sync                                         # deps (dev group default)
uv run pytest -q                                # tests (no weights needed)
uv run --group export python export.py          # models/*.onnx
uv run uvicorn ctc_mod.api:app --port 8000      # one process only
uv run python -m ctc_mod.api newkey <id> [admin]
uv run python eval.py score|report [--final]|robust|gate
CTC_API_KEY=… uv run locust -f loadtest.py --headless -u 2 -r 2 -t 1m --host http://localhost:8000
```

## Conventions
- Score keys are `"<model>.<signal>"`. Rules reference them. Reason codes in rules are an API contract (INTEGRATION.md table).
- Classifier signals: `p(label)`, `nsfw` = 1 − p(safe label), and for ordinal labels a cumulative `<label>+`.
- `nudenet` uses `channels: rgb` (bake-off winner); eval variant `nudenet-bgr` reproduces `nudenet==3.4.2` exactly.
- eval.py `VARIANTS` name exact files; `alias()` maps config models to the variant that scored them.
- API errors: `{"error": {"code", "message"}}` via `ApiError` / `ImageError`.
- `eval.py report` overwrites EVAL_REPORT.md (it appends runs/robustness.md if present). Run `robust` before `report`.

## PROGRESS (update at each checkpoint)
- [x] (a) eval harness + model export/parity + latency bake-off (accuracy bake-off **blocked on labeled data**)
- [x] (b) cascade + fusion rules + policy config (thresholds PROVISIONAL)
- [x] (c) FastAPI: auth, rate limit, backpressure, shadow, kill switch, review queue API, metrics, tests
- [x] (d) load test on M4. t3.small numbers are estimates (not measured on AWS)
- [x] (e) Containerfile/compose.yaml/Caddy/deploy.sh/CI/docs (container stack verified locally 2026-10-13)
- [x] Bake-off on 185 labeled images (2026-10-13): NudeNet-rgb + Freepik(INT8) + Falconsai gate; thresholds CV-tuned; baseline set
- [ ] Owner: check 2 suspect labels (README), grow labeled set toward ~1.5k/2.5k/750, re-run eval
- [ ] Owner: first deploy to t3.small in shadow mode; run loadtest.py there to replace the estimates
