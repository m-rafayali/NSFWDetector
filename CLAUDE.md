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
uv run --group export python export.py          # models/*.onnx for the enabled models (reproducible hashes)
uv run uvicorn ctc_mod.api:app --port 8000      # one process only; reads .env
uv run python -m ctc_mod.api token <sub> [admin] [hours]   # dev JWT
uv run python eval.py score|robust|report [--final]|gate
podman-compose up -d --build --force-recreate   # --force-recreate or the old container keeps running
```

## Conventions
- Score keys are `"<model>.<signal>"`. Rules reference them. Reason codes are an API contract (INTEGRATION.md).
- Responses only expose signals the policy reads (`Moderator.used`); eval scores every stage via `Moderator.score`.
- `nudenet` uses `channels: rgb` (bake-off winner over the BGR that `nudenet==3.4.2` feeds). Per-class score = best box
  whose argmax is that class (NudeNet semantics). Rules may have `unless: {signal, above}` (cancels REVIEW, never REJECT).
- API errors: `{"error": {"code", "message"}}` via `ApiError` / `ImageError`.
- Endpoints are deliberately limited (owner): /health, /v1/moderate/image(+/batch), /v1/feedback, /v1/review*.
  Auth = HS256 JWT (`CTC_JWT_SECRET`, from env or .env); caller = `sub` or `id` claim; `role: admin` for review/feedback.
- Never commit the JWT secret; never add debug prints of tokens/secrets.
- `eval.py report` overwrites EVAL_REPORT.md (appends runs/robustness.md). Run `robust` before `report`.

## PROGRESS (update at each checkpoint)
- [x] (a) eval harness + model export/parity + bake-off
- [x] (b) cascade + fusion rules + policy config
- [x] (c) FastAPI: JWT auth, rate limit, backpressure, shadow, kill switch, review queue API, tests
- [x] (d) load test on M4. t3.small numbers are estimates (not measured on AWS)
- [x] (e) Containerfile/compose.yaml/Caddy/deploy.sh/CI/docs (stack verified locally under Podman 6.0.2 + podman-compose, 2026-10-13)
- [x] Bake-off on 185 labeled images (2026-10-13): NudeNet-rgb + Freepik(INT8) + Falconsai gate; thresholds CV-tuned; baseline set
- [x] Cleanup 2026-10-13: removed image_url/SSRF, metrics, bake-off-only models, stacker/plots; metrics unchanged
- [ ] Owner: remove duplicate images (README), grow labeled set toward ~1.5k/2.5k/750, re-run eval
- [ ] Owner: first deploy to t3.small in shadow mode; run loadtest.py there to replace the estimates
