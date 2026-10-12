# CTC image moderation (Phase 1: images)

Self-hosted, CPU-only service that decides **APPROVE / REVIEW / REJECT** for user-uploaded images. It makes no external or paid API calls: all inference runs locally on ONNX Runtime.

- **App developers:** read [INTEGRATION.md](INTEGRATION.md).
- **Licenses and risks:** [LICENSES.md](LICENSES.md).
- **Accuracy:** [EVAL_REPORT.md](EVAL_REPORT.md). Not measured yet, because the labeled data hasn't been supplied.

## How it decides

```
bytes → validate (magic bytes, 15 MB, 40 MP, frames) → EXIF-rotate, to RGB, drop metadata, ≤1280 px
      → stage 1 (every image):    NudeNet-320n body-part detector + Falconsai ViT classifier   ~50–90 ms on M4
      → stage 2 (only if stage 1 isn't clearly safe and not already REJECT): Freepik EVA02   +~400 ms on M4
      → rules in config.yaml: any signal ≥ reject → REJECT, ≥ review → REVIEW (worst wins), per-context overrides
      → near-duplicate (pHash) of an earlier REJECT → at least REVIEW
```

- Every model is a `Scorer` (`ctc_mod/scorers.py`). Phase 2 video will reuse them as frame scorers.
- Turning a model off (`enabled: false`) also turns off its rules, so the AGPL NudeNet layer can be removed without code changes.
- Animated GIF/WebP: first, middle and last frames, max score. Transparent images: scored both composited and with alpha ignored, because hidden pixels reappear if alpha is later dropped.
- Cache: model scores by exact SHA-256, in memory. pHash is only ever used to *escalate*, never to approve.
- Hooks with interfaces only: `AgeEstimator` (may only add a REVIEW trigger) and `HashMatcher` (known-illegal-content matching through a vetted provider).

## Layout

```
ctc_mod/scorers.py  model adapters + hook interfaces       ctc_mod/engine.py  decode, cascade, rules, cache
ctc_mod/api.py      FastAPI, auth, limits, review queue    config.yaml        models + policy (versioned)
eval.py             bake-off, tuning, CIs, robustness, gate export.py          download/convert models to ONNX
loadtest.py         locust                                  client.py          typed client example
```

## Run locally

```bash
uv sync                                            # Python 3.12 (pinned), runtime + dev deps
uv run --group export python export.py             # one-off: models/*.onnx (~1.5 GB incl. bake-off variants)
uv run python -m ctc_mod.api newkey my-backend     # prints a key + a line for keys.yaml
uv run python -m ctc_mod.api newkey dashboard admin
uv run uvicorn ctc_mod.api:app --port 8000         # ONE process; /docs for OpenAPI
uv run pytest -q                                   # unit + API tests (fake model, no weights needed)
```

`keys.yaml` holds only SHA-256 hashes of keys: `keys: [{id, sha256, role: client|admin, rate_per_min}]`. It's re-read when the file changes. **To rotate a key:** add the new key, switch callers over, then delete the old line.

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `CTC_SHADOW` | 0 | 1 = return APPROVE plus `shadow_decision`, `enforced:false` |
| `CTC_KILL_SWITCH` | 0 | 1 = skip models; everything goes to REVIEW (`KILL_SWITCH`) |
| `CTC_CONCURRENCY` | 2 | parallel inferences. Set to the vCPU count. |
| `CTC_ORT_THREADS` | 0 (all) | threads per inference. Use vCPUs ÷ concurrency. |
| `CTC_QUEUE` | 16 | requests waiting beyond this get 503 `OVERLOADED` |
| `CTC_TIMEOUT_S` | 10 | max time spent waiting for a worker |
| `CTC_ALLOW_URL` | 0 | enable SSRF-guarded `image_url` input |
| `CTC_CONFIG`, `CTC_KEYS_FILE`, `CTC_DB`, `CTC_MODELS_DIR` | | file locations |

Policy changes (anything in `config.yaml`): edit, **bump `policy_version`**, re-run `eval.py report`, then deploy. Each response carries `policy_version`.

## Performance and capacity

**Measured on a Mac mini M4** (10 cores, locust, the 18 sample images, about 28% routed to stage 2, no cache hits):

| CTC_CONCURRENCY × ORT threads, clients | req/s | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 × all, 1 | 8.2 | 49 ms | 400 ms | 410 ms |
| 2 × 1, 2 | 9.0 | 93 ms | 750 ms | 800 ms |
| 4 × 2, 8 | 13.8 | 400 ms | 1300 ms | 1600 ms |

Memory: about 0.95 GB RSS (FP32 models), about 0.55 GB with INT8.

### t3.small (your instance): estimate, **not measured**

I haven't run anything on AWS. These numbers scale from the M4 per-thread timings with an **assumed** factor of 2–2.5× slower per core for t3's Xeon. **Verify** with `loadtest.py` on the instance.

- **Hardware:** 2 vCPU (one physical core with hyper-threading), 2 GiB RAM, burstable with a **20%-per-vCPU baseline**. On-demand price is $0.0208/h (about $15/month).
- **Per-image latency:** stage-1-only images about 150–250 ms. Images that go to stage 2 (Freepik) about **1.0–1.4 s**.
- **The ~500 ms p95 target is not reachable on a t3.small** if more than about 5% of uploads go to stage 2. Ways to fix it, to be decided after the bake-off: a lighter stage 2 (e.g. AdamCodd ~180 ms on M4), a tighter cascade, or a c7i/c7g instance.
- **Throughput while bursting:** about **2–4 images/s**, depending on the stage-2 rate.
- **Sustained throughput at the CPU-credit baseline:** about **0.6 images/s ≈ 50,000 uploads/day**, spread evenly. Peaks above that draw down credits. A full credit bank (576) covers about 6 h of flat-out load.
- **"Unlimited" mode:** T3's default. Sustained load over baseline is billed at $0.05 per vCPU-hour. At 100% CPU all month that's about +$58, which makes t3.small *more expensive* than a c7i.large.
- **When to scale:** move to **c7i.large** (x86, $0.0892/h ≈ $65/month) or **c7g.large** (Graviton, $0.0725/h ≈ $53/month; test the ARM build first) at about 40,000 uploads/day, or when `CPUCreditBalance` keeps trending down. After that, add instances behind a load balancer. The in-memory rate limit and cache are per instance, so move them to Redis at that point.

## Deploy (single EC2 instance, us-east-1)

1. **Launch the instance:** Ubuntu 24.04 LTS, t3.small, 20 GB gp3. Security group: 22 from your IP only; 80 and 443 open (or 443 from your backend's IP only).
2. **Networking:** attach an Elastic IP and point a DNS A record (`CTC_DOMAIN`) at it.
3. **Install Docker:** `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker ubuntu`
4. **Create the server-only files** in `~/ctc` on the instance: `.env` (from `.env.example`, starting with `CTC_SHADOW=1`) and `keys.yaml`. Make these on the server: `deploy.sh` never overwrites or copies them.
5. **Deploy from your Mac:** `./deploy.sh ubuntu@<elastic-ip>`. It runs the tests, rsyncs the code plus the enabled ONNX weights (about 0.7 GB the first time), and runs `docker compose up -d --build`. Caddy gets the TLS certificate automatically.

**Not verified here:** Docker isn't installed on the dev Mac, so the image has not been built locally. The first deploy is the first real build. Check `docker compose logs api` and `curl https://$CTC_DOMAIN/readyz`.

### Monthly cost model (us-east-1, on-demand, Oct 2026; check current prices)

| traffic | setup | compute | other | total ≈ |
|---|---|---|---|---|
| Low (≤ 20k uploads/day) | 1 × t3.small | $15 | IPv4 $3.65, 20 GB gp3 $1.60, snapshots ~$1 | **≈ $21** |
| Medium (20–80k/day) | 1 × c7i.large (or c7g.large) | $65 ($53) | same as above | **≈ $71 ($59)** |
| High (~500k/day, ~20 req/s peak) | 3 × c7i.xlarge + ALB | ~$390 | ALB ~$20–30, IPv4, EBS | **≈ $430+** (re-measure first) |

**Quiet costs:**
- Public IPv4 is $0.005/h ($3.65/month), including Elastic IPs.
- T3 unlimited CPU surplus is $0.05/vCPU-h. Watch `CPUSurplusCreditsCharged`.
- Each EBS volume and snapshot costs money.
- CloudWatch Logs ingestion is $0.50/GB. We log to Docker json-file locally by default.
- NAT gateways and load balancers have hourly fees. We use neither on one instance.
- Inbound data transfer is free and responses are about 1 KB, so egress is negligible.

**Billing alarm:** go to Billing → Budgets → create a monthly cost budget (e.g. $40) with email alerts at 80% and 100%. Also add a CloudWatch alarm on the instance's `CPUCreditBalance` < 50.

## Runbook

| situation | action |
|---|---|
| Model misbehaving or too many false rejects | `CTC_KILL_SWITCH=1` in `.env` → `docker compose up -d`. Everything goes to REVIEW. |
| New policy or thresholds | Edit `config.yaml`, bump `policy_version`, run `eval.py report` (and `--final` if this is a release), commit (CI gate), deploy. |
| Rollback | `git checkout <previous tag> && ./deploy.sh …` (models are pinned by sha256 prefix in each response's `models[].version`). |
| 503 `OVERLOADED` rising | Check `ctc_inflight` and CPU credits. Scale up the instance type. Don't raise `CTC_QUEUE` much, because that only adds latency. |
| Review queue growing | `GET /v1/review/stats` (`oldest_pending_age_s`). Add moderators or tune `SUGGESTIVE_CONTENT` thresholds after looking at the feedback data. |
| Going live | Run in shadow mode for 1–2 weeks. Compare `shadow_decision` with moderator outcomes (SQLite `data/ctc.db`, `decisions` table). Then set `CTC_SHADOW=0`. |
| Backups | `data/ctc.db` holds decisions and feedback (no images). Snapshot the EBS volume weekly, or copy the db to S3. |

**Metrics:** `GET /metrics` with an admin key. Key metrics: `ctc_decisions_total{decision,context,enforced}`, `ctc_latency_seconds`, `ctc_inflight`, `ctc_errors_total{code}`, `ctc_cache_hits_total`. Scrape them with a self-hosted Prometheus and Grafana, or with the CloudWatch agent's Prometheus support.

**Logs:** one JSON line per decision on stdout, containing request_id, key_id, user_id, decision, reasons, sha256, phash and latency. **Never** image content.

## Security

- **Decoders on untrusted input:** image decoders (libjpeg, libpng, libwebp, libheif/libde265, giflib via Pillow) are a classic attack surface. To mitigate:
  - Format is validated by magic bytes, and only five formats are accepted.
  - Byte, pixel and frame limits are checked before a full decode (decompression-bomb guard).
  - The container runs as a non-root user with no secrets inside it.
  - Pillow and pi-heif get security fixes when you run `uv lock --upgrade-package pillow pi-heif` regularly, then rebuild.
  - Optionally add a seccomp/AppArmor profile, or run decoding in a separate short-lived process.
- **No raw images stored:** bytes live only in memory for the request. EXIF and other metadata are never read except for orientation, and never returned.
- **Keys:** stored as SHA-256 hashes only. `admin` and `client` roles are separate. Per-key rate limits apply.
- **Network:** TLS terminates at Caddy. The body cap is 20 MB at the proxy, and limits are enforced again in the app.
- **`image_url`** is disabled by default. When enabled, it allows only http(s) on ports 80/443 and resolved public IPs (pinned to defeat DNS rebinding), with at most 2 redirects (each re-checked), a 5 s timeout and the size cap.
- **Dependencies:** pinned in `uv.lock`, and the base image is pinned to an exact tag.

## Minors and illegal content (read this)

This service **does not detect CSAM**, and an NSFW classifier must never be presented as doing so. The `AgeEstimator` and `HashMatcher` hooks exist but have no implementation.

- **Hash matching:** use a vetted provider. Options to apply for include Microsoft PhotoDNA (free for qualifying organizations), Thorn Safer (commercial), and the NCMEC hash-sharing programs (Meta's PDQ is the open-source hash algorithm used by many of them).
- **Obligations to check with counsel for your jurisdiction.** These are not legal conclusions:
  - **US:** electronic service providers that obtain actual knowledge of apparent CSAM must report it to NCMEC's CyberTipline (18 U.S.C. § 2258A), and there are evidence-preservation rules after a report (the period was extended by the 2024 REPORT Act). Know what you're allowed and required to retain, and who may view it.
  - **EU users:** the Digital Services Act (notice and action, transparency reporting) and GDPR. Note that age estimation from faces may count as biometric processing.
  - **UK users:** the Online Safety Act and Ofcom's codes.
  - Your app store terms (Apple and Google) also have UGC moderation requirements.
- **Moderator wellbeing:** moderators reviewing REVIEW items will see disturbing content. Plan for blurred-by-default previews, opt-in reveal, and support.

## Known limitations

- **No measured accuracy yet** (no labeled data). Thresholds are the prototype's values.
- Expected weak spots, to be confirmed by the eval:
  - Drawn, anime and AI-generated explicit content (NudeNet is photo-trained).
  - Sexual acts with no exposed parts.
  - Sheer or wet clothing.
  - Sex toys (no model has a dedicated class).
  - Heavily censored or sticker-covered explicit images.
  - Small regions in very large images (everything is downscaled to ≤1280 px and then to model size; no tiling yet).
- Real-world review rate depends on your upload mix. The eval's "review_rate" depends on the labeled mix and is not a forecast.
- One instance means a single point of failure, and the rate-limit and cache state is lost on restart.
- Batch requests are processed per-image (no tensor batching). Measured throughput was CPU-bound already.
- On a t3.small, p95 latency will probably exceed 500 ms. See "Performance and capacity".

## Sources for prices

[t3.small](https://calculator.holori.com/aws/ec2/t3.small/us-east-1), [c7g.large](https://calculator.holori.com/aws/ec2/c7g.large/us-east-1), [c7i.large](https://calculator.holori.com/aws/ec2/c7i.large) (third-party aggregators, 2026; confirm on aws.amazon.com/ec2/pricing).
