# CTC image moderation (Phase 1: images)

This is a self-hosted, CPU-only service that decides **APPROVE / REVIEW / REJECT** for user-uploaded images. It makes no external or paid API calls: inference runs locally on ONNX Runtime, in Podman containers behind Caddy (HTTPS).

- **App developers:** [INTEGRATION.md](INTEGRATION.md)
- **Licenses and risks:** [LICENSES.md](LICENSES.md)
- **Accuracy details:** [EVAL_REPORT.md](EVAL_REPORT.md)

## Policy (owner-confirmed 2026-10-13)

| decision | content |
|---|---|
| REJECT | **Any** exposed nudity, however small: genitals, anus, female nipple/areola/breast, bare buttocks. Also sexual acts, sex toys, and explicit drawn or AI content. |
| REVIEW | Breastfeeding, thong or extra-revealing swimwear, and other borderline images. |
| APPROVE | Normal bikini and swimwear, lingerie, shirtless men, everyday photos. |

## How it decides

```
bytes → validate (magic bytes, 15 MB, 40 MP, frames) → EXIF-rotate, to RGB, drop metadata, ≤1280 px
      → stage 1 (every image): NudeNet-320n body-part detector + Falconsai ViT gate (INT8)   ~50–90 ms on M4
      → stage 2 (unless stage 1 is clearly safe or already REJECT): Freepik EVA02 (INT8)      +~300 ms on M4
      → rules in config.yaml: signal ≥ reject → REJECT, ≥ review → REVIEW (worst wins)
      → near-duplicate (pHash) of an earlier REJECT → at least REVIEW
```

- **Model layer:** every model is a `Scorer` (`ctc_mod/scorers.py`), so Phase 2 video can reuse them per frame. Setting `enabled: false` on a model also disables its rules, which makes the AGPL NudeNet layer removable.
- **Special image types:** for animated GIF/WebP, the first, middle and last frames are scored and the strictest wins. Transparent images are scored both composited and with alpha ignored.
- **Caching:** model scores are cached in memory by exact SHA-256. pHash is only used to *escalate* a decision, never to approve.
- **Hooks for later:** `AgeEstimator` and `HashMatcher` are interfaces only (see "Minors and illegal content").

## Files

```
ctc_mod/api.py      FastAPI: JWT auth, rate limit, backpressure, shadow mode, kill switch, review queue
ctc_mod/engine.py   decode/validate, cascade, rules, cache          ctc_mod/scorers.py  ONNX model adapters
config.yaml         models + policy (versioned)                     export.py           build models/*.onnx
eval.py             scoring, tuning, CIs, robustness, CI gate       loadtest.py         locust load test
client.py           typed client to copy into your backend          tests/              synthetic-image tests
Containerfile, compose.yaml, Caddyfile, deploy.sh                    Podman deployment
```

## Commands (run from `NSFWDetector/`)

| goal | command |
|---|---|
| install | `uv sync` |
| build models (one-off, ~3 min) | `uv run --group export python export.py` |
| config | `cp .env.example .env`, then set `CTC_JWT_SECRET` (the same secret your backend signs tokens with) |
| dev server + Swagger | `uv run uvicorn ctc_mod.api:app --port 8000` → http://localhost:8000/docs (reads `.env`) |
| test token | `uv run python -m ctc_mod.api token my-backend` (add `admin` for the review and feedback endpoints) |
| containers | `podman-compose up -d --build --force-recreate` → https://localhost (set `CTC_DOMAIN=localhost`) |
| tests | `uv run pytest -q` (no model weights needed) |
| accuracy | `uv run python eval.py score` → `robust` → `report` (`--final` once per release) → `gate` |
| load test | `CTC_TOKEN=$(uv run python -m ctc_mod.api token load) uv run locust -f loadtest.py --headless -u 2 -r 2 -t 1m --host http://localhost:8000` |

**Policy changes:** edit `config.yaml`, **bump `policy_version`**, re-run `eval.py report`, then deploy. Every response carries `policy_version`.

## Configuration (`.env`)

| var | default | meaning |
|---|---|---|
| `CTC_JWT_SECRET` | none (required) | HS256 secret for bearer tokens. Use ≥ 32 random chars. |
| `CTC_RATE_PER_MIN` | 600 | requests per minute per caller (`sub` or `id` claim) |
| `CTC_SHADOW` | 0 | 1 = always return APPROVE and put the real verdict in `shadow_decision` (`enforced: false`) |
| `CTC_KILL_SWITCH` | 0 | 1 = skip the models; everything goes to REVIEW (`KILL_SWITCH`) |
| `CTC_CONCURRENCY` / `CTC_ORT_THREADS` | 2 / all | parallel inferences / threads each. Set concurrency to the vCPU count and threads to vCPUs ÷ concurrency. |
| `CTC_QUEUE` / `CTC_TIMEOUT_S` | 16 / 10 | waiting requests allowed before 503 `OVERLOADED` / max wait for a worker |
| `CTC_DOMAIN` | | HTTPS hostname for Caddy |

## Model choice and accuracy (bake-off, 2026-10-13)

**Data:** 185 unique labeled images (82 safe, 23 review, 80 reject), split into 149 for development and 36 held out. **This is too small for tight numbers.** The target is about 1,500 / 2,500 / 750.

**Chosen stack:** NudeNet-320n (RGB) plus Freepik, with Falconsai as a cheap gate, using 5-fold cross-validated thresholds.
- Freepik scored a ROC AUC of 0.975 and NudeNet (strongest exposed part) 0.953.
- Falconsai (0.926), Marqo (0.86), AdamCodd (0.80) and NudeNet-640m (0.957, but 10× slower) were tested and not used.
- INT8 models made **identical decisions** to FP32 on all dev images and use about 0.4 GB less RAM.
- A logistic-regression stacker was also tested and lost (it sent 98% of safe images to REVIEW).

**Results:**

| | explicit auto-approved | explicit auto-rejected | safe auto-rejected | safe sent to review |
|---|---|---|---|---|
| dev, cross-validated (149) | 1 / 70 = 1.4% [0.3–7.7] | 65 / 70 | 0 / 65 = 0% [0–5.6] | 8 / 65 = 12.3% [6.4–22.5] |
| held-out test (36) | 1 / 10 | 9 / 10 | 0 / 17 | 2 / 17 |

**What this means:**
- Most explicit images are auto-rejected and no safe photo was auto-rejected.
- **We cannot yet claim ≤1% of explicit images are missed.** The 95% upper bound is about 8% on dev and 40% on test.
- **Shirtless men:** NudeNet often reads one pec as "female breast". Since policy `.4` (2026-10-13), two changes stop this sending shirtless men to REVIEW:
  - each detector box now counts only as the class NudeNet picked for it;
  - a male-chest guard: a *moderate* breast score with a male chest present no longer triggers REVIEW (a strong score still REJECTs).
- **Effect:** safe photos sent to REVIEW fell from 21.5% to 12.3% (all 185 images: 17 → 10). Explicit detection is unchanged.
- **Sexy but fully clothed photos** sometimes still go to REVIEW, through Freepik's `high` score at 0.06–0.17. The threshold stays at 0.02 because raising it even to 0.05 let 2 explicit dev images through. More labeled data is the way to move it safely.

**Known misses:**
- **Vintage anatomical engravings.** `reject/images (48).jpeg` is one; it was labeled `reject` but every model scores it safe. This needs more examples like it.
- **Stickers covering the exposed parts:** 6 of 70 explicit images escaped to APPROVE in the evasion suite.
- **Other transforms** (crop, rotate, mirror, blur, heavy JPEG compression, text overlay, screenshot of a screen) let 0–3 escape. The rest dropped to REVIEW at worst.

**Data hygiene:** `images (48)213.jpeg` and `images (20)213.jpeg` look like copies of dev images. Delete the copies before the next eval.

## Performance and capacity

**Measured:** the API container on the Mac mini M4 limited to **2 CPUs**, through Caddy. These are pessimistic, because about 80% of the labeled images reach stage 2:

| concurrent clients | req/s | p50 | p95 | failures |
|---|---|---|---|---|
| 1 | 1.8 | 150 ms | 1200 ms | 0 |
| 2 | 3.6 | 160 ms | 1300 ms | 0 |
| 4 | 3.4 | 1400 ms | 2500 ms | 0 (queueing) |

Memory is about **0.47 GiB** under a burst of 80 parallel uploads.

**t3.small (estimate; not measured on AWS):** a t3's Xeon is assumed to be 2–2.5× slower per core than the M4. The t3.small's 2 vCPUs are one hyper-threaded core with a 20% CPU-credit baseline.

| | estimate |
|---|---|
| latency, stage 1 only | 0.3–0.5 s |
| latency, with stage 2 | 2–3 s, so **p95 < 500 ms is not met** |
| throughput while bursting | 1–2 images/s |
| sustained throughput | about 15–35k uploads/day |

**Verdict:** a t3.small is fine to **launch in shadow mode** and for under about 15k uploads/day. Beyond that, move to **c7i.large** (≈ $65/month) or **c7g.large** (≈ $53/month; measure the ARM build first). Run `loadtest.py` on the instance to replace these estimates.

## Deploy with Podman (one EC2 instance, us-east-1)

**One-time server setup.** Launch Ubuntu 24.04 LTS on a t3.small with a 20 GB gp3 disk. Security group: 22 from your IP only; 80 and 443 open (or 443 only from your backend's IP). Attach an Elastic IP and point a DNS A record (`CTC_DOMAIN`) at it. Then:

```bash
sudo apt update && sudo apt install -y podman podman-compose rsync
echo 'net.ipv4.ip_unprivileged_port_start=80' | sudo tee /etc/sysctl.d/99-ctc.conf && sudo sysctl --system
sudo loginctl enable-linger ubuntu                     # keep rootless containers running after logout
systemctl --user enable --now podman-restart.service   # bring containers back after a reboot
mkdir -p ~/ctc   # create ~/ctc/.env from .env.example: CTC_DOMAIN, CTC_JWT_SECRET, CTC_SHADOW=1 at first
```

**Deploy from your Mac:**

```bash
./deploy.sh ubuntu@<elastic-ip>
```

This runs the tests, rsyncs the code plus the enabled weights (about 0.2 GB), then runs `podman-compose up -d --build --force-recreate`. Caddy gets the Let's Encrypt certificate itself. `.env` lives only on the server and is never overwritten.

**Verify:**

```bash
curl https://$CTC_DOMAIN/health
podman logs -f ctc_api_1           # on the server (project dir ~/ctc); names are shown by `podman ps`
```

**Go live:** run in shadow mode for 1–2 weeks and compare `shadow_decision` with moderator outcomes. Then set `CTC_SHADOW=0` and recreate the containers.

**Monthly cost (us-east-1, on-demand, Oct 2026):**
- **t3.small setup ≈ $21:** compute $15.18, plus a public IPv4 at $3.65, 20 GB gp3 at $1.60 and snapshots at about $1.
- **c7i.large setup ≈ $71.**

**Quiet costs:**
- T3 "unlimited" mode bills sustained CPU above baseline at $0.05/vCPU-h. Watch `CPUSurplusCreditsCharged`.
- CloudWatch Logs costs $0.50/GB; we log to journald instead.
- NAT gateways and load balancers have hourly fees, and none are needed here.

**Billing alarm:** create a Billing → Budgets monthly budget (e.g. $40) with 80% and 100% email alerts, plus a CloudWatch alarm on `CPUCreditBalance` < 50.

## Runbook

| situation | action |
|---|---|
| Model misbehaving or too many false rejects | `CTC_KILL_SWITCH=1` in `.env`, then `podman-compose up -d --force-recreate`. Everything goes to REVIEW. |
| New thresholds or policy | Edit `config.yaml`, bump `policy_version`, `eval.py report` (`--final` for a release), commit (CI gate), deploy. |
| Rollback | `git checkout <tag> && ./deploy.sh …`. Each response's `models[].version` is the weights' sha256 prefix. |
| 503 `OVERLOADED` rising | Check CPU credits and scale up the instance type. Raising `CTC_QUEUE` only adds latency. |
| Review queue growing | Check `GET /v1/review/stats` (`oldest_pending_age_s`). Add moderators, or re-tune using the feedback data. |
| Backups | The `ctc_data` volume holds `ctc.db` (decisions and verdicts, no images): `podman volume export ctc_ctc_data > backup.tar` (on the server). |

**Logs:** one JSON line per decision (request_id, caller, user_id, decision, reasons, sha256, phash, latency). **Never** image content.

## Security

- **Untrusted image decoding** (libjpeg, libpng, libwebp, libheif/libde265 via Pillow) is the main attack surface. Mitigations:
  - the format is checked by magic bytes, and only 5 formats are accepted;
  - byte, pixel and frame limits are applied before the full decode;
  - the container runs as a non-root user;
  - keep the decoders current with `uv lock --upgrade-package pillow pi-heif` and a rebuild.
- **Images are never stored.** Only orientation is read from EXIF, and metadata is never returned.
- **Auth:**
  - HS256 JWT with `exp` required; only `role: admin` tokens can use the review and feedback endpoints.
  - **Use a long random `CTC_JWT_SECRET`.** A short one can be brute-forced from any single token.
- TLS terminates at Caddy, with a 20 MB body cap at the proxy and the limits enforced again in the app.
- Dependencies are pinned in `uv.lock`, and the base image is pinned to an exact tag.

## Minors and illegal content (read this)

This service **does not detect CSAM** and must never be presented as doing so. The `AgeEstimator` (which may only add a REVIEW trigger) and `HashMatcher` hooks are interfaces only.

- **Hash matching:** use a vetted provider. Options include Microsoft PhotoDNA (free for qualifying organizations), Thorn Safer, and the NCMEC hash-sharing programs (PDQ hashes).
- **Obligations to check with counsel** (not legal conclusions):
  - **US:** CyberTipline reporting under 18 U.S.C. § 2258A, and evidence preservation (extended by the 2024 REPORT Act).
  - **EU users:** the DSA and GDPR. Face-based age estimation may be biometric processing.
  - **UK users:** the Online Safety Act.
  - Apple's and Google's UGC rules also apply.
- **Moderators** will see disturbing content. Plan for blurred-by-default previews, opt-in reveal, and support.

## Known limitations

- **Small labeled set:** see "Model choice and accuracy". Thresholds should be re-tuned as data grows.
- **Weak spots:**
  - drawn, anime, AI-generated and anatomical art;
  - sexual acts with no exposed parts;
  - sheer clothing;
  - sex toys;
  - sticker-censored images;
  - small regions in very large images (no tiling yet).
- **Review rate:** the review rate on the labeled mix is not a forecast. Measure it from live traffic.
- **Single instance:** one point of failure, and the in-memory rate-limit and cache state is lost on restart.

Price sources: [t3.small](https://calculator.holori.com/aws/ec2/t3.small/us-east-1), [c7g.large](https://calculator.holori.com/aws/ec2/c7g.large/us-east-1), [c7i.large](https://calculator.holori.com/aws/ec2/c7i.large) (third-party aggregators; confirm on aws.amazon.com/ec2/pricing).
