# CTC image moderation (Phase 1: images)

Self-hosted, CPU-only service that decides **APPROVE / REVIEW / REJECT** for user-uploaded images. It makes no external or paid API calls: all inference runs locally on ONNX Runtime, in Podman containers.

- **App developers:** read [INTEGRATION.md](INTEGRATION.md).
- **Licenses and risks:** [LICENSES.md](LICENSES.md).
- **Accuracy:** [EVAL_REPORT.md](EVAL_REPORT.md). Measured on 185 labeled images, so the confidence intervals are wide. See "Model choice".

## Policy (owner-confirmed 2026-10-13)

| decision | content |
|---|---|
| REJECT | **any** exposed nudity, however small (genitals, anus, female nipple/areola/breast, bare buttocks), sexual acts, sex toys, explicit drawn/AI content |
| REVIEW | breastfeeding, thong / extra-revealing swimwear, other borderline images |
| APPROVE | normal bikini and swimwear, lingerie, shirtless men, everyday photos |

## How it decides

```
bytes → validate (magic bytes, 15 MB, 40 MP, frames) → EXIF-rotate, to RGB, drop metadata, ≤1280 px
      → stage 1 (every image): NudeNet-320n body-part detector + Falconsai ViT (INT8, gate only)  ~50–90 ms on M4
      → stage 2 (unless stage 1 is clearly safe or already REJECT): Freepik EVA02 (INT8)          +~300 ms on M4
      → rules in config.yaml: any signal ≥ reject → REJECT, ≥ review → REVIEW (worst wins), per-context overrides
      → near-duplicate (pHash) of an earlier REJECT → at least REVIEW
```

- Every model is a `Scorer` (`ctc_mod/scorers.py`). Phase 2 video will reuse them as frame scorers.
- Turning a model off (`enabled: false`) also turns off its rules, so the AGPL NudeNet layer can be removed without code changes.
- Animated GIF/WebP: first, middle and last frames, max score. Transparent images: scored both composited and with alpha ignored, because hidden pixels reappear if alpha is later dropped.
- Cache: model scores by exact SHA-256, in memory. pHash is only ever used to *escalate*, never to approve.
- Hooks with interfaces only: `AgeEstimator` (may only add a REVIEW trigger) and `HashMatcher` (known-illegal-content matching through a vetted provider).

## Model choice (bake-off, 2026-10-13)

The data: 185 unique labeled images (82 safe, 23 review, 80 reject). 149 were used for development and 36 held out for the final test. **This is far too small for tight numbers.** About 1,500 / 2,500 / 750 are needed (see EVAL_REPORT.md).

| model (dev split, explicit vs safe) | ROC AUC [95% CI] | ms/image (M4, 1 thread) |
|---|---|---|
| Freepik EVA02 (`high`) | **0.975** [0.939–0.997] | ~300 |
| NudeNet 320n, RGB input (max exposed part) | 0.953 [0.905–0.982] | 9 |
| NudeNet 320n, BGR input (what `nudenet==3.4.2` does) | 0.948 [0.901–0.979] | 9 |
| NudeNet 640m | 0.957 [0.909–0.985] | 96 |
| Falconsai ViT | 0.926 [0.877–0.964] | 36 |
| Marqo ViT-tiny | 0.856 | 20 |
| AdamCodd ViT | 0.801 | 109 |

**Chosen: NudeNet-320n (RGB) + Freepik, with Falconsai as a cheap gate.** Four simpler rule sets were compared using 5-fold cross-validation on dev. This one had the lowest cost (1.05–1.18, against 1.6–2.5 for the others), but the differences are within noise. INT8 models made **identical decisions** to FP32 on all 149 dev images and use about 0.4 GB less RAM, so the service runs INT8.

Rejected options:
- Tuning many thresholds overfits at this data size: tuned on all rules, cross-validated cost was 2.5 vs 2.2 untuned.
- The logistic-regression stacker sent 98% of safe images to REVIEW.
- A "suggestive" rule fights the bikini/lingerie-allowed policy.

| result (CV on dev, 149 images) | value [95% CI] |
|---|---|
| explicit auto-approved | 1 of 70 = 1.4% [0.3–7.7] |
| explicit auto-rejected | 65 of 70 = 92.9% |
| safe auto-rejected | 0 of 65 = 0% [0–5.6] |
| safe sent to review | 14 of 65 = 21.5% [13–33] |
| **held-out test (36 images)** | explicit approved 1 of 10, safe rejected 0 of 17, safe→review 3 of 17 |

**What this means:** with this data we can say "most explicit images are auto-rejected and almost no safe photo is auto-rejected". We **cannot** claim ≤1% explicit images missed: the upper bound is about 8% on dev and 40% on test. Expect about 1 in 5 bikini or lingerie photos to go to REVIEW until thresholds are re-tuned on more data.

**Please double-check these labels** (all models strongly disagree with the label):
- `labeled/safe/123.jpeg`: every model says explicit.
- `labeled/reject/images (48).jpeg`: every model says completely safe.

Also check whether `reject/images (48)213.jpeg` and `safe/images (20)213.jpeg` are near-copies of dev images. If they are, the test split is partly leaking.

**Evasion (84 explicit and borderline dev images):** mirror, crop, rotate, meme text, blur and heavy JPEG compression let at most 2 explicit images escape to APPROVE per transform. The rest drop to REVIEW at worst. **Stickers or emoji covering the exposed parts is the real weakness: 6 of 70 explicit images escaped.** Censored images should be labeled `review` or `reject` in future data, and re-tuned.

## Layout

```
ctc_mod/scorers.py  model adapters + hook interfaces       ctc_mod/engine.py  decode, cascade, rules, cache
ctc_mod/api.py      FastAPI, auth, limits, review queue    config.yaml        models + policy (versioned)
eval.py             bake-off, tuning, CIs, robustness, gate export.py          download/convert models to ONNX
loadtest.py         locust                                  client.py          typed client example
Containerfile, compose.yaml, Caddyfile, deploy.sh           Podman deployment
```

## Which file to run

| goal | command (from `NSFWDetector/`) |
|---|---|
| install deps | `uv sync` |
| download/convert models (one-off) | `uv run --group export python export.py freepik falconsai nudenet-320n` |
| set the JWT secret | put `CTC_JWT_SECRET=<same secret your backend signs with>` in `.env`. The dev server and the containers both read it from there; restart after changing it. |
| make a test token | `uv run python -m ctc_mod.api token my-backend` (add `admin` for the review endpoints; optional 3rd arg = hours) |
| run the API (dev) | `uv run uvicorn ctc_mod.api:app --port 8000`, then open http://localhost:8000/docs |
| run the API in containers | `cp .env.example .env` (set `CTC_DOMAIN=localhost`), then `podman-compose up -d --build --force-recreate` → https://localhost |
| unit + API tests | `uv run pytest -q` (no weights needed) |
| score labeled data | `uv run python eval.py score --data ../labeled` (resumable) |
| accuracy report | `uv run python eval.py report`, then `--final` once for a release |
| evasion suite | `uv run python eval.py robust` |
| regression gate (also in CI) | `uv run python eval.py gate` |
| load test | `CTC_API_KEY=… uv run locust -f loadtest.py --headless -u 2 -r 2 -t 1m --host http://localhost:8000` (add `CTC_INSECURE=1` against local Caddy) |
| try the client | `client.py` (copy into your backend) |

**Auth:** `Authorization: Bearer <JWT>`, HS256, signed with `CTC_JWT_SECRET`. `sub` and `exp` are required, and `role: admin` is needed for `/v1/review*`. Rate limits apply per `sub`. To rotate the secret, change it in both the token issuer and the server `.env`, then restart; old tokens stop working. See INTEGRATION.md.

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `CTC_JWT_SECRET` | (none, required) | HS256 secret for bearer tokens, ≥ 32 random chars |
| `CTC_RATE_PER_MIN` | 600 | requests per minute per token `sub` |
| `CTC_SHADOW` | 0 | 1 = return APPROVE plus `shadow_decision`, `enforced:false` |
| `CTC_KILL_SWITCH` | 0 | 1 = skip models; everything goes to REVIEW (`KILL_SWITCH`) |
| `CTC_CONCURRENCY` | 2 | parallel inferences. Set to the vCPU count. |
| `CTC_ORT_THREADS` | 0 (all) | threads per inference. Use vCPUs ÷ concurrency. |
| `CTC_QUEUE` | 16 | requests waiting beyond this get 503 `OVERLOADED` |
| `CTC_TIMEOUT_S` | 10 | max time spent waiting for a worker |
| `CTC_ALLOW_URL` | 0 | enable SSRF-guarded `image_url` input |
| `CTC_CONFIG`, `CTC_DB`, `CTC_MODELS_DIR` | | file locations |

**Policy changes** (anything in `config.yaml`): edit, **bump `policy_version`**, re-run `eval.py report`, then deploy. Each response carries `policy_version`.

## Performance and capacity

**Measured:** the API container on the Mac mini M4, **limited to 2 CPUs**, with the production config. Load came from locust through Caddy/HTTPS, using the labeled images (about 80% of them go to stage 2, which is pessimistic; real uploads are mostly ordinary photos):

| concurrent clients | req/s | p50 | p95 | p99 | failures |
|---|---|---|---|---|---|
| 1 | 1.8 | 150 ms | 1200 ms | 1300 ms | 0 |
| 2 | 3.6 | 160 ms | 1300 ms | 1400 ms | 0 |
| 4 | 3.4 | 1400 ms | 2500 ms | 2600 ms | 0 (queueing) |

Memory: **about 0.47 GiB** under a burst of 80 parallel uploads (INT8 models, ORT memory arena off).

**Stage-2 rate on the labeled data:** 55% of safe images, 78% of review, 99% of reject. Your real traffic is mostly ordinary selfies, so expect a lower rate, but measure it from logs: the `freepik.*` keys appear in `scores` only when stage 2 ran.

### t3.small (your instance): estimate, **not measured on AWS**

M4 cores are faster than a t3's Xeon (assumed 2–2.5× per core). A t3.small's 2 vCPUs are one physical core with hyper-threading.

- **Hardware:** 2 vCPU, 2 GiB RAM, burstable, **20%-per-vCPU baseline**. On-demand price is $0.0208/h (about $15/month).
- **Latency:** about 0.3–0.5 s for images settled in stage 1, about **2–3 s** for images that go to stage 2. **The 500 ms p95 target is not met on a t3.small.**
- **Throughput while bursting:** about **1–2 images/s**, depending on the stage-2 rate.
- **Sustained throughput at the CPU-credit baseline:** about **0.2–0.4 images/s ≈ 15–35k uploads/day**, spread evenly. A full credit bank (576) covers about 6 h of flat-out bursting. T3 "unlimited" mode (the default) bills sustained load over baseline at $0.05/vCPU-h, up to about +$58/month.
- **Recommendation:** a t3.small is fine to **launch in shadow mode** and for under about 15k uploads/day. Beyond that, or if you need p95 < 1 s, move to **c7i.large** ($0.0892/h ≈ $65/month, non-burstable, faster cores) or **c7g.large** (Graviton, ≈ $53/month; measure it first). Run `loadtest.py` on the real instance to replace these estimates.

## Deploy with Podman (single EC2 instance, us-east-1)

**One-time server setup.** Launch Ubuntu 24.04 LTS on a t3.small with a 20 GB gp3 disk. Security group: 22 from your IP only; 80 and 443 open (or 443 from your backend's IP only). Attach an Elastic IP and point a DNS A record (`CTC_DOMAIN`, e.g. `moderation.yourdomain.com`) at it. Then, over SSH:

```bash
sudo apt update && sudo apt install -y podman podman-compose rsync
echo 'net.ipv4.ip_unprivileged_port_start=80' | sudo tee /etc/sysctl.d/99-ctc.conf && sudo sysctl --system   # rootless 80/443
sudo loginctl enable-linger ubuntu                      # keep rootless containers running after logout
systemctl --user enable --now podman-restart.service    # restart containers with restart policy after reboot
mkdir -p ~/ctc
```

**Server-only files.** Create these on the server; `deploy.sh` never copies or overwrites them:
- `~/ctc/.env`: copy `.env.example` and set `CTC_DOMAIN`. **Keep `CTC_SHADOW=1` for the first 1–2 weeks.**
- In `~/ctc/.env`, set **`CTC_JWT_SECRET`** to a long random secret (`openssl rand -base64 48`) and share it only with the backend and dashboard that mint tokens.

**Deploy from your Mac:**

```bash
cd NSFWDetector
./deploy.sh ubuntu@<elastic-ip>
```

This runs the tests, rsyncs the code plus the enabled ONNX weights (about 0.2 GB), then runs `podman-compose up -d --build --force-recreate` on the server. Caddy gets the Let's Encrypt certificate automatically.

**Verify:**

```bash
curl https://$CTC_DOMAIN/health
curl -H "Authorization: Bearer <token>" -F file=@photo.jpg https://$CTC_DOMAIN/v1/moderate/image
```

For logs, run on the server: `podman logs -f ctc_api_1` (names are shown by `podman ps`).

**Go live:** after the shadow period, compare `shadow_decision` with moderator outcomes. Then set `CTC_SHADOW=0` in `.env` and run `podman-compose up -d --force-recreate` on the server.

### Monthly cost model (us-east-1, on-demand, Oct 2026; check current prices)

| traffic | setup | compute | other | total ≈ |
|---|---|---|---|---|
| Low (≤ 15k uploads/day) | 1 × t3.small | $15 | IPv4 $3.65, 20 GB gp3 $1.60, snapshots ~$1 | **≈ $21** |
| Medium (15–80k/day) | 1 × c7i.large (or c7g.large) | $65 ($53) | same as above | **≈ $71 ($59)** |
| High (~500k/day, ~20 req/s peak) | 3 × c7i.xlarge + ALB | ~$390 | ALB ~$20–30, IPv4, EBS | **≈ $430+** (re-measure first) |

**Quiet costs:**
- Public IPv4 is $0.005/h ($3.65/month), including Elastic IPs.
- T3 unlimited CPU surplus is $0.05/vCPU-h. Watch `CPUSurplusCreditsCharged`.
- Each EBS volume and snapshot costs money.
- CloudWatch Logs ingestion is $0.50/GB. Containers log to journald on the instance by default.
- NAT gateways and load balancers have hourly fees. We use neither on one instance.
- Inbound data transfer is free and responses are about 1 KB, so egress is negligible.

**Billing alarm:** go to Billing → Budgets → create a monthly cost budget (e.g. $40) with email alerts at 80% and 100%. Also add a CloudWatch alarm on the instance's `CPUCreditBalance` < 50.

## Runbook

| situation | action |
|---|---|
| Model misbehaving or too many false rejects | `CTC_KILL_SWITCH=1` in `.env` → `podman-compose up -d --force-recreate`. Everything goes to REVIEW. |
| New policy or thresholds | Edit `config.yaml`, bump `policy_version`, run `eval.py report` (and `--final` if this is a release), commit (CI gate), deploy. |
| Rollback | `git checkout <previous tag> && ./deploy.sh …` (models are pinned by sha256 prefix in each response's `models[].version`). |
| 503 `OVERLOADED` rising | Check `ctc_inflight` and CPU credits. Scale up the instance type. Don't raise `CTC_QUEUE` much, because that only adds latency. |
| Review queue growing | `GET /v1/review/stats` (`oldest_pending_age_s`). Add moderators or tune `SUGGESTIVE_CONTENT` thresholds after looking at the feedback data. |
| Going live | Run in shadow mode for 1–2 weeks. Compare `shadow_decision` with moderator outcomes (SQLite `data/ctc.db`, `decisions` table). Then set `CTC_SHADOW=0`. |
| Backups | The `ctc_data` volume holds `ctc.db` (decisions and feedback, no images). Use `podman volume export ctc_ctc_data > backup.tar` weekly, or snapshot the EBS volume. |


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
