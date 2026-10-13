# Integrating CTC image moderation

For app/backend developers. You don't need any ML knowledge. Interactive API docs are at `https://<host>/docs` (OpenAPI JSON at `/openapi.json`).

## 1. The flow

```
phone ──upload──▶ your backend ──store privately as PENDING──▶ POST /v1/moderate/image
                                                             │
          APPROVE ──▶ mark visible                           │
          REVIEW  ──▶ keep hidden; moderators decide in admin dashboard (section 4)
          REJECT  ──▶ keep hidden / delete; tell the user it broke the photo rules
```

Rules that matter:

1. **An image is never publicly visible before a decision.** Upload it to private storage with status `pending`, and only serve it to other users after `APPROVE` (or after a moderator approves it).
2. **Call the API from your backend, never from the mobile app.** Tokens are signed with a shared secret (`CTC_JWT_SECRET`) that must stay server-side.

## Authentication (JWT)

Every call except `/health` needs `Authorization: Bearer <token>`. The token must be:

- an **HS256** JWT signed with `CTC_JWT_SECRET` (other algorithms are refused);
- carrying **`sub`**, the caller's identity (e.g. `"app-backend"`), which is also used as the rate-limit bucket;
- carrying **`exp`**, an expiry time. Keep it short (e.g. 1 h) and mint new tokens as needed;
- carrying **`"role": "admin"`** for the `/v1/review*` endpoints. Only the admin dashboard backend should mint admin tokens.

Example in Python (PyJWT), Node (`jsonwebtoken`) is the same idea:

```python
import jwt, time
token = jwt.encode({"sub": "app-backend", "exp": int(time.time()) + 3600}, CTC_JWT_SECRET, algorithm="HS256")
admin = jwt.encode({"sub": "dashboard", "role": "admin", "exp": int(time.time()) + 3600}, CTC_JWT_SECRET, algorithm="HS256")
```

**Don't sign these tokens with your app's user-login secret.** If you do, any logged-in app user's token would also work here. Use a dedicated secret only for service-to-service calls.
3. **Save the `request_id`** next to the image record. You need it for the review queue and for feedback.
4. **Fail closed.** If moderation fails after retries, keep the image `pending` and retry later (or treat it as REVIEW). Never auto-publish on error.
5. **Shadow mode:** while `enforced` is `false`, `decision` is always `APPROVE` and the real verdict is in `shadow_decision`. Publish as you normally would, but store `shadow_decision` so we can compare it with human review before we switch enforcement on.

## 2. Request

```bash
curl -s https://moderation.example.com/v1/moderate/image \
  -H "Authorization: Bearer $CTC_TOKEN" \
  -F file=@photo.jpg \
  -F context=profile_photo \
  -F user_id=u_123 \
  -F idempotency_key=upload_9f8e7d
```

| field | required | notes |
|---|---|---|
| `file` | yes | JPEG, PNG, WebP, GIF, HEIC/HEIF. Max 15 MB and 40 megapixels. The format is detected from the bytes, not the filename. |
| `context` | no | `profile_photo` (stricter), `chat`, or `post`. |
| `user_id` | no | Your opaque user id. It's shown to moderators. Don't send emails or names. |
| `idempotency_key` | recommended | Use your upload id. If you retry with the same key, you get the original decision back. |

**HEIC/HEIF:** the API accepts HEIC directly. Converting to JPEG on the phone before upload is fine too, and it makes uploads smaller.

Animated GIF/WebP: the first, middle and last frames are checked, and the strictest result wins.

Batch: `POST /v1/moderate/image/batch` takes up to 8 `files` fields. It returns a list with one `result` or `error` per file, and each file counts against your rate limit.

## 3. Response

```json
{
  "request_id": "1f0c…",
  "decision": "REVIEW",
  "enforced": true,
  "shadow_decision": null,
  "reasons": ["EXPOSED_BREAST"],
  "scores": {"nudenet.FEMALE_BREAST_EXPOSED": 0.31, "falconsai.nsfw": 0.12},
  "regions": [{"label": "FEMALE_BREAST_EXPOSED", "score": 0.31, "box": [120, 80, 64, 60]}],
  "models": [{"name": "falconsai", "version": "ebc054600c04"}],
  "policy_version": "2026-10-12.1",
  "image_sha256": "548c…",
  "latency_ms": 212.4,
  "cached": false
}
```

Only use `decision` (and `enforced`) to decide what to do. `scores` and `regions` are for debugging and moderators, and their keys can change when models change. `reasons` codes are stable:

| reason | meaning |
|---|---|
| `EXPOSED_GENITALIA`, `EXPOSED_ANUS`, `EXPOSED_BREAST`, `EXPOSED_BUTTOCKS` | exposed body part detected |
| `EXPLICIT_CONTENT` | the whole image looks sexually explicit |
| `SUGGESTIVE_CONTENT` | borderline or suggestive (lingerie, poses…) |
| `NEAR_DUPLICATE_OF_REJECTED` | looks like an image we already rejected |
| `KILL_SWITCH` | automation is switched off and everything goes to humans |

Suggested user-facing messages: on REJECT, *"This photo doesn't meet our community guidelines."* On REVIEW, *"Your photo is being reviewed. This usually takes a few minutes."* Don't show reason codes to users, because that teaches people how to get around the filter.

## 4. Review queue (admin dashboard)

The REVIEW endpoints need a token with **`"role": "admin"`**. Mint admin tokens only on the admin dashboard's backend. **This service never stores images**, so your dashboard shows the image from your own storage, looked up by the `request_id` you saved.

```bash
# oldest-first pending items (max 200 per call; page with ?before=<created_at of last item>)
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" "https://…/v1/review?status=pending&limit=50"
# one item / queue stats (pending count, oldest pending age)
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" https://…/v1/review/<request_id>
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" https://…/v1/review/stats
# moderator verdict
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"decision":"REJECT","reviewer":"mod_anna","notes":"sheer top"}' https://…/v1/review/<request_id>
```

After you post a verdict, **your backend** publishes the image (APPROVE) or deletes or hides it (REJECT). Recently resolved items: `?status=resolved`. In shadow mode, items go into the queue marked `shadow: true` and are hidden unless you pass `include_shadow=true`.

`POST /v1/feedback {"request_id", "human_decision": "APPROVE"|"REJECT", "notes"}` records a human verdict on **any** decision (for example, a user appeals a REJECT and a moderator overturns it). Please send these. They're how we tune the thresholds.

## 5. Errors, retries, timeouts

All errors look like `{"error": {"code": "FILE_TOO_LARGE", "message": "…"}}`.

| HTTP | codes | what to do |
|---|---|---|
| 400 | `NO_IMAGE`, `EMPTY_FILE`, `BATCH_TOO_LARGE`, `URL_*` | Bug in the caller. Don't retry. |
| 401 / 403 | `UNAUTHORIZED`, `TOKEN_EXPIRED` / `FORBIDDEN` | Bad, missing or expired token, or no admin role. On `TOKEN_EXPIRED`, mint a new token and retry once. Otherwise don't retry; alert. |
| 413 | `FILE_TOO_LARGE`, `IMAGE_TOO_LARGE` | Ask the user for a smaller image, or downscale to ≤ 4096 px before uploading. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Not an accepted image format. |
| 422 | `CORRUPT_IMAGE`, `TOO_MANY_FRAMES`, `VALIDATION_ERROR` | Bad file or bad field. Don't retry. |
| 429 | `RATE_LIMITED` | Wait for `Retry-After` seconds, then retry. |
| 503 | `OVERLOADED`, `NOT_READY` | Retry with backoff: wait `Retry-After` plus jitter, at most ~3 tries, then keep the image pending and retry from a background job. |

- Set a client timeout of **15 s**. Typical latency is well under 1 s, but a borderline image on a small server can take 1–2 s.
- Always send `idempotency_key`, so a retry after a timeout doesn't create a duplicate review item.
- Better still, moderate from a **background job** after upload, not inside the user's HTTP request. The user sees "processing…" and the image appears once it's approved.

`client.py` in this repo is a ready-to-copy typed Python client that already does all of this.

## 6. Other endpoints

- `GET /health`: no auth. 200 `{"status":"ok","policy_version":…}` once models are loaded, 503 `NOT_READY` while starting.
