"""FastAPI service. Run: uvicorn ctc_mod.api:app --host 0.0.0.0 --port 8000 (ONE worker; see README).

Env: CTC_JWT_SECRET (required), CTC_RATE_PER_MIN, CTC_CONFIG, CTC_DB, CTC_SHADOW, CTC_KILL_SWITCH,
     CTC_CONCURRENCY, CTC_QUEUE, CTC_TIMEOUT_S, CTC_ALLOW_URL, CTC_MODELS_DIR, CTC_ORT_THREADS.
Auth: `Authorization: Bearer <JWT>`, HS256, `exp` required; caller id = `sub` or `id` (CTC app tokens use `id`);
      `role: admin` (any case) for /v1/review* and /v1/feedback.
Dev token: CTC_JWT_SECRET=... python -m ctc_mod.api token <sub> [admin] [hours]
"""
import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import socket
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import httpx
import jwt
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .engine import Decision, ImageError, Moderator



def _load_dotenv(path: str = ".env") -> None:
    """Local `uvicorn` runs: read KEY=VALUE lines from .env (podman-compose passes them itself).
    Variables already set in the environment win; .env is never copied into the image."""
    p = Path(path)
    if p.is_file():
        for line in p.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
env = os.environ.get
CONCURRENCY = int(env("CTC_CONCURRENCY", "2"))
QUEUE = int(env("CTC_QUEUE", "16"))
TIMEOUT_S = float(env("CTC_TIMEOUT_S", "10"))
SHADOW = env("CTC_SHADOW", "0") == "1"
KILL = env("CTC_KILL_SWITCH", "0") == "1"
ALLOW_URL = env("CTC_ALLOW_URL", "0") == "1"
BATCH_MAX = 8
JWT_SECRET = env("CTC_JWT_SECRET", "")
RATE_PER_MIN = int(env("CTC_RATE_PER_MIN", "600"))

log = logging.getLogger("ctc")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


# ---------------------------------------------------------------- schemas
class ModelInfo(BaseModel):
    name: str
    version: str


class Region(BaseModel):
    label: str
    score: float
    box: list[int] = Field(description="[x, y, width, height] in pixels of the EXIF-rotated, downscaled image")


class ModerationResult(BaseModel):
    request_id: str
    decision: Literal["APPROVE", "REVIEW", "REJECT"] = Field(description="What the app must enforce")
    enforced: bool = Field(description="False in shadow mode: decision is always APPROVE, see shadow_decision")
    shadow_decision: Literal["APPROVE", "REVIEW", "REJECT"] | None = None
    reasons: list[str] = Field(description="Stable machine-readable reason codes, e.g. EXPOSED_GENITALIA")
    scores: dict[str, float]
    regions: list[Region] = []
    models: list[ModelInfo]
    policy_version: str
    image_sha256: str
    latency_ms: float
    cached: bool


class BatchItem(BaseModel):
    filename: str | None
    result: ModerationResult | None = None
    error: dict | None = None


class Feedback(BaseModel):
    request_id: str
    human_decision: Literal["APPROVE", "REJECT"]
    reviewer: str | None = Field(None, max_length=128)
    notes: str | None = Field(None, max_length=2000)


class ReviewItem(BaseModel):
    request_id: str
    created_at: float
    user_id: str | None
    context: str | None
    reasons: list[str]
    scores: dict[str, float]
    image_sha256: str
    shadow: bool
    status: Literal["pending", "resolved"]
    human_decision: str | None = None
    reviewer: str | None = None
    notes: str | None = None


class ReviewResolve(BaseModel):
    decision: Literal["APPROVE", "REJECT"]
    reviewer: str | None = Field(None, max_length=128)
    notes: str | None = Field(None, max_length=2000)


# ---------------------------------------------------------------- storage (hashes + scores only, never images)
class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS decisions (
              request_id TEXT PRIMARY KEY, created_at REAL, key_id TEXT, idem TEXT, user_id TEXT, context TEXT,
              decision TEXT, shadow INTEGER, reasons TEXT, scores TEXT, sha256 TEXT, policy_version TEXT,
              response TEXT, review_status TEXT, human_decision TEXT, reviewer TEXT, notes TEXT, reviewed_at REAL);
            CREATE UNIQUE INDEX IF NOT EXISTS idem_idx ON decisions(key_id, idem) WHERE idem IS NOT NULL;
            CREATE INDEX IF NOT EXISTS review_idx ON decisions(review_status, created_at);""")

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            self.db.row_factory = sqlite3.Row
            return self.db.execute(sql, args).fetchall()


# ---------------------------------------------------------------- auth + rate limit
class RateLimiter:
    """Token bucket per JWT `sub`; burst = one minute of quota. In-memory (single process)."""

    def __init__(self, per_min: int):
        self.rate, self.buckets, self.lock = per_min / 60.0, {}, threading.Lock()

    def allow(self, sub: str, cost: float = 1) -> bool:
        with self.lock:
            tokens, last = self.buckets.get(sub, (self.rate * 60, time.monotonic()))
            now = time.monotonic()
            tokens = min(self.rate * 60, tokens + (now - last) * self.rate)
            ok = tokens >= cost
            self.buckets[sub] = (tokens - cost if ok else tokens, now)
            return ok


# ---------------------------------------------------------------- app state
@asynccontextmanager
async def lifespan(_):
    if len(JWT_SECRET) < 32:
        log.warning(json.dumps({"event": "weak_or_missing_jwt_secret", "length": len(JWT_SECRET),
                                "advice": "use >= 32 random chars, e.g. openssl rand -base64 48"}))
    threading.Thread(target=_load, daemon=True).start()  # /health answers (503) while models load
    yield


app = FastAPI(title="CTC Image Moderation", version="1.0.0", lifespan=lifespan,
              description="Decides APPROVE / REVIEW / REJECT for user-uploaded images. See INTEGRATION.md.")
S: dict = {"ready": False}
limiter = RateLimiter(RATE_PER_MIN)
store = Store(env("CTC_DB", "data/ctc.db"))
pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
slots = asyncio.Semaphore(CONCURRENCY)
waiting = 0


def _load():
    m = Moderator(env("CTC_CONFIG", "config.yaml"))
    m.warmup()
    S["mod"], S["ready"] = m, True
    log.info(json.dumps({"event": "ready", "models": m.models, "policy_version": m.policy_version,
                         "shadow": SHADOW, "kill_switch": KILL}))


BEARER = HTTPBearer(auto_error=False, description="JWT signed with CTC_JWT_SECRET (HS256). Swagger: Authorize.")


def auth(role: str = "client"):
    def dep(request: Request, cred: HTTPAuthorizationCredentials | None = Depends(BEARER)) -> dict:
        if not cred or not JWT_SECRET:
            print(JWT_SECRET)
            print(cred)
            raise ApiError(401, "UNAUTHORIZED", "Missing bearer token")
        token = cred.credentials.removeprefix("Bearer ").strip()  # tolerate "Bearer Bearer <jwt>" (Swagger paste)
        try:
            claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"require": ["exp"]})
        except jwt.ExpiredSignatureError:
            raise ApiError(401, "TOKEN_EXPIRED", "Token has expired")
        except jwt.InvalidTokenError:
            raise ApiError(401, "UNAUTHORIZED", "Invalid token")
        sub = str(claims.get("sub") or claims.get("id") or "")
        if not sub:
            raise ApiError(401, "UNAUTHORIZED", "Token needs a `sub` or `id` claim")
        if role == "admin" and str(claims.get("role", "")).lower() != "admin":
            raise ApiError(403, "FORBIDDEN", "This endpoint needs a token with role=admin")
        if not limiter.allow(sub):
            raise ApiError(429, "RATE_LIMITED", "Rate limit exceeded", {"Retry-After": "5"})
        request.state.key_id = sub
        return {"id": sub, "role": claims.get("role")}
    return dep


# ---------------------------------------------------------------- errors
def _err(status, code, message, headers=None):
    return JSONResponse({"error": {"code": code, "message": message}}, status, headers=headers)


@app.exception_handler(ApiError)
async def _api_err(_, e: ApiError):
    return _err(e.status, e.code, e.message, e.headers)


@app.exception_handler(ImageError)
async def _img_err(_, e: ImageError):
    return _err(e.status, e.code, e.message)


@app.exception_handler(RequestValidationError)
async def _val_err(_, e: RequestValidationError):
    return _err(422, "VALIDATION_ERROR", "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))


@app.exception_handler(StarletteHTTPException)
async def _http_err(_, e: StarletteHTTPException):
    return _err(e.status_code, {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR"), str(e.detail))


@app.middleware("http")
async def body_limit(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > 20_000_000 * (BATCH_MAX if request.url.path.endswith("/batch") else 1):
        return _err(413, "FILE_TOO_LARGE", "Request body too large")
    return await call_next(request)


# ---------------------------------------------------------------- core
async def _infer(data: bytes, context: str | None) -> tuple[Decision, bool]:
    global waiting
    if not S["ready"]:
        raise ApiError(503, "NOT_READY", "Models are still loading", {"Retry-After": "10"})
    if waiting >= QUEUE + CONCURRENCY:
        raise ApiError(503, "OVERLOADED", "Server busy, retry with backoff", {"Retry-After": "2"})
    waiting += 1
    try:
        await asyncio.wait_for(slots.acquire(), TIMEOUT_S)
    except TimeoutError:
        raise ApiError(503, "OVERLOADED", "Timed out waiting for a worker", {"Retry-After": "2"})
    finally:
        waiting -= 1
    try:
        return await asyncio.get_running_loop().run_in_executor(pool, S["mod"].moderate, data, context)
    finally:
        slots.release()


async def _moderate(data: bytes, k: dict, context, user_id, idem) -> ModerationResult:
    if idem:
        row = store.q("SELECT response FROM decisions WHERE key_id=? AND idem=?", (k["id"], idem))
        if row:
            return ModerationResult(**{**json.loads(row[0]["response"]), "cached": True})
    t0 = time.perf_counter()
    rid = uuid.uuid4().hex
    if KILL:  # skip models entirely; everything goes to humans
        sha = hashlib.sha256(data).hexdigest()
        d, cached = Decision("REVIEW", ["KILL_SWITCH"], {}, [], [], S["mod"].policy_version if S["ready"] else "", sha, ""), False
    else:
        d, cached = await _infer(data, context)
    real = d.decision
    res = ModerationResult(
        request_id=rid, decision="APPROVE" if SHADOW else real, enforced=not SHADOW,
        shadow_decision=real if SHADOW else None, reasons=d.reasons, scores=d.scores, regions=d.regions,
        models=d.models, policy_version=d.policy_version, image_sha256=d.image_sha256,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1), cached=cached)
    try:
        store.q("INSERT INTO decisions(request_id, created_at, key_id, idem, user_id, context, decision, shadow, reasons,"
                " scores, sha256, policy_version, response, review_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, time.time(), k["id"], idem, user_id, context, real, int(SHADOW), json.dumps(d.reasons),
                 json.dumps(d.scores), d.image_sha256, d.policy_version, res.model_dump_json(),
                 "pending" if real == "REVIEW" else None))
    except sqlite3.IntegrityError:  # concurrent duplicate idempotency key: return the stored one
        row = store.q("SELECT response FROM decisions WHERE key_id=? AND idem=?", (k["id"], idem))
        return ModerationResult(**{**json.loads(row[0]["response"]), "cached": True})
    log.info(json.dumps({"event": "moderated", "request_id": rid, "key_id": k["id"], "user_id": user_id,
                         "context": context, "decision": real, "enforced": not SHADOW, "reasons": d.reasons,
                         "sha256": d.image_sha256, "phash": d.phash, "latency_ms": res.latency_ms, "cached": cached}))
    return res


async def _read(file: UploadFile) -> bytes:
    limit = S["mod"].limits["max_bytes"] if S["ready"] else 15_000_000
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, "FILE_TOO_LARGE", f"File exceeds {limit} bytes")
    if not data:
        raise ApiError(400, "EMPTY_FILE", "Uploaded file is empty")
    return data


def _safe_ip(host: str) -> str:
    """Resolve and refuse anything that is not a public unicast address."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ApiError(400, "URL_NOT_ALLOWED", "Host does not resolve")
    ips = {i[4][0] for i in infos}
    for ip in ips:
        a = ipaddress.ip_address(ip)
        if not a.is_global or a.is_multicast:
            raise ApiError(400, "URL_NOT_ALLOWED", "URL resolves to a non-public address")
    return sorted(ips)[0]


async def _fetch(url: str) -> bytes:
    """SSRF-hardened fetch: http(s) on 80/443 only, public IPs only (pinned, so no DNS rebinding),
    max 2 redirects (each re-validated), size and time capped."""
    limit = S["mod"].limits["max_bytes"] if S["ready"] else 15_000_000
    async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False) as c:
        for _ in range(3):
            u = urlparse(url)
            if u.scheme not in ("http", "https") or not u.hostname or u.port not in (None, 80, 443) or u.username:
                raise ApiError(400, "URL_NOT_ALLOWED", "Only public http(s) URLs on ports 80/443")
            ip = _safe_ip(u.hostname)
            pinned = u._replace(netloc=(f"[{ip}]" if ":" in ip else ip)).geturl()
            async with c.stream("GET", pinned, headers={"Host": u.hostname},
                                extensions={"sni_hostname": u.hostname}) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code != 200:
                    raise ApiError(400, "URL_FETCH_FAILED", f"URL returned HTTP {r.status_code}")
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > limit:
                        raise ApiError(413, "FILE_TOO_LARGE", f"File exceeds {limit} bytes")
                return bytes(buf)
    raise ApiError(400, "URL_FETCH_FAILED", "Too many redirects")


Context = Literal["profile_photo", "chat", "post"]


# ---------------------------------------------------------------- routes
@app.post("/v1/moderate/image", response_model=ModerationResult, tags=["moderation"],
          responses={s: {"description": d} for s, d in {400: "Bad input", 401: "Bad or expired token", 413: "Too large",
                     415: "Unsupported format", 422: "Undecodable/invalid", 429: "Rate limited", 503: "Busy or loading"}.items()})
async def moderate_image(file: UploadFile | None = File(None, description="JPEG, PNG, GIF, WebP or HEIC/HEIF"),
                         image_url: str | None = Form(None, description="Only if enabled server-side (CTC_ALLOW_URL=1)"),
                         context: Context | None = Form(None), user_id: str | None = Form(None, max_length=128),
                         idempotency_key: str | None = Form(None, max_length=128), k: dict = Depends(auth())):
    if file is not None:
        data = await _read(file)
    elif image_url:
        if not ALLOW_URL:
            raise ApiError(400, "URL_INPUT_DISABLED", "image_url input is disabled; upload the file")
        data = await _fetch(image_url)
    else:
        raise ApiError(400, "NO_IMAGE", "Send a `file` field")
    return await _moderate(data, k, context, user_id, idempotency_key)


@app.post("/v1/moderate/image/batch", response_model=list[BatchItem], tags=["moderation"])
async def moderate_batch(files: list[UploadFile] = File(..., description=f"Up to {BATCH_MAX} images"),
                         context: Context | None = Form(None), user_id: str | None = Form(None, max_length=128),
                         k: dict = Depends(auth())):
    if len(files) > BATCH_MAX:
        raise ApiError(400, "BATCH_TOO_LARGE", f"At most {BATCH_MAX} files per batch")
    if not limiter.allow(k["id"], len(files) - 1):  # each image counts against the rate limit
        raise ApiError(429, "RATE_LIMITED", "Rate limit exceeded", {"Retry-After": "5"})

    async def one(f: UploadFile) -> BatchItem:
        try:
            return BatchItem(filename=f.filename, result=await _moderate(await _read(f), k, context, user_id, None))
        except (ApiError, ImageError) as e:
            return BatchItem(filename=f.filename, error={"code": e.code, "message": e.message})
    return list(await asyncio.gather(*(one(f) for f in files)))


@app.post("/v1/feedback", tags=["feedback"])
async def feedback(fb: Feedback, k: dict = Depends(auth("admin"))):
    """Moderator outcome for any decision (e.g. an appeal); used to retune thresholds. Admin only, so app users
    can't inject fake verdicts into the tuning data."""
    return _resolve(fb.request_id, fb.human_decision, fb.reviewer, fb.notes)


def _resolve(rid, decision, reviewer, notes):
    rows = store.q("SELECT review_status FROM decisions WHERE request_id=?", (rid,))
    if not rows:
        raise ApiError(404, "NOT_FOUND", "Unknown request_id")
    store.q("UPDATE decisions SET human_decision=?, reviewer=?, notes=?, reviewed_at=?, "
            "review_status=CASE WHEN review_status IS NULL THEN NULL ELSE 'resolved' END WHERE request_id=?",
            (decision, reviewer, notes, time.time(), rid))
    log.info(json.dumps({"event": "human_decision", "request_id": rid, "decision": decision, "reviewer": reviewer}))
    return {"request_id": rid, "human_decision": decision, "status": "recorded"}


def _item(r) -> ReviewItem:
    return ReviewItem(request_id=r["request_id"], created_at=r["created_at"], user_id=r["user_id"], context=r["context"],
                      reasons=json.loads(r["reasons"]), scores=json.loads(r["scores"]), image_sha256=r["sha256"],
                      shadow=bool(r["shadow"]), status=r["review_status"], human_decision=r["human_decision"],
                      reviewer=r["reviewer"], notes=r["notes"])


@app.get("/v1/review", response_model=list[ReviewItem], tags=["review queue (admin)"])
async def review_list(status: Literal["pending", "resolved"] = "pending", include_shadow: bool = False,
                      before: float | None = None, limit: int = 50, k: dict = Depends(auth("admin"))):
    """REVIEW items for the moderator dashboard, oldest first for pending. Images are NOT stored here:
    look the image up in your app's storage by request_id (save it when you call /moderate)."""
    rows = store.q("SELECT * FROM decisions WHERE review_status=? AND (?=1 OR shadow=0) AND (? IS NULL OR created_at<?) "
                   f"ORDER BY created_at {'ASC' if status == 'pending' else 'DESC'} LIMIT ?",
                   (status, int(include_shadow), before, before, min(max(limit, 1), 200)))
    return [_item(r) for r in rows]


@app.get("/v1/review/stats", tags=["review queue (admin)"])
async def review_stats(k: dict = Depends(auth("admin"))):
    rows = store.q("SELECT review_status s, human_decision h, COUNT(*) n, MIN(created_at) oldest FROM decisions "
                   "WHERE review_status IS NOT NULL AND shadow=0 GROUP BY 1, 2")
    pending = [r for r in rows if r["s"] == "pending"]
    return {"pending": sum(r["n"] for r in pending),
            "oldest_pending_age_s": round(time.time() - min(r["oldest"] for r in pending), 1) if pending else None,
            "resolved": {r["h"]: r["n"] for r in rows if r["s"] == "resolved"}}


@app.get("/v1/review/{request_id}", response_model=ReviewItem, tags=["review queue (admin)"])
async def review_get(request_id: str, k: dict = Depends(auth("admin"))):
    rows = store.q("SELECT * FROM decisions WHERE request_id=? AND review_status IS NOT NULL", (request_id,))
    if not rows:
        raise ApiError(404, "NOT_FOUND", "Unknown review item")
    return _item(rows[0])


@app.post("/v1/review/{request_id}", tags=["review queue (admin)"])
async def review_resolve(request_id: str, body: ReviewResolve, k: dict = Depends(auth("admin"))):
    """Moderator verdict. Your app must then publish (APPROVE) or delete (REJECT) the image itself."""
    if not store.q("SELECT 1 FROM decisions WHERE request_id=? AND review_status IS NOT NULL", (request_id,)):
        raise ApiError(404, "NOT_FOUND", "Unknown review item")
    return _resolve(request_id, body.decision, body.reviewer, body.notes)


@app.get("/health", tags=["ops"])
async def health():
    """200 once models are loaded and warmed up; 503 while starting. No auth (used by health checks)."""
    if not S["ready"]:
        return _err(503, "NOT_READY", "Models are still loading")
    return {"status": "ok", "policy_version": S["mod"].policy_version}


if __name__ == "__main__" and sys.argv[1:2] == ["token"]:
    if not JWT_SECRET:
        sys.exit("Set CTC_JWT_SECRET first")
    sub, role = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "client")
    hours = float(sys.argv[4]) if len(sys.argv) > 4 else 24
    print(jwt.encode({"sub": sub, "role": role, "exp": int(time.time() + hours * 3600)}, JWT_SECRET, algorithm="HS256"))
