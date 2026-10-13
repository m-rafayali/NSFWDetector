import time

import jwt
import pytest
from fastapi.testclient import TestClient

from conftest import img_bytes

SECRET = "test-secret-0123456789abcdef0123456789"


def tok(sub="app", role="client", ttl=3600, secret=SECRET, alg="HS256"):
    return "Bearer " + jwt.encode({"sub": sub, "role": role, "exp": int(time.time() + ttl)}, secret, algorithm=alg)


CLIENT_KEY, ADMIN_KEY = tok(), tok("dash", "admin")


@pytest.fixture
def api(tmp_path, monkeypatch, mod):
    from ctc_mod import api as a
    monkeypatch.setattr(a, "JWT_SECRET", SECRET)
    monkeypatch.setattr(a, "limiter", a.RateLimiter(30))
    monkeypatch.setattr(a, "store", a.Store(str(tmp_path / "ctc.db")))
    a.S.update(mod=mod, ready=True)  # inject fake-model Moderator; lifespan not run
    return a


def post(c, data, key=CLIENT_KEY, **form):
    return c.post("/v1/moderate/image", files={"file": ("x.bin", data)}, data=form, headers={"Authorization": key})


def test_moderate_and_error_format(api):
    c = TestClient(api.app)
    r = post(c, img_bytes((255, 0, 0)), context="chat", user_id="u1")
    body = r.json()
    assert r.status_code == 200 and body["decision"] == "REJECT" and body["enforced"] is True
    assert body["policy_version"] == "test.1" and len(body["image_sha256"]) == 64 and body["regions"]
    assert post(c, img_bytes(), key="Bearer nope").json()["error"]["code"] == "UNAUTHORIZED"
    assert post(c, img_bytes(), key=tok(secret="wrong-secret-wrong-secret-wrong!!")).status_code == 401
    assert post(c, img_bytes(), key=tok(ttl=-10)).json()["error"]["code"] == "TOKEN_EXPIRED"
    assert post(c, img_bytes(), key=tok(secret=SECRET, alg="HS512")).status_code == 401  # HS256 only
    no_exp = "Bearer " + jwt.encode({"sub": "x"}, SECRET, algorithm="HS256")
    assert post(c, img_bytes(), key=no_exp).status_code == 401
    assert c.post("/v1/moderate/image", files={"file": ("x", img_bytes())}).status_code == 401
    r = post(c, b"GIF89a-not-really")
    assert r.status_code == 422 and r.json()["error"]["code"] == "CORRUPT_IMAGE"
    r = post(c, b"plain text")
    assert r.status_code == 415
    r = post(c, img_bytes(), context="bogus")
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert c.post("/v1/moderate/image", data={"image_url": "http://169.254.169.254/"},
                  headers={"Authorization": CLIENT_KEY}).json()["error"]["code"] == "URL_INPUT_DISABLED"


def test_idempotency_batch_and_rate_limit(api):
    c = TestClient(api.app)
    a = post(c, img_bytes(), idempotency_key="k1").json()
    b = post(c, img_bytes((1, 2, 3)), idempotency_key="k1").json()
    assert a["request_id"] == b["request_id"] and b["cached"]
    r = c.post("/v1/moderate/image/batch", headers={"Authorization": CLIENT_KEY},
               files=[("files", ("a", img_bytes())), ("files", ("b", b"junk"))]).json()
    assert r[0]["result"]["decision"] == "APPROVE" and r[1]["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    codes = [post(c, img_bytes()).status_code for _ in range(40)]
    assert 429 in codes  # rate_per_min: 30


def test_shadow_mode_and_kill_switch(api, monkeypatch):
    c = TestClient(api.app)
    monkeypatch.setattr(api, "SHADOW", True)
    body = post(c, img_bytes((255, 0, 0))).json()
    assert body["decision"] == "APPROVE" and body["enforced"] is False and body["shadow_decision"] == "REJECT"
    monkeypatch.setattr(api, "SHADOW", False)
    monkeypatch.setattr(api, "KILL", True)
    body = post(c, img_bytes((9, 9, 9))).json()
    assert body["decision"] == "REVIEW" and body["reasons"] == ["KILL_SWITCH"]


def test_review_queue_flow(api):
    c = TestClient(api.app)
    rid = post(c, img_bytes((200, 60, 60)), user_id="u7").json()["request_id"]  # borderline -> REVIEW
    adm = {"Authorization": ADMIN_KEY}
    assert c.get("/v1/review", headers={"Authorization": CLIENT_KEY}).status_code == 403
    items = c.get("/v1/review", headers=adm).json()
    assert [i["request_id"] for i in items] == [rid] and items[0]["user_id"] == "u7"
    assert c.get("/v1/review/stats", headers=adm).json()["pending"] == 1
    r = c.post(f"/v1/review/{rid}", json={"decision": "REJECT", "reviewer": "mod1"}, headers=adm)
    assert r.status_code == 200
    assert c.get("/v1/review", headers=adm).json() == []
    assert c.get(f"/v1/review/{rid}", headers=adm).json()["human_decision"] == "REJECT"
    assert c.post("/v1/feedback", json={"request_id": "nope", "human_decision": "APPROVE"},
                  headers={"Authorization": CLIENT_KEY}).status_code == 404


def test_health_and_removed_endpoints(api):
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok", "policy_version": "test.1"}
    for path in ("/healthz", "/readyz", "/metrics", "/v1/policy"):
        assert c.get(path).status_code == 404
    api.S["ready"] = False
    assert c.get("/health").status_code == 503
    assert post(c, img_bytes()).json()["error"]["code"] == "NOT_READY"


def test_ssrf_guard_blocks_private_addresses(api):
    for host in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "localhost", "::1"):
        with pytest.raises(api.ApiError):
            api._safe_ip(host)
