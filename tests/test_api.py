import hashlib

import pytest
from fastapi.testclient import TestClient

from conftest import img_bytes

CLIENT_KEY, ADMIN_KEY = "client-secret", "admin-secret"


@pytest.fixture
def api(tmp_path, monkeypatch, mod):
    keys = tmp_path / "keys.yaml"
    keys.write_text(f"""keys:
  - {{id: app, sha256: {hashlib.sha256(CLIENT_KEY.encode()).hexdigest()}, role: client, rate_per_min: 30}}
  - {{id: dash, sha256: {hashlib.sha256(ADMIN_KEY.encode()).hexdigest()}, role: admin}}
""")
    monkeypatch.setenv("CTC_DB", str(tmp_path / "ctc.db"))
    from ctc_mod import api as a
    monkeypatch.setattr(a, "keys", a.Keys(str(keys)))
    monkeypatch.setattr(a, "store", a.Store(str(tmp_path / "ctc.db")))
    a.S.update(mod=mod, ready=True)  # inject fake-model Moderator; lifespan not run
    return a


def post(c, data, key=CLIENT_KEY, **form):
    return c.post("/v1/moderate/image", files={"file": ("x.bin", data)}, data=form, headers={"X-API-Key": key})


def test_moderate_and_error_format(api):
    c = TestClient(api.app)
    r = post(c, img_bytes((255, 0, 0)), context="chat", user_id="u1")
    body = r.json()
    assert r.status_code == 200 and body["decision"] == "REJECT" and body["enforced"] is True
    assert body["policy_version"] == "test.1" and len(body["image_sha256"]) == 64 and body["regions"]
    assert post(c, img_bytes(), key="nope").json()["error"]["code"] == "UNAUTHORIZED"
    r = post(c, b"GIF89a-not-really")
    assert r.status_code == 422 and r.json()["error"]["code"] == "CORRUPT_IMAGE"
    r = post(c, b"plain text")
    assert r.status_code == 415
    r = post(c, img_bytes(), context="bogus")
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert c.post("/v1/moderate/image", data={"image_url": "http://169.254.169.254/"},
                  headers={"X-API-Key": CLIENT_KEY}).json()["error"]["code"] == "URL_INPUT_DISABLED"


def test_idempotency_batch_and_rate_limit(api):
    c = TestClient(api.app)
    a = post(c, img_bytes(), idempotency_key="k1").json()
    b = post(c, img_bytes((1, 2, 3)), idempotency_key="k1").json()
    assert a["request_id"] == b["request_id"] and b["cached"]
    r = c.post("/v1/moderate/image/batch", headers={"X-API-Key": CLIENT_KEY},
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
    adm = {"X-API-Key": ADMIN_KEY}
    assert c.get("/v1/review", headers={"X-API-Key": CLIENT_KEY}).status_code == 403
    items = c.get("/v1/review", headers=adm).json()
    assert [i["request_id"] for i in items] == [rid] and items[0]["user_id"] == "u7"
    assert c.get("/v1/review/stats", headers=adm).json()["pending"] == 1
    r = c.post(f"/v1/review/{rid}", json={"decision": "REJECT", "reviewer": "mod1"}, headers=adm)
    assert r.status_code == 200
    assert c.get("/v1/review", headers=adm).json() == []
    assert c.get(f"/v1/review/{rid}", headers=adm).json()["human_decision"] == "REJECT"
    assert c.post("/v1/feedback", json={"request_id": "nope", "human_decision": "APPROVE"},
                  headers={"X-API-Key": CLIENT_KEY}).status_code == 404


def test_ops_endpoints(api):
    c = TestClient(api.app)
    assert c.get("/healthz").json()["status"] == "ok"
    assert c.get("/readyz").json()["policy_version"] == "test.1"
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"X-API-Key": ADMIN_KEY}).status_code == 200
    assert c.get("/v1/policy", headers={"X-API-Key": CLIENT_KEY}).json()["policy_version"] == "test.1"
    api.S["ready"] = False
    assert c.get("/readyz").status_code == 503
    assert post(c, img_bytes()).json()["error"]["code"] == "NOT_READY"


def test_ssrf_guard_blocks_private_addresses(api):
    for host in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "localhost", "::1"):
        with pytest.raises(api.ApiError):
            api._safe_ip(host)
