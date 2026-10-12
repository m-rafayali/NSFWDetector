"""Typed client example for app backends (copy into your codebase; only needs httpx).

    client = ModerationClient("https://moderation.example.com", api_key=os.environ["CTC_API_KEY"])
    result = client.moderate(open("photo.jpg", "rb").read(), context="profile_photo",
                             user_id="u_123", idempotency_key=upload_id)
    if result.decision == "APPROVE": publish(...)
"""
import random
import time
from dataclasses import dataclass
from typing import Literal

import httpx

Decision = Literal["APPROVE", "REVIEW", "REJECT"]


@dataclass
class Result:
    request_id: str
    decision: Decision          # what to enforce (always APPROVE in shadow mode)
    enforced: bool
    shadow_decision: Decision | None
    reasons: list[str]
    policy_version: str
    image_sha256: str


class ModerationError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code = status, code


class ModerationClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 15.0, retries: int = 3):
        self.http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=timeout)
        self.retries = retries

    def moderate(self, image: bytes, context: Literal["profile_photo", "chat", "post"] | None = None,
                 user_id: str | None = None, idempotency_key: str | None = None) -> Result:
        """Retries 429/503/network errors with jittered backoff. Always pass idempotency_key so a
        retry after a timeout returns the original decision instead of a second one."""
        form = {k: v for k, v in {"context": context, "user_id": user_id, "idempotency_key": idempotency_key}.items() if v}
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post("/v1/moderate/image", files={"file": ("upload", image)}, data=form)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt + random.random())
                continue
            if r.status_code in (429, 503) and attempt < self.retries:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)) + random.random())
                continue
            if r.status_code != 200:
                err = r.json().get("error", {})
                raise ModerationError(r.status_code, err.get("code", "UNKNOWN"), err.get("message", r.text))
            j = r.json()
            return Result(j["request_id"], j["decision"], j["enforced"], j.get("shadow_decision"), j["reasons"],
                          j["policy_version"], j["image_sha256"])
        raise AssertionError("unreachable")

    def feedback(self, request_id: str, human_decision: Literal["APPROVE", "REJECT"], notes: str | None = None) -> None:
        self.http.post("/v1/feedback", json={"request_id": request_id, "human_decision": human_decision,
                                             "notes": notes}).raise_for_status()
