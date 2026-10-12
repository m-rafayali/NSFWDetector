"""Load test (locust). Each request gets unique bytes so the SHA cache never hits.

    CTC_API_KEY=... CTC_LOAD_DIR=path/to/jpegs uv run locust -f loadtest.py --headless \
        -u 4 -r 4 -t 2m --host http://localhost:8000 --csv runs/load

Without CTC_LOAD_DIR it sends synthetic photos (these mostly clear stage 1, so they understate
stage-2 load: use a folder that mirrors your real upload mix for capacity numbers).
"""
import io
import os
import random
from pathlib import Path

import numpy as np
from locust import HttpUser, constant, task
from PIL import Image


def _jpegs():
    d = os.environ.get("CTC_LOAD_DIR")
    out = []
    if d:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                buf = io.BytesIO()
                Image.open(p).convert("RGB").save(buf, "JPEG", quality=90)
                out.append(buf.getvalue())
    if not out:
        rng = np.random.default_rng(0)
        for _ in range(20):  # smooth random "photos", ~1080x1350 like phone uploads
            small = rng.integers(0, 255, (12, 10, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(small).resize((1080, 1350), Image.BICUBIC).save(buf, "JPEG", quality=88)
            out.append(buf.getvalue())
    return out


IMAGES = _jpegs()


class Uploader(HttpUser):
    wait_time = constant(0)

    @task
    def moderate(self):
        data = random.choice(IMAGES) + os.urandom(16)  # bytes after JPEG EOI: decodes the same, new SHA
        self.client.post("/v1/moderate/image", files={"file": ("x.jpg", data)}, data={"context": "post"},
                         headers={"X-API-Key": os.environ["CTC_API_KEY"]}, name="/v1/moderate/image")
