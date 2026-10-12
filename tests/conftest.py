"""Shared fixtures. No real images or model weights: a fake scorer treats 'redness' as explicitness."""
import io

import numpy as np
import pytest
import yaml
from PIL import Image

from ctc_mod import engine, scorers

CONFIG = {
    "policy_version": "test.1",
    "models": {},
    "limits": {"max_bytes": 2_000_000, "max_pixels": 4_000_000, "max_frames": 50, "max_side": 512},
    "cache_size": 100,
    "cascade": {"clear_if_below": {"fake.nsfw": 0.2}},
    "policy": {
        "rules": [
            {"signal": "fake.nsfw", "reject": 0.8, "review": 0.4, "reason": "EXPLICIT_CONTENT"},
            {"signal": "heavy.nsfw", "reject": 0.9, "review": 0.5, "reason": "EXPLICIT_CONTENT"},
            {"signal": "fake.medium", "review": 0.5, "reason": "SUGGESTIVE_CONTENT"},
        ],
        "profiles": {"profile_photo": {"SUGGESTIVE_CONTENT": {"review": 0.3}}},
        "region_labels": ["FEMALE_BREAST_EXPOSED"],
    },
}


class Fake:
    """nsfw = mean red-dominance of the image; medium = mean green. Counts calls for cascade tests."""

    def __init__(self, name="fake"):
        self.name, self.version, self.calls = name, "test", 0

    def score(self, img):
        self.calls += 1
        a = np.asarray(img, np.float32) / 255
        red = float(np.clip(a[..., 0] - a[..., 1:].mean(-1), 0, 1).mean())
        green = float(np.clip(a[..., 1] - a[..., [0, 2]].mean(-1), 0, 1).mean())
        return scorers.Result({f"{self.name}.nsfw": red, f"{self.name}.medium": green},
                              [{"label": "FEMALE_BREAST_EXPOSED", "score": red, "box": [0, 0, 1, 1]}] if red > 0.4 else [])


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(CONFIG))
    return p


@pytest.fixture
def mod(config_path):
    m = engine.Moderator(config_path)
    m.scorers = [Fake(), Fake("heavy")]
    m.stage = {"fake": 1, "heavy": 2}
    return m


def img_bytes(color=(200, 200, 200), size=(64, 48), fmt="JPEG", mode="RGB", **kw):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, fmt, **kw)
    return buf.getvalue()
