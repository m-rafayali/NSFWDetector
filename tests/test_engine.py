import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from conftest import CONFIG, img_bytes
from ctc_mod import engine

LIM = CONFIG["limits"]
RED, GREY, PINKISH, GREEN = (255, 0, 0), (128, 128, 128), (200, 60, 60), (100, 200, 100)


def test_sniff_by_magic_not_extension():
    assert engine.sniff(img_bytes(fmt="PNG")) == "png"
    assert engine.sniff(img_bytes(fmt="WEBP")) == "webp"
    assert engine.sniff(b"\x00\x00\x00\x18ftypheic....") == "heif"
    assert engine.sniff(b"<html>not an image") is None
    with pytest.raises(engine.ImageError) as e:
        engine.decode(b"%PDF-1.4 fake", LIM)
    assert e.value.status == 415


def test_size_pixel_and_corrupt_limits():
    with pytest.raises(engine.ImageError) as e:
        engine.decode(b"\xff\xd8\xff" + b"0" * (LIM["max_bytes"] + 1), LIM)
    assert e.value.code == "FILE_TOO_LARGE"
    with pytest.raises(engine.ImageError) as e:  # decompression bomb: tiny file, huge canvas
        engine.decode(img_bytes(size=(3000, 2000), fmt="PNG", mode="L", color=0), LIM)
    assert e.value.code == "IMAGE_TOO_LARGE"
    with pytest.raises(engine.ImageError) as e:
        engine.decode(img_bytes()[:40], LIM)
    assert e.value.code == "CORRUPT_IMAGE"


def test_modes_exif_and_downscale():
    for mode, fmt in [("CMYK", "JPEG"), ("L", "PNG"), ("P", "PNG"), ("I;16", "PNG"), ("LA", "PNG")]:
        frames = engine.decode(img_bytes(mode=mode, fmt=fmt, color=0 if mode != "LA" else (0, 255)), LIM)
        assert all(f.mode == "RGB" for f in frames)
    big = engine.decode(img_bytes(size=(1600, 400)), LIM)[0]
    assert max(big.size) == LIM["max_side"]
    im = Image.new("RGB", (60, 20))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert engine.decode(buf.getvalue(), LIM)[0].size == (20, 60)


def test_alpha_adds_raw_rgb_view_and_animation_samples_three_frames():
    hidden = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # red pixels, fully transparent
    buf = io.BytesIO()
    hidden.save(buf, "PNG")
    frames = engine.decode(buf.getvalue(), LIM)
    assert len(frames) == 2 and frames[1].getpixel((0, 0)) == (255, 0, 0)
    gif = [Image.new("RGB", (16, 16), c) for c in (GREY, GREY, RED, GREY, (0, 0, 255))]
    buf = io.BytesIO()
    gif[0].save(buf, "GIF", save_all=True, append_images=gif[1:])
    assert len(engine.decode(buf.getvalue(), LIM)) == 3


def test_rules_cascade_and_profiles(mod):
    d, cached = mod.moderate(img_bytes(GREY))
    assert d.decision == "APPROVE" and not cached and "heavy.nsfw" not in d.scores  # stage 2 skipped
    assert mod.moderate(img_bytes(GREY))[1] is True  # exact-hash cache
    d, _ = mod.moderate(img_bytes(RED))
    assert d.decision == "REJECT" and d.reasons == ["EXPLICIT_CONTENT"] and d.regions
    assert set(d.scores) <= {"fake.nsfw", "heavy.nsfw", "fake.medium"}  # only signals the policy reads
    assert "heavy.nsfw" not in d.scores  # stage 1 rejected, stage 2 skipped
    d, _ = mod.moderate(img_bytes(PINKISH))
    assert "heavy.nsfw" in d.scores and d.decision == "REVIEW"  # borderline -> stage 2 ran
    g = img_bytes(GREEN)
    assert mod.moderate(g)[0].decision == "APPROVE"
    assert mod.moderate(g, "profile_photo")[0].reasons == ["SUGGESTIVE_CONTENT"]  # stricter profile


def test_unless_cancels_review_but_not_reject(mod):
    rule = mod.cfg["policy"]["rules"][0]  # fake.nsfw: reject 0.8, review 0.4
    rule["unless"] = {"signal": "fake.medium", "above": 0.1}
    assert mod.decide({"fake.nsfw": 0.5, "fake.medium": 0.2}) == ("APPROVE", [])
    assert mod.decide({"fake.nsfw": 0.5, "fake.medium": 0.0}) == ("REVIEW", ["EXPLICIT_CONTENT"])
    assert mod.decide({"fake.nsfw": 0.9, "fake.medium": 0.2}) == ("REJECT", ["EXPLICIT_CONTENT"])


def test_disabled_model_rules_are_skipped(mod):
    mod.scorers = mod.scorers[:1]
    mod.stage = {"fake": 1}
    d, _ = mod.moderate(img_bytes(PINKISH))
    assert "heavy.nsfw" not in d.scores and d.decision == "REVIEW"


def test_near_duplicate_of_rejected_is_escalated_never_approved(mod):
    rng = np.random.default_rng(0)
    blocks = rng.integers(170, 256, (8, 8, 1)) * np.array([1, 0, 0])  # blocky red texture
    a = Image.fromarray(blocks.astype(np.uint8)).resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    a.save(buf, "PNG")
    assert mod.moderate(buf.getvalue())[0].decision == "REJECT"
    mod.scorers = []  # now models would approve anything
    buf = io.BytesIO()
    a.save(buf, "JPEG", quality=70)  # re-encoded copy: different SHA, same pHash
    d, _ = mod.moderate(buf.getvalue())
    assert d.decision == "REVIEW" and d.reasons == ["NEAR_DUPLICATE_OF_REJECTED"]


def test_eval_stats_helpers():
    import eval as ev
    p, lo, hi = ev.wilson(0, 300)
    assert p == 0 and lo == 0 and 0.009 < hi < 0.0135  # rule of three ~1%
    assert ev.auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0


def test_quality_flags_unreadable_and_qr_codes():
    import cv2
    from ctc_mod.scorers import Quality
    q = Quality("quality", {"min_side": 100, "min_brightness": 20, "min_sharpness": 15})
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (300, 300, 3), np.uint8))
    assert q.score(photo).scores == {"quality.unreadable": 0.0, "quality.qr_code": 0.0}
    assert q.score(photo.resize((60, 60))).scores["quality.unreadable"] == 1.0          # too small
    assert q.score(Image.new("RGB", (300, 300), (5, 5, 5))).scores["quality.unreadable"] == 1.0  # dark/flat
    code = cv2.QRCodeEncoder.create().encode("https://example.com/dm-me")
    qr = Image.fromarray(cv2.resize(code, (300, 300), interpolation=cv2.INTER_NEAREST)).convert("RGB")
    assert q.score(ImageOps.expand(qr, 40, "white")).scores["quality.qr_code"] == 1.0
