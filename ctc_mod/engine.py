"""Image decoding/validation, cascade, fusion rules and decision cache. No web code here."""
import hashlib
import io
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pi_heif
import yaml
from PIL import Image, ImageOps

from . import scorers

pi_heif.register_heif_opener()
ORDER = {"APPROVE": 0, "REVIEW": 1, "REJECT": 2}
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}


class ImageError(Exception):
    """Bad input. `code` is a stable machine-readable error code, `status` the HTTP status."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def sniff(data: bytes) -> str | None:
    """Detect format from magic bytes (never trust extension or Content-Type)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in HEIF_BRANDS:
        return "heif"
    return None


def decode(data: bytes, limits: dict) -> list[Image.Image]:
    """Validate + decode untrusted bytes into RGB frames ready for inference.

    Returns >1 frame for animated images (first/middle/last) and for images with transparency
    (alpha-composited view + raw RGB view, since hidden pixels reappear if alpha is later dropped).
    Fusion takes the max over frames.
    """
    if len(data) > limits["max_bytes"]:
        raise ImageError("FILE_TOO_LARGE", f"File exceeds {limits['max_bytes']} bytes", 413)
    fmt = sniff(data)
    if fmt is None:
        raise ImageError("UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, GIF, WebP and HEIC/HEIF are accepted", 415)
    Image.MAX_IMAGE_PIXELS = limits["max_pixels"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if w * h > limits["max_pixels"] or min(w, h) < 1:
                raise ImageError("IMAGE_TOO_LARGE", f"Image exceeds {limits['max_pixels']} pixels", 413)
            n = getattr(img, "n_frames", 1)
            if n > limits["max_frames"]:
                raise ImageError("TOO_MANY_FRAMES", f"Animated image exceeds {limits['max_frames']} frames", 422)
            frames = []
            for i in sorted({0, n // 2, n - 1}):
                img.seek(i)
                frames += _prepare(img, limits["max_side"])
            return frames
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("IMAGE_TOO_LARGE", f"Image exceeds {limits['max_pixels']} pixels", 413)
    except Exception:
        raise ImageError("CORRUPT_IMAGE", "Image could not be decoded", 422)


def _prepare(img: Image.Image, max_side: int) -> list[Image.Image]:
    if img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))  # fast DCT-domain downscale for big photos
    img = ImageOps.exif_transpose(img)  # apply EXIF orientation; metadata is otherwise ignored
    if img.mode in ("I;16", "I", "F"):
        img = ImageOps.autocontrast(img.convert("L"))
    if img.mode == "P" and "transparency" in img.info or img.mode in ("RGBA", "LA", "PA", "La"):
        rgba = img.convert("RGBA")
        out = [Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba).convert("RGB")]
        if rgba.getextrema()[3][0] < 255:
            out.append(rgba.convert("RGB"))  # raw colour channels, alpha ignored
    else:
        out = [img.convert("RGB")]  # also handles CMYK, palette, grayscale
    for f in out:
        f.thumbnail((max_side, max_side), Image.BICUBIC)
    return out


def phash(img: Image.Image) -> str:
    """64-bit DCT perceptual hash as 16 hex chars."""
    a = np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), np.float64)
    k = np.arange(32)
    d = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / 64)  # DCT-II basis
    low = (d @ a @ d.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


@dataclass
class Decision:
    decision: str
    reasons: list[str]
    scores: dict[str, float]
    regions: list[dict]
    models: list[dict]
    policy_version: str
    image_sha256: str
    phash: str


class Moderator:
    """Loads models once; thread-safe `moderate()` (onnxruntime releases the GIL)."""

    def __init__(self, config_path: str | Path = "config.yaml", only: list[str] | None = None):
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        self.policy_version = self.cfg["policy_version"]
        self.limits = self.cfg["limits"]
        self.scorers = scorers.load(self.cfg["models"], only)
        self.stage = {s.name: self.cfg["models"][s.name].get("stage", 1) for s in self.scorers}
        self._cache: OrderedDict = OrderedDict()
        self._rejected = np.zeros(0, np.uint64)  # pHashes of rejected images
        self._lock = threading.Lock()

    @property
    def models(self) -> list[dict]:
        return [{"name": s.name, "version": s.version} for s in self.scorers]

    def warmup(self) -> None:
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 384, 3), np.uint8))
        for s in self.scorers:
            s.score(img)

    def score(self, frames: list[Image.Image], stage: int | None = None) -> tuple[dict, list]:
        """Run models (optionally just one cascade stage) on all frames; max per signal."""
        scores: dict[str, float] = {}
        regions: list[dict] = []
        for s in self.scorers:
            if stage is not None and self.stage[s.name] != stage:
                continue
            for f in frames:
                r = s.score(f)
                for k, v in r.scores.items():
                    scores[k] = max(v, scores.get(k, 0.0))
                regions += r.regions
        return scores, regions

    def decide(self, scores: dict, context: str | None = None) -> tuple[str, list[str]]:
        """Transparent rules: REJECT/REVIEW if a signal crosses its (profile-adjusted) threshold."""
        pol = self.cfg["policy"]
        overrides = pol.get("profiles", {}).get(context or "", {})
        decision, reasons = "APPROVE", []
        for rule in pol["rules"]:
            v = scores.get(rule["signal"])
            if v is None:  # model disabled -> rule inactive
                continue
            t = {**rule, **overrides.get(rule["reason"], {})}
            level = "REJECT" if v >= t.get("reject", 2) else "REVIEW" if v >= t.get("review", 2) else None
            if level:
                if rule["reason"] not in reasons:
                    reasons.append(rule["reason"])
                decision = max(decision, level, key=ORDER.get)
        return decision, reasons

    def needs_stage2(self, stage1_scores: dict, stage1_decision: str) -> bool:
        """Stage 2 runs unless stage 1 already rejects or ALL `clear_if_below` signals are low."""
        if stage1_decision == "REJECT" or not any(v == 2 for v in self.stage.values()):
            return False
        clear = self.cfg.get("cascade", {}).get("clear_if_below", {})
        return not (clear and all(stage1_scores.get(k, 0.0) < v for k, v in clear.items()))

    def moderate(self, data: bytes, context: str | None = None) -> tuple[Decision, bool]:
        """Returns (decision, cached). Raises ImageError on bad input.
        Model scores are cached by SHA-256; the (cheap) policy is re-applied per context."""
        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            entry = self._cache.get(sha)
            if entry:
                self._cache.move_to_end(sha)
        cached, frames = entry is not None, None
        if entry is None:
            frames = decode(data, self.limits)
            scores, regions = self.score(frames, stage=1)
            entry = {"scores": scores, "regions": regions, "phash": phash(frames[0]), "stage2": False}
        decision, reasons = self.decide(entry["scores"], context)
        if not entry["stage2"] and self.needs_stage2(entry["scores"], decision):
            frames = frames or decode(data, self.limits)
            s2, r2 = self.score(frames, stage=2)
            entry = {**entry, "scores": entry["scores"] | s2, "regions": entry["regions"] + r2, "stage2": True}
            decision, reasons = self.decide(entry["scores"], context)
            cached = False
        if decision == "APPROVE" and self._near_rejected(entry["phash"]):
            decision, reasons = "REVIEW", ["NEAR_DUPLICATE_OF_REJECTED"]
        regions = [r for r in entry["regions"] if r["label"] in self.cfg["policy"].get("region_labels", [])]
        d = Decision(decision, reasons, {k: round(v, 4) for k, v in sorted(entry["scores"].items())}, regions,
                     self.models, self.policy_version, sha, entry["phash"])
        with self._lock:
            self._cache[sha] = entry
            if len(self._cache) > self.cfg.get("cache_size", 10000):
                self._cache.popitem(last=False)
            if decision == "REJECT" and not cached:
                self._rejected = np.append(self._rejected, np.uint64(int(entry["phash"], 16)))[-50000:]
        return d, cached

    def _near_rejected(self, ph: str, max_dist: int = 4) -> bool:
        """pHash is only used to escalate near-duplicates of rejected images, never to approve."""
        with self._lock:
            arr = self._rejected
        if not len(arr):
            return False
        x = np.bitwise_xor(arr, np.uint64(int(ph, 16)))
        return bool((np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1) <= max_dist).any())

