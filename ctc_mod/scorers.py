"""Model adapters. Every model is a Scorer: RGB PIL image -> Result(scores, regions).

Scores are flat `"<model>.<signal>": float` so fusion rules in config.yaml can reference any of them.
Phase 2 (video) reuses these unchanged as frame scorers.
"""
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(os.environ.get("CTC_MODELS_DIR", "models"))


@dataclass
class Result:
    scores: dict[str, float]
    regions: list[dict] = field(default_factory=list)


class Scorer(Protocol):
    name: str
    version: str  # first 12 hex chars of the weights' sha256

    def score(self, img: Image.Image) -> Result: ...


def _session(path: Path) -> tuple[ort.InferenceSession, str]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("CTC_ORT_THREADS", "0"))  # 0 = all cores
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Classifier:
    """Whole-image classifier. Emits p(label) per label, plus `nsfw` = 1 - p(first/safe label)
    and, for ordinal label sets (e.g. neutral<low<medium<high), cumulative `<label>+` scores."""

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.sess, self.version = _session(MODELS_DIR / f"{cfg.get('file', name)}.onnx")
        self.input = self.sess.get_inputs()[0].name
        self.mean = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
        labels = cfg["labels"]
        # The safe label is not always first (Marqo is [nsfw, sfw]), so look it up by name.
        self.safe = next(l for l in labels if l in ("neutral", "normal", "sfw"))

    def preprocess(self, img: Image.Image) -> np.ndarray:
        s = self.cfg["size"]
        x = np.asarray(img.resize((s, s), Image.BICUBIC), np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    def score(self, img: Image.Image) -> Result:
        p = _softmax(self.sess.run(None, {self.input: self.preprocess(img)})[0][0])
        labels = self.cfg["labels"]
        out = {f"{self.name}.{l}": float(v) for l, v in zip(labels, p)}
        out[f"{self.name}.nsfw"] = 1.0 - out[f"{self.name}.{self.safe}"]
        if len(labels) > 2:
            for i, l in enumerate(labels[1:], 1):
                out[f"{self.name}.{l}+"] = float(p[i:].sum())
        return Result(out)


NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED",
    "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
]


class Detector:
    """NudeNet v3 (YOLOv8) region detector. Emits max score per body-part class + boxes."""

    def __init__(self, name: str, cfg: dict):
        self.name, self.size = name, cfg["size"]
        self.bgr = cfg.get("channels", "rgb") == "bgr"  # nudenet==3.4.2 effectively feeds BGR
        self.sess, self.version = _session(MODELS_DIR / f"{cfg.get('file', name)}.onnx")
        self.input = self.sess.get_inputs()[0].name

    def score(self, img: Image.Image) -> Result:
        # Pad bottom/right to a square (as NudeNet does), resize, scale to [0,1].
        # cv2's non-antialiased resize matches NudeNet's own preprocessing (PIL's shifted scores by up to 0.2).
        a = np.asarray(img)[..., ::-1] if self.bgr else np.asarray(img)
        side = max(a.shape[:2])
        a = cv2.copyMakeBorder(a, 0, side - a.shape[0], 0, side - a.shape[1], cv2.BORDER_CONSTANT)
        x = cv2.resize(a, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = (np.ascontiguousarray(x.transpose(2, 0, 1)) / 255.0)[None]
        pred = self.sess.run(None, {self.input: x})[0][0].T  # (anchors, 4 + classes)
        cls = pred[:, 4:]
        best = cls.max(axis=0)
        out = {f"{self.name}.{l}": float(s) for l, s in zip(NUDENET_LABELS, best)}
        # Boxes for explainability: per-anchor best class >= 0.25, greedy NMS (IoU 0.45).
        conf, ids = cls.max(axis=1), cls.argmax(axis=1)
        keep = np.where(conf >= 0.25)[0]
        keep = keep[np.argsort(-conf[keep])]
        boxes = pred[keep, :4] * (side / self.size)  # cx, cy, w, h in original pixels
        xyxy = np.c_[boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2]
        regions, taken = [], []
        for i, b in enumerate(xyxy):
            if all(_iou(b, xyxy[j]) < 0.45 for j in taken):
                taken.append(i)
                x1, y1 = np.maximum(b[:2], 0)
                x2, y2 = np.minimum(b[2:], img.size)
                regions.append({"label": NUDENET_LABELS[ids[keep[i]]], "score": round(float(conf[keep[i]]), 4),
                                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]})
        return Result(out, regions)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load(models_cfg: dict, only: list[str] | None = None) -> list[Scorer]:
    """Instantiate enabled models (or exactly `only`, for the bake-off)."""
    out = []
    for name, cfg in models_cfg.items():
        if (only is None and not cfg.get("enabled")) or (only is not None and name not in only):
            continue
        out.append((Detector if cfg["kind"] == "detector" else Classifier)(name, cfg))
    return out


# --- Hooks (interfaces only; no implementation in Phase 1) -------------------------------------
class AgeEstimator(Protocol):
    """Returns estimated probability that a person in the image is under 18.
    May ONLY add a REVIEW trigger (reason POSSIBLE_MINOR); never auto-approve or auto-reject."""

    def p_minor(self, img: Image.Image) -> float | None: ...


class HashMatcher(Protocol):
    """Matches image hashes against a vetted known-illegal-content provider (e.g. via an approved
    program). A match must block, preserve evidence per legal counsel's guidance, and alert a human."""

    def match(self, sha256: str, phash: str) -> bool: ...
