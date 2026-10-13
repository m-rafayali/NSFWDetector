"""Evaluation harness for the production config: scoring, threshold tuning, CIs, robustness, regression gate.

    uv run python eval.py score [--data ../labeled]  # run the enabled models once per image -> runs/scores.jsonl
    uv run python eval.py robust                     # evasion suite with the production pipeline
    uv run python eval.py report [--final]           # CV-tuned thresholds + metrics -> EVAL_REPORT.md
    uv run python eval.py gate                       # exit 1 if eval_metrics.json regresses vs eval_baseline.json

Data layout: labeled/{safe,review,reject}/[edge_case/]*.jpg. Split is deterministic by SHA-256: 20% held-out
test (only touched with --final), 80% dev with 5 CV folds. Delete runs/scores.jsonl after changing models.
"""
import argparse
import hashlib
import io
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from ctc_mod import engine

RUNS = Path("runs")
SCORES = RUNS / "scores.jsonl"
CLASSES = ["safe", "review", "reject"]
PRED = ["APPROVE", "REVIEW", "REJECT"]
# COST[true][pred]. Owner policy: any exposed nudity is REJECT, so a missed explicit image costs 100 and
# explicit->REVIEW costs 3; a false reject of a safe photo costs 10, an unnecessary review of a safe photo 3
# (owner: clothed photos, sexy ones included, should auto-approve).
COST = {"safe": [0, 3, 10], "review": [5, 0, 2], "reject": [100, 3, 0]}
EXPOSED = ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"]
GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 0.98, 0.99, 2.0]  # 2.0 = rule disabled


# ---------------------------------------------------------------- scoring
def cmd_score(args):
    m = engine.Moderator()
    RUNS.mkdir(exist_ok=True)
    done = {json.loads(l)["sha"] for l in SCORES.open()} if SCORES.exists() else set()
    files = [p for p in sorted(Path(args.data).rglob("*")) if p.is_file() and not p.name.startswith(".")]
    with SCORES.open("a") as fh:
        for i, p in enumerate(files):
            rel = p.relative_to(args.data).parts
            data = p.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if rel[0] not in CLASSES or sha in done:
                continue
            done.add(sha)
            try:
                scores, _ = m.score(engine.decode(data, m.limits))  # every stage, so any cascade can be simulated
            except engine.ImageError as e:
                print(f"{p}: {e.code}", file=sys.stderr)
                continue
            fh.write(json.dumps({"path": str(p), "sha": sha, "label": rel[0], "edge": rel[1] if len(rel) > 2 else "",
                                 "split": "test" if int(sha[:8], 16) % 5 == 0 else "dev", "fold": int(sha[8:16], 16) % 5,
                                 "scores": {k: round(v, 5) for k, v in scores.items()}}) + "\n")
            print(f"[{i + 1}/{len(files)}] {p.name}", file=sys.stderr)


def load_records():
    if not SCORES.exists():
        sys.exit(f"No {SCORES}. Run `eval.py score` first.")
    recs = [json.loads(l) for l in SCORES.open()]
    for r in recs:  # strongest exposed-part score, for the per-signal table
        r["scores"]["nudenet.max_exposed"] = max(r["scores"].get(f"nudenet.{c}", 0.0) for c in EXPOSED)
    return recs


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    """95% Wilson interval: valid at 0 errors, unlike a percentile bootstrap (which gives [0, 0])."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def bootstrap(fn, n, b=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [v for v in (fn(rng.integers(0, n, n)) for _ in range(b)) if not math.isnan(v)]
    return np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"),) * 2


def auc(pos, neg):
    """ROC AUC via the Mann-Whitney statistic (ties averaged)."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = allv.argsort().argsort() + 1.0
    for v in np.unique(allv):
        ranks[allv == v] = ranks[allv == v].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fmt(t):
    return f"{t[0] * 100:.1f}% [{t[1] * 100:.1f}–{t[2] * 100:.1f}]" if not math.isnan(t[0]) else "n/a"


def cost(pred, y):
    return float(np.mean([COST[CLASSES[t]][p] for t, p in zip(y, pred)])) if len(y) else float("nan")


# ---------------------------------------------------------------- rule simulation (vectorised)
class RuleSim:
    """Exact vectorised replica of Moderator.decide + cascade (default context)."""

    def __init__(self, cfg, recs):
        self.rules = cfg["policy"]["rules"]
        stage = {n: m.get("stage", 1) for n, m in cfg["models"].items() if m.get("enabled")}
        clear = cfg.get("cascade", {}).get("clear_if_below", {})
        self.cleared = np.array([bool(clear) and all(r["scores"].get(k, 0) < v for k, v in clear.items())
                                 for r in recs]) if 2 in stage.values() else np.ones(len(recs), bool)
        self.S = np.full((len(recs), len(self.rules)), -1.0)
        for j, rule in enumerate(self.rules):
            model = rule["signal"].split(".")[0]
            if model in stage:  # disabled model -> rule inactive
                col = np.array([r["scores"].get(rule["signal"], -1.0) for r in recs])
                self.S[:, j] = np.where(self.cleared, -1.0, col) if stage[model] == 2 else col
        self.rej = np.array([r.get("reject", 2.0) for r in self.rules])
        # `unless` cancels a rule's REVIEW level: mask those cells below review but keep them able to REJECT.
        self.unless = np.zeros_like(self.S, bool)
        for j, rule in enumerate(self.rules):
            if "unless" in rule:
                u = rule["unless"]
                self.unless[:, j] = np.array([r["scores"].get(u["signal"], 0) >= u["above"] for r in recs])
        self.rev = np.array([r.get("review", 2.0) for r in self.rules])

    def predict(self, rej=None, rev=None):
        rej = self.rej if rej is None else rej
        rev = self.rev if rev is None else rev
        lv = np.where(self.S >= rej, 2, np.where(self.S >= rev, 1, 0))
        return np.where(self.unless & (lv == 1), 0, lv).max(axis=1)


def tune(sim, y, idx, passes=3):
    """Coordinate descent over each rule's review/reject thresholds, minimising mean cost on idx."""
    rej, rev = sim.rej.copy(), sim.rev.copy()
    best = cost(sim.predict(rej, rev)[idx], y[idx])
    for _ in range(passes):
        for j in range(len(rej)):
            for arr, other, is_rej in ((rej, rev, True), (rev, rej, False)):
                for g in GRID:
                    if (is_rej and g < other[j] and g != 2.0) or (not is_rej and g > other[j] and other[j] != 2.0):
                        continue  # keep review <= reject
                    old, arr[j] = arr[j], g
                    c = cost(sim.predict(rej, rev)[idx], y[idx])
                    if c < best - 1e-12:
                        best = c
                    else:
                        arr[j] = old
    return rej, rev


# ---------------------------------------------------------------- report
def summarize(pred, y, title):
    lines = [f"### {title}", "", f"n = {len(y)}  ({', '.join(f'{c}: {(y == i).sum()}' for i, c in enumerate(CLASSES))})",
             "", "| true \\ pred | APPROVE | REVIEW | REJECT |", "|---|---|---|---|"]
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | " + " | ".join(str(int(((y == i) & (pred == j)).sum())) for j in range(3)) + " |")
    rj, sf, rv = y == 2, y == 0, y == 1
    m = {
        "explicit_auto_approved": wilson(int((pred[rj] == 0).sum()), int(rj.sum())),
        "explicit_caught_auto": wilson(int((pred[rj] == 2).sum()), int(rj.sum())),
        "safe_auto_rejected": wilson(int((pred[sf] == 2).sum()), int(sf.sum())),
        "safe_sent_to_review": wilson(int((pred[sf] == 1).sum()), int(sf.sum())),
        "borderline_auto_approved": wilson(int((pred[rv] == 0).sum()), int(rv.sum())),
        "review_rate_this_mix": wilson(int((pred == 1).sum()), len(y)),
        "reject_precision": wilson(int(((pred == 2) & rj).sum()), int((pred == 2).sum())),
    }
    lo, hi = bootstrap(lambda i: cost(pred[i], y[i]), len(y))
    lines += ["", "| metric | value [95% CI] |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in m.items()]
    lines += [f"| mean cost / image | {cost(pred, y):.3f} [{lo:.3f}–{hi:.3f}] (bootstrap) |", ""]
    if rj.sum() < 300:
        lines.append(f"> **Too small:** {int(rj.sum())} explicit images. Showing ≤1% explicit auto-approval at 95% "
                     "confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.\n")
    return lines, {k: v[0] for k, v in m.items()} | {"cost": cost(pred, y)}


def cmd_report(args):
    cfg = yaml.safe_load(open(args.config))
    recs = load_records()
    y = np.array([CLASSES.index(r["label"]) for r in recs])
    dev = np.array([r["split"] == "dev" for r in recs])
    test, fold = ~dev, np.array([r["fold"] for r in recs])
    out = ["# Evaluation report", "", f"policy_version `{cfg['policy_version']}` · generated "
           f"{time.strftime('%Y-%m-%d %H:%M')} · {len(recs)} images ({dev.sum()} dev / {test.sum()} test)", "",
           "All numbers come from `eval.py` on our labeled set. Rates use 95% Wilson intervals; cost uses a 1000x "
           f"percentile bootstrap. Cost matrix (true→pred APPROVE/REVIEW/REJECT): {COST}.", ""]

    # 1) How well each signal separates explicit from safe images.
    out += ["## Signals (dev, explicit vs safe)", "", "| signal | ROC AUC [95% CI] | recall at 1% false positives |",
            "|---|---|---|"]
    for s in sorted({rule["signal"] for rule in cfg["policy"]["rules"]} | {"nudenet.max_exposed"}):
        v = np.array([r["scores"].get(s, np.nan) for r in recs])
        pos, neg = v[dev & (y == 2)], v[dev & (y == 0)]
        if len(pos) and len(neg):
            both, lab = np.r_[pos, neg], np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            lo, hi = bootstrap(lambda i: auc(both[i][lab[i] == 1], both[i][lab[i] == 0]), len(both), b=300)
            out.append(f"| {s} | {auc(pos, neg):.3f} [{lo:.3f}–{hi:.3f}] | {(pos > np.quantile(neg, 0.99)).mean():.1%} |")

    # 2) Thresholds: CV estimate of tuning (honest), then tune on all dev -> suggestion for config.yaml.
    sim = RuleSim(cfg, recs)
    cv_pred = np.zeros(len(recs), int)
    for k in range(5):
        tr, va = dev & (fold != k), dev & (fold == k)
        if tr.sum() and va.sum():
            cv_pred[va] = sim.predict(*tune(sim, y, np.where(tr)[0]))[va]
    rej, rev = tune(sim, y, np.where(dev)[0])
    out += ["", "## Decisions", ""] + summarize(sim.predict()[dev], y[dev], "Current config thresholds (dev)")[0]
    out += summarize(cv_pred[dev], y[dev], "Re-tuned thresholds, 5-fold cross-validated (dev)")[0]
    tuned = [{**r, "reject": float(a), "review": float(b)} for r, a, b in zip(sim.rules, rej, rev)]
    tuned = [{k: v for k, v in t.items() if not (k in ("reject", "review") and v >= 2.0)} for t in tuned]
    RUNS.mkdir(exist_ok=True)
    (RUNS / "tuned_rules.yaml").write_text(yaml.safe_dump({"rules": tuned}, sort_keys=False))
    out += ["Suggested thresholds (tuned on all dev) are in `runs/tuned_rules.yaml`. Adopt them only if the CV "
            "row beats the current row; copy into config.yaml and bump `policy_version`.", ""]

    # 3) Held-out test, only with --final, on the frozen config thresholds.
    if args.final and test.sum():
        pred = sim.predict()
        lines, metrics = summarize(pred[test], y[test], "HELD-OUT TEST (frozen config thresholds)")
        out += ["## Final test", ""] + lines
        Path("eval_metrics.json").write_text(json.dumps(metrics, indent=2))
        mask = test
    else:
        out += ["## Final test", "", "_Not run. Use `--final` once thresholds are frozen._", ""]
        pred, mask = cv_pred, dev

    # 4) Edge cases + every misclassified file.
    groups = defaultdict(list)
    for i in np.where(mask)[0]:
        if recs[i]["edge"]:
            groups[f"{recs[i]['label']}/{recs[i]['edge']}"].append(i)
    if groups:
        out += ["## Edge cases", "", "| class/edge | n | auto-approved | auto-rejected | review |", "|---|---|---|---|---|"]
        for g, idx in sorted(groups.items()):
            p = pred[idx]
            out.append(f"| {g}{' ⚠ n<50' if len(idx) < 50 else ''} | {len(idx)} | {fmt(wilson((p == 0).sum(), len(idx)))} | "
                       f"{fmt(wilson((p == 2).sum(), len(idx)))} | {fmt(wilson((p == 1).sum(), len(idx)))} |")
    used = {rule["signal"] for rule in cfg["policy"]["rules"]}
    out += ["", "## Misclassified files", "", "Severe = explicit approved or safe rejected.", "",
            "| severity | file | true | pred | top signals |", "|---|---|---|---|---|"]
    for i in np.where(mask & (pred != y))[0]:
        sev = "SEVERE" if (y[i] == 2 and pred[i] == 0) or (y[i] == 0 and pred[i] == 2) else "minor"
        top = sorted(((v, k) for k, v in recs[i]["scores"].items() if k in used), reverse=True)[:3]
        out.append(f"| {sev} | {recs[i]['path']} | {CLASSES[y[i]]} | {PRED[pred[i]]} | "
                   + ", ".join(f"{k}={v:.2f}" for v, k in top) + " |")
    if (RUNS / "robustness.md").exists():
        out += ["", "#" + (RUNS / "robustness.md").read_text()]
    Path("EVAL_REPORT.md").write_text("\n".join(out) + "\n")
    print("wrote EVAL_REPORT.md")


# ---------------------------------------------------------------- robustness
def _transforms(img, regions):
    w, h = img.size
    sticker = img.copy()
    for r in regions:  # cover detected exposed parts with emoji-like discs
        x, y, bw, bh = r["box"]
        ImageDraw.Draw(sticker).ellipse([x, y, x + bw, y + bh], fill=(255, 205, 0))
    meme = ImageOps.expand(img, border=(0, h // 5, 0, h // 5), fill="white")
    ImageDraw.Draw(meme).text((10, 10), "WHEN THE WEEKEND HITS " * 3, fill="black")
    screen = Image.new("RGB", (w, h), (25, 25, 25))
    screen.paste(img.resize((int(w * 0.6), int(h * 0.6))).filter(ImageFilter.GaussianBlur(1)), (int(w * 0.2), int(h * 0.2)))
    noise = np.random.default_rng(0).integers(-12, 12, (h, w, 3))
    screen = Image.fromarray(np.clip(np.asarray(screen, np.int16) + noise, 0, 255).astype(np.uint8))
    return {
        "crop_center_70": img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))),
        "rotate_90": img.rotate(90, expand=True), "rotate_15": img.rotate(15, expand=True),
        "mirror": ImageOps.mirror(img), "grayscale": ImageOps.grayscale(img).convert("RGB"),
        "blur_r4": img.filter(ImageFilter.GaussianBlur(4)), "jpeg_q10": img,
        "pad_border_40": ImageOps.expand(img, border=int(max(w, h) * 0.4), fill="black"),
        "stickers_on_regions": sticker, "meme_text": meme, "screenshot_of_screen": screen,
    }


def cmd_robust(args):
    m = engine.Moderator()
    recs = [r for r in load_records() if r["label"] in ("reject", "review") and r["split"] == "dev"][: args.limit]
    res = defaultdict(Counter)
    for r in recs:
        data = Path(r["path"]).read_bytes()
        base, _ = m.moderate(data)
        img = engine.decode(data, m.limits)[0]
        res["original"][base.decision] += 1
        for name, t in _transforms(img, [x for x in m.score([img])[1] if x["label"] in EXPOSED]).items():
            buf = io.BytesIO()
            t.save(buf, "JPEG", quality=10 if name == "jpeg_q10" else 90)
            d, _ = m.moderate(buf.getvalue())
            res[name][d.decision] += 1
            res[name]["_escaped"] += d.decision == "APPROVE" and base.decision != "APPROVE" and r["label"] == "reject"
    lines = ["# Robustness (dev reject+review images, production pipeline)", "", f"n = {len(recs)}", "",
             "| transform | APPROVE | REVIEW | REJECT | explicit escaped to APPROVE |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {c['APPROVE']} | {c['REVIEW']} | {c['REJECT']} | {c['_escaped']} |" for n, c in res.items()]
    RUNS.mkdir(exist_ok=True)
    (RUNS / "robustness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- regression gate
def cmd_gate(args):
    if not Path("eval_baseline.json").exists() or not Path("eval_metrics.json").exists():
        sys.exit(print("gate: no eval_baseline.json/eval_metrics.json yet, skipping") or 0)
    base, cur, tol = json.loads(Path("eval_baseline.json").read_text()), json.loads(Path("eval_metrics.json").read_text()), 0.005
    worse = {k: (base[k], cur[k]) for k in ("explicit_auto_approved", "safe_auto_rejected", "cost") if cur[k] > base[k] + tol}
    worse |= {k: (base[k], cur[k]) for k in ("explicit_caught_auto",) if cur[k] < base[k] - tol}
    for k, (b, c) in worse.items():
        print(f"REGRESSION {k}: baseline {b:.4f} -> now {c:.4f}")
    sys.exit(1 if worse else print("gate: OK"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score").add_argument("--data", default="../labeled")
    r = sub.add_parser("report")
    r.add_argument("--final", action="store_true", help="evaluate the held-out test split (do this once per release)")
    r.add_argument("--config", default="config.yaml", help="candidate config to evaluate")
    sub.add_parser("robust").add_argument("--limit", type=int, default=500)
    sub.add_parser("gate")
    a = ap.parse_args()
    {"score": cmd_score, "report": cmd_report, "robust": cmd_robust, "gate": cmd_gate}[a.cmd](a)
