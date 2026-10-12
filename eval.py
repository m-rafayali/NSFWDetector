"""Evaluation harness: bake-off, threshold tuning, CIs, robustness, regression gate.

    uv run python eval.py score  [--data labeled]   # run every model variant once, cached in runs/scores.jsonl
    uv run python eval.py report [--final]           # tune on dev (5-fold CV), write EVAL_REPORT.md
    uv run python eval.py robust                     # evasion suite with the production pipeline
    uv run python eval.py gate                       # exit 1 if eval_metrics.json regresses vs eval_baseline.json

Data layout: labeled/{safe,review,reject}/[edge_case/]*.jpg. Split is deterministic by SHA-256:
20% held-out test (only touched with --final), 80% dev with 5 CV folds.
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

from ctc_mod import engine, scorers

RUNS = Path("runs")
CLASSES = ["safe", "review", "reject"]
PRED = ["APPROVE", "REVIEW", "REJECT"]
# COST[true][pred]: a missed explicit image >> unnecessary review; false reject in between.
COST = {"safe": [0, 1, 10], "review": [5, 0, 2], "reject": [100, 1, 0]}
# Bake-off variants: name -> (config model, overrides). Score keys are "<variant>.<signal>".
VARIANTS = {
    "nudenet": ("nudenet", {}),
    "nudenet-rgb": ("nudenet", {"channels": "rgb"}),
    "nudenet-640m": ("nudenet", {"file": "nudenet-640m", "size": 640}),
    "freepik": ("freepik", {}),
    "freepik-int8": ("freepik", {"file": "freepik-int8"}),
    "falconsai": ("falconsai", {}),
    "falconsai-int8": ("falconsai", {"file": "falconsai-int8"}),
    "adamcodd": ("adamcodd", {}),
    "marqo": ("marqo", {}),
}
EXPOSED = ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"]
GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 0.98, 0.99, 2.0]  # 2.0 = rule disabled


# ---------------------------------------------------------------- scoring
def cmd_score(args):
    cfg = yaml.safe_load(open("config.yaml"))
    RUNS.mkdir(exist_ok=True)
    out = RUNS / "scores.jsonl"
    done = {json.loads(l)["sha"] for l in out.open()} if out.exists() else set()
    models = []
    for v, (base, over) in VARIANTS.items():
        try:
            c = {**cfg["models"][base], **over}
            c.setdefault("file", base)
            models.append((scorers.Detector if c["kind"] == "detector" else scorers.Classifier)(v, c))
        except Exception as e:
            print(f"skip {v}: {e}", file=sys.stderr)
    files = [p for p in sorted(Path(args.data).rglob("*")) if p.is_file() and not p.name.startswith(".")]
    with out.open("a") as fh:
        for i, p in enumerate(files):
            rel = p.relative_to(args.data).parts
            if rel[0] not in CLASSES:
                continue
            data = p.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha in done:
                continue
            done.add(sha)
            try:
                frames = engine.decode(data, cfg["limits"])
            except engine.ImageError as e:
                print(f"{p}: {e.code}", file=sys.stderr)
                continue
            scores, ms = {}, {}
            for m in models:
                t = time.perf_counter()
                for f in frames:
                    for k, v in m.score(f).scores.items():
                        scores[k] = max(v, scores.get(k, 0.0))
                ms[m.name] = round((time.perf_counter() - t) * 1000, 1)
            rec = {"path": str(p), "sha": sha, "label": rel[0], "edge": rel[1] if len(rel) > 2 else "",
                   "split": "test" if int(sha[:8], 16) % 5 == 0 else "dev", "fold": int(sha[8:16], 16) % 5,
                   "scores": {k: round(v, 5) for k, v in scores.items()}, "ms": ms}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(files)}] {p.name}", file=sys.stderr)


def load_records():
    p = RUNS / "scores.jsonl"
    if not p.exists():
        sys.exit("No runs/scores.jsonl. Run `eval.py score` first.")
    recs = [json.loads(l) for l in p.open()]
    for r in recs:  # derived signal: strongest exposed-part score per detector variant
        for v in [k for k in VARIANTS if k.startswith("nudenet")]:
            vals = [r["scores"].get(f"{v}.{c}") for c in EXPOSED]
            if None not in vals:
                r["scores"][f"{v}.max_exposed"] = max(vals)
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
    vals = [fn(rng.integers(0, n, n)) for _ in range(b)]
    vals = [v for v in vals if not math.isnan(v)]
    return np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"),) * 2


def auc(pos, neg):
    """ROC AUC via the Mann-Whitney statistic."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = allv.argsort().argsort() + 1.0
    for v in np.unique(allv):  # average tied ranks
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fmt(t):
    return f"{t[0] * 100:.1f}% [{t[1] * 100:.1f}–{t[2] * 100:.1f}]" if not math.isnan(t[0]) else "n/a"


# ---------------------------------------------------------------- rule simulation (vectorised)
class RuleSim:
    """Exact vectorised replica of Moderator.decide + cascade for the default context."""

    def __init__(self, cfg, recs):
        self.rules = cfg["policy"]["rules"]
        stage = {n: m.get("stage", 1) for n, m in cfg["models"].items() if m.get("enabled")}
        n = len(recs)
        self.S = np.full((n, len(self.rules)), -1.0)
        clear = cfg.get("cascade", {}).get("clear_if_below", {})
        s2 = any(v == 2 for v in stage.values())
        self.cleared = np.array([bool(clear) and all(r["scores"].get(k, 0) < v for k, v in clear.items())
                                 for r in recs]) if s2 else np.ones(n, bool)
        for j, rule in enumerate(self.rules):
            model = rule["signal"].split(".")[0]
            if model not in stage:
                continue  # disabled model -> rule inactive
            col = np.array([r["scores"].get(rule["signal"], -1.0) for r in recs])
            if stage[model] == 2:
                col = np.where(self.cleared, -1.0, col)  # stage 2 never ran for these
            self.S[:, j] = col
        self.rej = np.array([r.get("reject", 2.0) for r in self.rules])
        self.rev = np.array([r.get("review", 2.0) for r in self.rules])

    def predict(self, rej=None, rev=None):
        rej = self.rej if rej is None else rej
        rev = self.rev if rev is None else rev
        lv = np.where(self.S >= rej, 2, np.where(self.S >= rev, 1, 0))
        return lv.max(axis=1)


def cost(pred, y):
    return float(np.mean([COST[CLASSES[t]][p] for t, p in zip(y, pred)])) if len(y) else float("nan")


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
                    old = arr[j]
                    arr[j] = g
                    c = cost(sim.predict(rej, rev)[idx], y[idx])
                    if c < best - 1e-12:
                        best = c
                    else:
                        arr[j] = old
    return rej, rev


# ---------------------------------------------------------------- report
def summarize(pred, y, title):
    lines = [f"### {title}", "", f"n = {len(y)}  ({', '.join(f'{c}: {(y == i).sum()}' for i, c in enumerate(CLASSES))})", "",
             "| true \\ pred | APPROVE | REVIEW | REJECT |", "|---|---|---|---|"]
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
    n = len(y)
    lo, hi = bootstrap(lambda i: cost(pred[i], y[i]), n)
    lines += ["", "| metric | value [95% CI] |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in m.items()]
    lines += [f"| mean cost / image | {cost(pred, y):.3f} [{lo:.3f}–{hi:.3f}] (bootstrap) |", ""]
    n_rej = int(rj.sum())
    if n_rej < 300:
        lines.append(f"> **Too small:** {n_rej} explicit images. Showing ≤1% explicit auto-approval at 95% "
                     f"confidence needs ≥300 with zero misses. The upper CI bound above is the honest number.\n")
    return lines, {k: v[0] for k, v in m.items()} | {"cost": cost(pred, y)}


def cmd_report(args):
    cfg = yaml.safe_load(open("config.yaml"))
    recs = load_records()
    y = np.array([CLASSES.index(r["label"]) for r in recs])
    dev = np.array([r["split"] == "dev" for r in recs])
    test = ~dev
    fold = np.array([r["fold"] for r in recs])
    out = [f"# Evaluation report", "", f"policy_version `{cfg['policy_version']}` · generated {time.strftime('%Y-%m-%d %H:%M')} · "
           f"{len(recs)} images ({dev.sum()} dev / {test.sum()} test)", "",
           "All numbers come from `eval.py` on our labeled set. Rates use 95% Wilson intervals; cost uses a "
           f"1000x percentile bootstrap. Cost matrix (true→pred A/RV/RJ): {COST}.", ""]
    RUNS.mkdir(exist_ok=True)

    # 1) Per-model bake-off: separability of explicit vs safe, plus latency.
    out += ["## Per-model bake-off (dev split, reject vs safe)", "",
            "| signal | ROC AUC [95% CI] | recall@1% FPR | p50 ms | p95 ms |", "|---|---|---|---|---|"]
    sigs = sorted({k for r in recs for k in r["scores"] if k.endswith((".nsfw", ".high", ".medium+", ".max_exposed"))})
    curves = {}
    for s in sigs:
        v = np.array([r["scores"].get(s, np.nan) for r in recs])
        pos, neg = v[dev & (y == 2)], v[dev & (y == 0)]
        if not len(pos) or not len(neg):
            continue
        a = auc(pos, neg)
        both = np.r_[pos, neg]
        lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        lo, hi = bootstrap(lambda i: auc(both[i][lab[i] == 1], both[i][lab[i] == 0]), len(both), b=300)
        t1 = np.quantile(neg, 0.99)
        lat = [r["ms"].get(s.split(".")[0]) for r in recs if s.split(".")[0] in r["ms"]]
        out.append(f"| {s} | {a:.3f} [{lo:.3f}–{hi:.3f}] | {(pos > t1).mean() * 100:.1f}% | "
                   f"{np.percentile(lat, 50):.0f} | {np.percentile(lat, 95):.0f} |")
        curves[s] = (pos, neg)
    out += ["", "Latency = per-model time on this machine incl. preprocessing, single process.", ""]
    _plots(curves)

    # 2) Rules: tune on dev with 5-fold CV (honest estimate), then tune on all dev for the final thresholds.
    sim = RuleSim(cfg, recs)
    cv_pred = np.zeros(len(recs), int)
    for k in range(5):
        tr, va = dev & (fold != k), dev & (fold == k)
        if tr.sum() and va.sum():
            rj, rv = tune(sim, y, np.where(tr)[0])
            cv_pred[va] = sim.predict(rj, rv)[va]
    rej, rev = tune(sim, y, np.where(dev)[0])
    lines, _ = summarize(sim.predict()[dev], y[dev], "Current config thresholds (dev, untuned)")
    out += ["## Ensemble decisions", ""] + lines
    lines, cvm = summarize(cv_pred[dev], y[dev], "Tuned rules, 5-fold cross-validated (dev)")
    out += lines
    stack = _stacker_cv(recs, y, dev, fold, cfg)
    if stack is not None:
        lines, sm = summarize(stack[dev], y[dev], "Logistic-regression stacker, 5-fold CV (dev)")
        verdict = ("**beats**" if sm["cost"] < cvm["cost"] else "does **not** beat")
        out += lines + [f"Stacker {verdict} tuned rules on CV cost ({sm['cost']:.3f} vs {cvm['cost']:.3f}). "
                        "Only adopt it if it wins clearly (outside bootstrap noise).", ""]

    tuned = [{**r, "reject": float(a), "review": float(b)} for r, a, b in zip(sim.rules, rej, rev)]
    for t in tuned:
        for k in ("reject", "review"):
            if t[k] >= 2.0:
                t.pop(k)
    (RUNS / "tuned_rules.yaml").write_text(yaml.safe_dump({"rules": tuned}, sort_keys=False))
    out += ["Suggested thresholds (tuned on all dev) written to `runs/tuned_rules.yaml`. Copy into config.yaml "
            "and bump `policy_version` by hand after review.", ""]

    # 3) Final held-out test: only with --final, using the tuned thresholds.
    metrics = None
    if args.final and test.sum():
        tp = sim.predict(rej, rev)
        lines, metrics = summarize(tp[test], y[test], "HELD-OUT TEST (tuned thresholds, touched once)")
        out += ["## Final test", ""] + lines
        Path("eval_metrics.json").write_text(json.dumps(metrics, indent=2))
        eval_pred, eval_mask = tp, test
    else:
        out += ["## Final test", "", "_Not run. Use `--final` once thresholds are frozen._", ""]
        eval_pred, eval_mask = cv_pred, dev

    # 4) Edge cases + every misclassified file.
    out += ["## Edge cases", "", "| class/edge | n | auto-approved | auto-rejected | review |", "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        if eval_mask[i] and r["edge"]:
            groups[f"{r['label']}/{r['edge']}"].append(i)
    for g, idx in sorted(groups.items()):
        p = eval_pred[idx]
        flag = " ⚠ n<50" if len(idx) < 50 else ""
        out.append(f"| {g}{flag} | {len(idx)} | {fmt(wilson((p == 0).sum(), len(idx)))} | "
                   f"{fmt(wilson((p == 2).sum(), len(idx)))} | {fmt(wilson((p == 1).sum(), len(idx)))} |")
    out += ["", "## Calibration (dev): predicted vs observed explicit rate", ""]
    out += _calibration(recs, y, dev)
    out += ["## Misclassified files", "", "Severe = explicit approved or safe rejected.", "",
            "| severity | file | true | pred | top signals |", "|---|---|---|---|---|"]
    for i in np.where(eval_mask)[0]:
        t, p = y[i], eval_pred[i]
        if (t == 2 and p != 2) or (t == 0 and p != 0) or (t == 1 and p != 1):
            sev = "SEVERE" if (t == 2 and p == 0) or (t == 0 and p == 2) else "minor"
            top = sorted(((v, k) for k, v in recs[i]["scores"].items() if k.split(".")[0] in cfg["models"]), reverse=True)[:3]
            out.append(f"| {sev} | {recs[i]['path']} | {CLASSES[t]} | {PRED[p]} | "
                       + ", ".join(f"{k}={v:.2f}" for v, k in top) + " |")
    Path("EVAL_REPORT.md").write_text("\n".join(out) + "\n")
    print("wrote EVAL_REPORT.md")


def _stacker_cv(recs, y, dev, fold, cfg):
    """Multinomial LR on production signals; decision = argmin expected cost. Returns CV predictions."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    keys = sorted({r_["signal"] for r_ in cfg["policy"]["rules"] if r_["signal"].split(".")[0] in cfg["models"]})
    X = np.array([[r["scores"].get(k, 0.0) for k in keys] for r in recs])
    C = np.array([COST[c] for c in CLASSES])
    pred = np.zeros(len(recs), int)
    for k in range(5):
        tr, va = dev & (fold != k), dev & (fold == k)
        if len(set(y[tr])) < 3 or not va.sum():
            return None
        lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        pred[va] = (lr.predict_proba(X[va]) @ C).argmin(1)
    return pred


def _calibration(recs, y, dev):
    lines = ["| signal | bins (mean score → observed explicit rate, n) |", "|---|---|"]
    for s in ("freepik.high", "falconsai.nsfw", "nudenet.max_exposed"):
        v = np.array([r["scores"].get(s, np.nan) for r in recs])
        m = dev & ~np.isnan(v) & (y != 1)
        if not m.sum():
            continue
        bins = np.digitize(v[m], [0.1, 0.3, 0.5, 0.7, 0.9])
        cells = [f"{v[m][bins == b].mean():.2f}→{(y[m][bins == b] == 2).mean():.2f} (n={int((bins == b).sum())})"
                 for b in range(6) if (bins == b).sum()]
        lines.append(f"| {s} | " + "; ".join(cells) + " |")
    return lines + [""]


def _plots(curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for s, (pos, neg) in curves.items():
        th = np.unique(np.r_[pos, neg])[::-1]
        tpr = [(pos >= t).mean() for t in th]
        fpr = [(neg >= t).mean() for t in th]
        prec = [(pos >= t).sum() / max(1, (pos >= t).sum() + (neg >= t).sum()) for t in th]
        a.plot(fpr, tpr, label=s)
        b.plot(tpr, prec, label=s)
    a.set(xlabel="false positive rate (safe flagged)", ylabel="recall (explicit)", title="ROC", xscale="log")
    b.set(xlabel="recall", ylabel="precision", title="PR")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS / "curves.png", dpi=110)


# ---------------------------------------------------------------- robustness
def _transforms(img, regions):
    w, h = img.size
    sticker = img.copy()
    d = ImageDraw.Draw(sticker)
    for r in regions:  # cover detected exposed parts with emoji-like discs
        x, y, bw, bh = r["box"]
        d.ellipse([x, y, x + bw, y + bh], fill=(255, 205, 0))
    meme = ImageOps.expand(img, border=(0, h // 5, 0, h // 5), fill="white")
    ImageDraw.Draw(meme).text((10, 10), "WHEN THE WEEKEND HITS " * 3, fill="black")
    screen = Image.new("RGB", (w, h), (25, 25, 25))
    small = img.resize((int(w * 0.6), int(h * 0.6))).filter(ImageFilter.GaussianBlur(1))
    screen.paste(small, (int(w * 0.2), int(h * 0.2)))
    screen = Image.fromarray(np.clip(np.asarray(screen, np.int16) + np.random.default_rng(0).integers(-12, 12, (h, w, 3)), 0, 255).astype(np.uint8))
    return {
        "crop_center_70": img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))),
        "rotate_90": img.rotate(90, expand=True), "rotate_15": img.rotate(15, expand=True),
        "mirror": ImageOps.mirror(img), "grayscale": ImageOps.grayscale(img).convert("RGB"),
        "blur_r4": img.filter(ImageFilter.GaussianBlur(4)), "jpeg_q10": img, "pad_border_40": ImageOps.expand(img, border=int(max(w, h) * 0.4), fill="black"),
        "stickers_on_regions": sticker, "meme_text": meme, "screenshot_of_screen": screen,
    }


def cmd_robust(args):
    m = engine.Moderator()
    recs = [r for r in load_records() if r["label"] in ("reject", "review") and r["split"] == "dev"][: args.limit]
    res = defaultdict(Counter)
    for r in recs:
        base, _ = m.moderate(Path(r["path"]).read_bytes())
        img = engine.decode(Path(r["path"]).read_bytes(), m.limits)[0]
        regions = m.score([img])[1]
        res["original"][base.decision] += 1
        for name, t in _transforms(img, [x for x in regions if x["label"] in EXPOSED]).items():
            buf = io.BytesIO()
            t.save(buf, "JPEG", quality=10 if name == "jpeg_q10" else 90)
            d, _ = m.moderate(buf.getvalue())
            res[name][d.decision] += 1
            if d.decision == "APPROVE" and base.decision != "APPROVE" and r["label"] == "reject":
                res[name]["_escaped"] += 1
    lines = ["# Robustness (dev reject+review images, production pipeline)", "", f"n = {len(recs)}", "",
             "| transform | APPROVE | REVIEW | REJECT | explicit escaped to APPROVE |", "|---|---|---|---|---|"]
    for name, c in res.items():
        lines.append(f"| {name} | {c['APPROVE']} | {c['REVIEW']} | {c['REJECT']} | {c['_escaped']} |")
    (RUNS / "robustness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- regression gate
def cmd_gate(args):
    if not Path("eval_baseline.json").exists() or not Path("eval_metrics.json").exists():
        sys.exit(print("gate: no eval_baseline.json/eval_metrics.json yet, skipping") or 0)
    base = json.loads(Path("eval_baseline.json").read_text())
    cur = json.loads(Path("eval_metrics.json").read_text())
    tol = 0.005
    worse = {k: (base[k], cur[k]) for k in ("explicit_auto_approved", "safe_auto_rejected", "cost")
             if cur.get(k, 1e9) > base[k] + tol}
    worse |= {k: (base[k], cur[k]) for k in ("explicit_caught_auto",) if cur.get(k, -1) < base[k] - tol}
    for k, (b, c) in worse.items():
        print(f"REGRESSION {k}: baseline {b:.4f} -> now {c:.4f}")
    sys.exit(1 if worse else print("gate: OK"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", default="labeled")
    r = sub.add_parser("report")
    r.add_argument("--final", action="store_true", help="evaluate the held-out test split (do this once)")
    rb = sub.add_parser("robust")
    rb.add_argument("--limit", type=int, default=500)
    sub.add_parser("gate")
    a = ap.parse_args()
    {"score": cmd_score, "report": cmd_report, "robust": cmd_robust, "gate": cmd_gate}[a.cmd](a)
