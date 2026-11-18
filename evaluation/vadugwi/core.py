"""Data loading, metrics and bootstrap for the VADUGWI held-out evaluation.

Standard library only. Nothing here prints or stores sentence text: items are
identified by a truncated SHA-256 of the text.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LABELS = ("neg", "neutral", "pos")
NEG_BELOW = 110   # the repo's own thresholds (benchmarks/eval_holdout_v2.py)
POS_FROM = 145
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261118


def data_root() -> Path:
    root = os.environ.get("CLANKER_VADUGWI_DATA")
    if not root:
        raise SystemExit(
            "Set CLANKER_VADUGWI_DATA to the private checkout that holds "
            "benchmarks/ and datasets/ (they are not in this repository)."
        )
    return Path(root).expanduser()


def item_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Datasets ────────────────────────────────────────────────────
# role: "fit" may be used to choose rules and constants; "heldout" may only
# be measured; "regression" is the in-repo material the engine was tuned on.

def _council_train(root: Path) -> Tuple[List[dict], Path]:
    path = root / "datasets" / "council_v2_train.json"
    rows = json.loads(path.read_text())
    return [{"text": r["text"], "label": r["label"], "category": r["category"]}
            for r in rows], path


def _probes(rel: str) -> Callable[[Path], Tuple[List[dict], Path]]:
    def load(root: Path) -> Tuple[List[dict], Path]:
        path = root / rel
        rows = json.loads(path.read_text())
        return [{"text": r["text"], "label": r["expected"],
                 "category": r.get("category", "")} for r in rows], path
    return load


def _emobank(half: int) -> Callable[[Path], Tuple[List[dict], Path]]:
    """EmoBank human valence, split in two by a hash of the text."""
    def load(root: Path) -> Tuple[List[dict], Path]:
        path = root / "training" / "data" / "emobank_vadug.jsonl"
        out = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row["english"]
            if int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 2 == half:
                out.append({"text": text, "score": float(row["vadug"][0])})
        return out, path
    return load


def _sst2(root: Path) -> Tuple[List[dict], Path]:
    path = Path(os.environ.get("CLANKER_VADUGWI_SST2", root / "sst2_validation.json"))
    return json.loads(path.read_text()), path


def _stress(root: Path) -> Tuple[List[dict], Path]:
    path = root / "benchmarks" / "stress_test.py"
    spec = importlib.util.spec_from_file_location("_vadugwi_stress", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports the engine already on sys.path
    return [{"text": text, "label": label, "category": category}
            for category, rows in module.CATEGORIES.items()
            for text, label in rows], path


DATASETS: Dict[str, dict] = {
    "council_train": {"role": "fit", "kind": "class", "load": _council_train},
    "emobank_dev": {"role": "fit", "kind": "score", "load": _emobank(0)},
    "council_heldout": {"role": "heldout", "kind": "class",
                        "load": _probes("benchmarks/holdout_v2_probes.json")},
    "heldout_v1": {"role": "heldout", "kind": "class",
                   "load": _probes("benchmarks/holdout_probes.json")},
    "emobank_test": {"role": "heldout", "kind": "score", "load": _emobank(1)},
    "sst2_validation": {"role": "heldout", "kind": "binary", "load": _sst2},
    # stress_test.py counts V == 110 as negative, so this set does too.
    "stress": {"role": "regression", "kind": "class", "load": _stress, "neg_below": 111},
}


def load(name: str, purpose: str) -> Tuple[List[dict], str]:
    """Load a dataset. purpose is "fit" or "measure"; fitting on anything
    that is not a fit set fails closed."""
    spec = DATASETS[name]
    if purpose == "fit" and spec["role"] != "fit":
        raise PermissionError(f"{name} is {spec['role']} data and cannot be fitted on")
    if purpose not in ("fit", "measure"):
        raise ValueError(purpose)
    rows, path = spec["load"](data_root())
    return rows, file_sha256(path)


# ── Metrics ─────────────────────────────────────────────────────

def classify(v: float, neg_below: float = NEG_BELOW, pos_from: float = POS_FROM) -> str:
    if v >= pos_from:
        return "pos"
    if v < neg_below:
        return "neg"
    return "neutral"


def auc_pos_vs_neg(values: Sequence[float], labels: Sequence[str]) -> float:
    """P(V of a pos item > V of a neg item), ties count half. Threshold-free."""
    pos = [v for v, g in zip(values, labels) if g == "pos"]
    neg = sorted(v for v, g in zip(values, labels) if g == "neg")
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for v in pos:
        lo = bisect.bisect_left(neg, v)
        hi = bisect.bisect_right(neg, v)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def class_metrics(values: Sequence[float], labels: Sequence[str],
                  neg_below: float = NEG_BELOW) -> Dict[str, float]:
    pred = [classify(v, neg_below) for v in values]
    n = len(values)
    out = {"accuracy": sum(p == g for p, g in zip(pred, labels)) / n}
    recalls, f1s = [], []
    for c in LABELS:
        tp = sum(p == c and g == c for p, g in zip(pred, labels))
        fn = sum(p != c and g == c for p, g in zip(pred, labels))
        fp = sum(p == c and g != c for p, g in zip(pred, labels))
        if tp + fn == 0:
            continue  # class absent from this set
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(recall)
        f1s.append(2 * recall * precision / (recall + precision) if recall + precision else 0.0)
        out[f"recall_{c}"] = recall
    out["balanced_accuracy"] = sum(recalls) / len(recalls)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["auc_pos_vs_neg"] = auc_pos_vs_neg(values, labels)
    counts = {c: sum(g == c for g in labels) for c in LABELS}
    out["majority_class_accuracy"] = max(counts.values()) / n
    return out


def binary_metrics(values: Sequence[float], labels: Sequence[str]) -> Dict[str, float]:
    n = len(values)
    hits = sum(1.0 if (v > 128) == (g == "pos") else 0.0 for v, g in zip(values, labels) if v != 128)
    hits += 0.5 * sum(v == 128 for v in values)  # no opinion counts half
    return {"sign_accuracy": hits / n, "auc_pos_vs_neg": auc_pos_vs_neg(values, labels),
            "no_opinion_rate": sum(v == 128 for v in values) / n}


def score_metrics(values: Sequence[float], scores: Sequence[float]) -> Dict[str, float]:
    return {"pearson_r": pearson(values, scores)}


def metrics_for(kind: str, values: Sequence[float], truth: Sequence,
                neg_below: float = NEG_BELOW) -> Dict[str, float]:
    if kind == "class":
        return class_metrics(values, truth, neg_below)
    if kind == "binary":
        return binary_metrics(values, truth)
    return score_metrics(values, truth)


def _percentile(sorted_values: List[float], q: float) -> float:
    idx = q * (len(sorted_values) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def bootstrap(kind: str, truth: Sequence, values: Sequence[float],
              other: Optional[Sequence[float]] = None,
              draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
              neg_below: float = NEG_BELOW) -> Dict[str, List[float]]:
    """95% percentile intervals from item resampling with a fixed seed.

    With `other`, the interval is for metric(values) - metric(other) on the
    same resampled items (a paired comparison).
    """
    rng = random.Random(seed)
    n = len(values)
    samples: Dict[str, List[float]] = {}
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        t = [truth[i] for i in idx]
        m = metrics_for(kind, [values[i] for i in idx], t, neg_below)
        if other is not None:
            mo = metrics_for(kind, [other[i] for i in idx], t, neg_below)
            m = {k: m[k] - mo[k] for k in m}
        for k, v in m.items():
            if not math.isnan(v):
                samples.setdefault(k, []).append(v)
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = [_percentile(vals, 0.025), _percentile(vals, 0.975)]
    return out
