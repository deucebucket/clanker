"""Command line for the VADUGWI held-out evaluation.

  python -m evaluation.vadugwi run --label NAME [--sets a,b,c]
  python -m evaluation.vadugwi compare BASE.json NEW.json
  python -m evaluation.vadugwi fit-reliability

`run` writes results/NAME.json (aggregate metrics with bootstrap intervals)
and results/NAME.items.json (V per item, keyed by a hash, never the sentence).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.pendulum import compute_vadug  # noqa: E402

from . import core  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_SETS = ("council_train", "council_heldout", "heldout_v1", "emobank_dev",
                "emobank_test", "sst2_validation", "stress")


def _truth(kind: str, rows):
    return [r["score"] for r in rows] if kind == "score" else [r["label"] for r in rows]


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "engine"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty-engine" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_run(args) -> int:
    report = {"label": args.label, "engine_commit": _commit(),
              "thresholds": [core.NEG_BELOW, core.POS_FROM],
              "bootstrap": {"draws": core.BOOTSTRAP_DRAWS, "seed": core.BOOTSTRAP_SEED},
              "sets": {}}
    items = {}
    for name in args.sets.split(","):
        spec = core.DATASETS[name]
        try:
            rows, sha = core.load(name, "measure")
        except FileNotFoundError as exc:
            print(f"{name}: skipped ({exc.filename} not found)")
            continue
        values = [compute_vadug(r["text"])[0].v for r in rows]
        truth = _truth(spec["kind"], rows)
        neg_below = spec.get("neg_below", core.NEG_BELOW)
        metrics = core.metrics_for(spec["kind"], values, truth, neg_below)
        ci = core.bootstrap(spec["kind"], truth, values, neg_below=neg_below)
        report["sets"][name] = {
            "role": spec["role"], "kind": spec["kind"], "n": len(rows),
            "source_sha256": sha, "metrics": metrics, "ci95": ci,
        }
        items[name] = {core.item_id(r["text"]): v for r, v in zip(rows, values)}
        shown = ", ".join(
            f"{k} {v:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else f"{k} {v:.3f}"
            for k, v in metrics.items() if not k.startswith("recall_"))
        print(f"{name} ({spec['role']}, n={len(rows)}): {shown}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (RESULTS / f"{args.label}.items.json").write_text(
        json.dumps(items, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {out}")
    return 0


def cmd_compare(args) -> int:
    base = json.loads(Path(args.base).read_text())
    new = json.loads(Path(args.new).read_text())
    base_items = json.loads(Path(args.base).with_suffix(".items.json").read_text())
    new_items = json.loads(Path(args.new).with_suffix(".items.json").read_text())
    print(f"{new['label']} minus {base['label']} (paired bootstrap, 95% interval)")
    for name, new_set in new["sets"].items():
        base_set = base["sets"].get(name)
        if base_set is None:
            continue
        spec = core.DATASETS[name]
        rows, sha = core.load(name, "measure")
        if sha != new_set["source_sha256"] or sha != base_set["source_sha256"]:
            raise SystemExit(f"{name}: data file changed since the runs")
        truth = _truth(spec["kind"], rows)
        ids = [core.item_id(r["text"]) for r in rows]
        a = [new_items[name][i] for i in ids]
        b = [base_items[name][i] for i in ids]
        neg_below = spec.get("neg_below", core.NEG_BELOW)
        ci = core.bootstrap(spec["kind"], truth, a, other=b, neg_below=neg_below)
        ma = core.metrics_for(spec["kind"], a, truth, neg_below)
        mb = core.metrics_for(spec["kind"], b, truth, neg_below)
        print(f"{name} ({spec['role']}, n={len(rows)}):")
        for k in ma:
            if k.startswith("recall_") or k == "majority_class_accuracy" or k not in ci:
                continue
            lo, hi = ci[k]
            flag = "" if lo <= 0 <= hi else "  *"
            print(f"    {k:18} {mb[k]:.3f} -> {ma[k]:.3f}   diff {ma[k]-mb[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.vadugwi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--sets", default=",".join(DEFAULT_SETS))
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare")
    compare.add_argument("base")
    compare.add_argument("new")
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
