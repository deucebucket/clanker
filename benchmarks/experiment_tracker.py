#!/usr/bin/env python3
"""NASA-style experiment tracking for Clanker-Lang.

Every test run gets a versioned ID, theory name, full config snapshot,
results across all benchmarks, and deltas vs baseline.

Usage:
    # Log an experiment programmatically
    from benchmarks.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    exp_id = tracker.log_experiment(
        theory="baseline-v2-untuned",
        engine="v2",
        config={"momentum": 0.82, "force_scale": 0.50, ...},
        results={"sst2": {"accuracy": 60.9, ...}, ...},
        notes="First V2 benchmark run"
    )

    # View history
    python3 benchmarks/experiment_tracker.py --history
    python3 benchmarks/experiment_tracker.py --best
    python3 benchmarks/experiment_tracker.py --compare EXP-0001 EXP-0042
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Any

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "experiment_log.jsonl")
BASELINE_FILE = os.path.join(LOG_DIR, "experiment_baseline.json")


class ExperimentTracker:
    """Append-only experiment log with versioned IDs."""

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file

    def _next_id(self) -> str:
        """Get next experiment ID (EXP-0001, EXP-0002, ...)."""
        count = 0
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        count += 1
        return f"EXP-{count + 1:04d}"

    def _load_baseline(self) -> Optional[Dict]:
        """Load baseline results for delta calculation."""
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE) as f:
                return json.load(f)
        return None

    def _compute_deltas(self, results: Dict) -> Dict:
        """Compute accuracy deltas vs baseline."""
        baseline = self._load_baseline()
        if not baseline:
            return {}
        deltas = {}
        for dataset in results:
            if dataset in baseline and "accuracy" in results[dataset] and "accuracy" in baseline[dataset]:
                deltas[dataset] = round(results[dataset]["accuracy"] - baseline[dataset]["accuracy"], 2)
        return deltas

    def log_experiment(
        self,
        theory: str,
        engine: str,
        config: Dict[str, Any],
        results: Dict[str, Dict],
        notes: str = "",
        generation: Optional[int] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Log a single experiment run.

        Args:
            theory: Human-readable theory name (e.g. "anchor-word-entanglement-v3")
            engine: Engine version ("v1", "v2", "model", "v2+model")
            config: Full parameter snapshot
            results: Per-dataset results {dataset: {accuracy, f1, neutral_pct, ...}}
            notes: Free-form notes
            generation: If from genetic algorithm, which generation
            parent_id: If derived from another experiment

        Returns:
            Experiment ID (e.g. "EXP-0042")
        """
        exp_id = self._next_id()
        deltas = self._compute_deltas(results)

        # Compute composite score (weighted avg across benchmarks)
        accs = [r.get("accuracy", 0) for r in results.values()]
        composite = round(sum(accs) / len(accs), 2) if accs else 0

        entry = {
            "id": exp_id,
            "theory": theory,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "engine": engine,
            "config": config,
            "results": results,
            "composite_accuracy": composite,
            "delta_vs_baseline": deltas,
            "notes": notes,
        }
        if generation is not None:
            entry["generation"] = generation
        if parent_id:
            entry["parent_id"] = parent_id

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  Logged: {exp_id} | {theory} | composite={composite}% | {notes}")
        return exp_id

    def set_baseline(self, exp_id: str):
        """Set an experiment as the baseline for delta calculations."""
        exp = self.get_experiment(exp_id)
        if not exp:
            print(f"  ERROR: {exp_id} not found")
            return
        with open(BASELINE_FILE, "w") as f:
            json.dump(exp["results"], f, indent=2)
        print(f"  Baseline set to {exp_id} ({exp['theory']})")

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Retrieve a single experiment by ID."""
        if not os.path.exists(self.log_file):
            return None
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["id"] == exp_id:
                    return entry
        return None

    def load_all(self):
        """Load all experiments."""
        experiments = []
        if not os.path.exists(self.log_file):
            return experiments
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    experiments.append(json.loads(line))
        return experiments

    def print_history(self, limit: int = 50):
        """Print experiment history as a formatted table."""
        experiments = self.load_all()
        if not experiments:
            print("  No experiments logged yet.")
            return

        # Header
        print(f"\n  {'ID':<10} {'Theory':<40} {'Engine':<8} {'SST-2':>7} {'GoEmo':>7} {'Tweet':>7} {'Comp':>7} {'Date':<12}")
        print(f"  {'-'*100}")

        for exp in experiments[-limit:]:
            r = exp.get("results", {})
            sst2 = r.get("sst2", {}).get("accuracy", "—")
            goemo = r.get("goemotions", {}).get("accuracy", "—")
            tweet = r.get("tweeteval", {}).get("accuracy", "—")
            comp = exp.get("composite_accuracy", "—")
            date = exp.get("timestamp", "")[:10]
            theory = exp.get("theory", "")[:38]

            sst2_s = f"{sst2:>6.1f}%" if isinstance(sst2, (int, float)) else f"{'—':>7}"
            goemo_s = f"{goemo:>6.1f}%" if isinstance(goemo, (int, float)) else f"{'—':>7}"
            tweet_s = f"{tweet:>6.1f}%" if isinstance(tweet, (int, float)) else f"{'—':>7}"
            comp_s = f"{comp:>6.1f}%" if isinstance(comp, (int, float)) else f"{'—':>7}"

            print(f"  {exp['id']:<10} {theory:<40} {exp['engine']:<8} {sst2_s} {goemo_s} {tweet_s} {comp_s} {date}")

        print(f"\n  Total experiments: {len(experiments)}")

    def print_best(self, metric: str = "composite_accuracy"):
        """Print top 10 experiments by composite accuracy."""
        experiments = self.load_all()
        if not experiments:
            print("  No experiments logged yet.")
            return

        sorted_exps = sorted(experiments, key=lambda e: e.get(metric, 0), reverse=True)

        print(f"\n  TOP 10 by {metric}:")
        print(f"  {'Rank':<6} {'ID':<10} {'Theory':<40} {'Composite':>10}")
        print(f"  {'-'*70}")

        for i, exp in enumerate(sorted_exps[:10], 1):
            theory = exp.get("theory", "")[:38]
            comp = exp.get(metric, 0)
            print(f"  {i:<6} {exp['id']:<10} {theory:<40} {comp:>9.2f}%")

    def print_compare(self, id_a: str, id_b: str):
        """Compare two experiments side by side."""
        a = self.get_experiment(id_a)
        b = self.get_experiment(id_b)
        if not a or not b:
            print(f"  ERROR: Could not find both {id_a} and {id_b}")
            return

        print(f"\n  COMPARISON: {id_a} vs {id_b}")
        print(f"  {'':15} {id_a:<25} {id_b:<25} {'Delta':>10}")
        print(f"  {'-'*80}")

        print(f"  {'Theory':<15} {a['theory']:<25} {b['theory']:<25}")
        print(f"  {'Engine':<15} {a['engine']:<25} {b['engine']:<25}")

        datasets = set(list(a.get("results", {}).keys()) + list(b.get("results", {}).keys()))
        for ds in sorted(datasets):
            acc_a = a.get("results", {}).get(ds, {}).get("accuracy", 0)
            acc_b = b.get("results", {}).get(ds, {}).get("accuracy", 0)
            delta = acc_b - acc_a
            sign = "+" if delta > 0 else ""
            print(f"  {ds:<15} {acc_a:>24.1f}% {acc_b:>24.1f}% {sign}{delta:>9.1f}pp")

        comp_a = a.get("composite_accuracy", 0)
        comp_b = b.get("composite_accuracy", 0)
        delta = comp_b - comp_a
        sign = "+" if delta > 0 else ""
        print(f"  {'COMPOSITE':<15} {comp_a:>24.2f}% {comp_b:>24.2f}% {sign}{delta:>9.2f}pp")

        # Config diff
        print(f"\n  CONFIG DIFFERENCES:")
        config_a = a.get("config", {})
        config_b = b.get("config", {})
        all_keys = set(list(config_a.keys()) + list(config_b.keys()))
        diffs = 0
        for key in sorted(all_keys):
            va = config_a.get(key)
            vb = config_b.get(key)
            if va != vb and not isinstance(va, dict) and not isinstance(vb, dict):
                print(f"    {key}: {va} -> {vb}")
                diffs += 1
        if diffs == 0:
            print(f"    (identical configs)")


def main():
    parser = argparse.ArgumentParser(description="Clanker Experiment Tracker")
    parser.add_argument("--history", action="store_true", help="Show experiment history")
    parser.add_argument("--best", action="store_true", help="Show top experiments")
    parser.add_argument("--compare", nargs=2, metavar=("ID_A", "ID_B"), help="Compare two experiments")
    parser.add_argument("--get", metavar="ID", help="Get full experiment details")
    parser.add_argument("--limit", type=int, default=50, help="Limit history rows")
    args = parser.parse_args()

    tracker = ExperimentTracker()

    if args.history:
        tracker.print_history(limit=args.limit)
    elif args.best:
        tracker.print_best()
    elif args.compare:
        tracker.print_compare(args.compare[0], args.compare[1])
    elif args.get:
        exp = tracker.get_experiment(args.get)
        if exp:
            print(json.dumps(exp, indent=2))
        else:
            print(f"  Not found: {args.get}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
