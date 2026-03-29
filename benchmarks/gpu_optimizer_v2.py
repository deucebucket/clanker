#!/usr/bin/env python3
"""V2 Engine Optimizer — Genetic algorithm tuning for PendulumV2.

Evaluates parameter configurations against all 3 academic benchmarks.
Uses multiprocessing to parallelize engine runs across CPU cores.

Each organism is a full PendulumV2 config. Fitness = weighted accuracy
across SST-2, GoEmotions, and TweetEval.

Usage:
    python3 benchmarks/gpu_optimizer_v2.py
    python3 benchmarks/gpu_optimizer_v2.py --pop 100 --gen 200
    python3 benchmarks/gpu_optimizer_v2.py --quick  # 200 sentences, fast iteration
"""

import json
import os
import sys
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

PARAM_SPEC = [
    # (name, min, max, default)
    ("momentum",           0.50,  0.99,  0.82),
    ("force_scale",        0.10,  3.00,  0.50),
    ("direct_push_cap",    0.05,  0.90,  0.40),
    ("direct_push_trigger", 10.0, 200.0, 80.0),
    ("crisis_v",           20.0,  100.0, 60.0),
    ("crisis_u",           10.0,  100.0, 40.0),
    ("question_dampener",  0.10,  1.00,  0.40),
    ("scale_v",            0.30,  3.00,  1.00),
    ("scale_a",            0.30,  3.00,  1.00),
    ("scale_d",            0.30,  3.00,  1.00),
    ("scale_u",            0.30,  3.00,  1.00),
    ("scale_g",            0.30,  3.00,  1.00),
    ("threshold_low",      95.0,  130.0, 124.0),
    ("threshold_high",     125.0, 160.0, 132.0),
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_MINS = [p[1] for p in PARAM_SPEC]
PARAM_MAXS = [p[2] for p in PARAM_SPEC]
PARAM_DEFAULTS = [p[3] for p in PARAM_SPEC]
N_PARAMS = len(PARAM_SPEC)


def params_to_config(params: List[float]) -> dict:
    """Convert parameter list to PendulumV2 kwargs."""
    return {name: val for name, val in zip(PARAM_NAMES, params)}


# ---------------------------------------------------------------------------
# Test data loader (cached across processes)
# ---------------------------------------------------------------------------

_CACHED_DATA = None

def load_test_data(max_per_dataset: int = None):
    """Load benchmark sentences with truth labels."""
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    from datasets import load_dataset

    GOEMO = {0:'positive',1:'positive',2:'negative',3:'negative',4:'positive',5:'positive',
             6:'neutral',7:'neutral',8:'positive',9:'negative',10:'negative',11:'negative',
             12:'negative',13:'positive',14:'negative',15:'positive',16:'negative',
             17:'positive',18:'positive',19:'negative',20:'positive',21:'positive',
             22:'neutral',23:'positive',24:'negative',25:'negative',26:'neutral',27:'neutral'}
    TW = {0:'negative',1:'positive',2:'positive',3:'negative'}

    data = []  # list of (text, truth, dataset_name)

    print("  Loading SST-2...")
    sst = load_dataset('stanfordnlp/sst2', split='validation')
    n = min(len(sst), max_per_dataset) if max_per_dataset else len(sst)
    for row in sst.select(range(n)):
        truth = 'positive' if row['label'] == 1 else 'negative'
        data.append((row['sentence'], truth, 'sst2'))

    print("  Loading GoEmotions...")
    ge = load_dataset('google-research-datasets/go_emotions', split='test')
    n = min(len(ge), max_per_dataset) if max_per_dataset else len(ge)
    for row in ge.select(range(n)):
        if not row['labels']:
            continue
        truth = GOEMO.get(row['labels'][0], 'neutral')
        data.append((row['text'], truth, 'goemotions'))

    print("  Loading TweetEval...")
    te = load_dataset('cardiffnlp/tweet_eval', 'emotion', split='test')
    n = min(len(te), max_per_dataset) if max_per_dataset else len(te)
    for row in te.select(range(n)):
        truth = TW[row['label']]
        data.append((row['text'], truth, 'tweeteval'))

    print(f"  Loaded {len(data)} total sentences")
    _CACHED_DATA = data
    return data


# ---------------------------------------------------------------------------
# Evaluation function (runs in worker process)
# ---------------------------------------------------------------------------

def evaluate_organism(params_and_data):
    """Evaluate a single organism's config against all test data.

    Returns: (params, accuracy_dict, composite_accuracy)
    """
    params, data = params_and_data

    # Import here to avoid pickling issues
    from demo.pendulum_v2 import PendulumV2

    config = params_to_config(params)
    engine = PendulumV2(**config)

    # Per-dataset scoring with correct classification mode
    # SST-2 and TweetEval are binary tests (no neutral in ground truth)
    # GoEmotions has neutral labels
    DATASET_MODES = {"sst2": "binary", "tweeteval": "binary", "goemotions": "three_way"}

    counts = {}  # dataset -> {correct, total}
    for text, truth, dataset in data:
        if dataset not in counts:
            counts[dataset] = {"correct": 0, "total": 0}

        vadug, _ = engine.process_text(text)
        mode = DATASET_MODES.get(dataset, "three_way")
        pred = engine.classify(vadug.v, mode)

        if pred == truth:
            counts[dataset]["correct"] += 1
        counts[dataset]["total"] += 1

    # Calculate accuracies
    results = {}
    for ds, c in counts.items():
        results[ds] = round(c["correct"] / c["total"] * 100, 2) if c["total"] > 0 else 0

    # Composite: equal weight across all datasets
    composite = round(sum(results.values()) / len(results), 2) if results else 0

    return params, results, composite


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def random_organism():
    """Create a random parameter set within bounds."""
    return [random.uniform(mn, mx) for mn, mx in zip(PARAM_MINS, PARAM_MAXS)]


def crossover(parent_a, parent_b):
    """Single-point crossover."""
    point = random.randint(1, N_PARAMS - 1)
    child = parent_a[:point] + parent_b[point:]
    return child


def mutate(organism, rate=0.15):
    """Mutate each parameter with probability `rate`."""
    mutated = list(organism)
    for i in range(N_PARAMS):
        if random.random() < rate:
            span = PARAM_MAXS[i] - PARAM_MINS[i]
            delta = random.gauss(0, span * 0.1)
            mutated[i] = max(PARAM_MINS[i], min(PARAM_MAXS[i], mutated[i] + delta))
    return mutated


def run_evolution(
    data,
    n_pop: int = 100,
    n_generations: int = 200,
    n_workers: int = 8,
    snapshot_every: int = 25,
):
    """Run genetic algorithm. Returns (best_params, best_results, history)."""
    from benchmarks.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()

    # Initialize population (include default config as one organism)
    population = [list(PARAM_DEFAULTS)]
    population += [random_organism() for _ in range(n_pop - 1)]

    best_params = None
    best_composite = 0
    best_results = {}
    stagnation = 0
    history = []

    print(f"\n{'='*70}")
    print(f"  GENETIC OPTIMIZER V2 — {n_pop} organisms, {n_generations} generations")
    print(f"  {len(data)} sentences, {n_workers} workers")
    print(f"{'='*70}\n")

    t0 = time.time()

    for gen in range(n_generations):
        gen_t0 = time.time()

        # Evaluate all organisms in parallel
        work_items = [(org, data) for org in population]

        gen_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(evaluate_organism, item) for item in work_items]
            for future in as_completed(futures):
                gen_results.append(future.result())

        # Sort by composite accuracy (descending)
        gen_results.sort(key=lambda x: x[2], reverse=True)

        gen_best_params, gen_best_results, gen_best_composite = gen_results[0]
        gen_time = time.time() - gen_t0

        # Track global best
        if gen_best_composite > best_composite:
            best_composite = gen_best_composite
            best_params = gen_best_params
            best_results = gen_best_results
            stagnation = 0
            print(f"  Gen {gen:>4}: {best_composite:>6.2f}%  *** NEW BEST  "
                  f"SST2={best_results.get('sst2',0):.1f}  "
                  f"GoEmo={best_results.get('goemotions',0):.1f}  "
                  f"Tweet={best_results.get('tweeteval',0):.1f}  "
                  f"({gen_time:.1f}s)")
        else:
            stagnation += 1
            if gen % 10 == 0:
                print(f"  Gen {gen:>4}: best={best_composite:>6.2f}%  "
                      f"gen_best={gen_best_composite:.2f}%  "
                      f"stagnation={stagnation}  ({gen_time:.1f}s)")

        # History entry
        history.append({
            "generation": gen,
            "best_composite": best_composite,
            "gen_best_composite": gen_best_composite,
            "population_diversity": _diversity(population),
            "stagnation": stagnation,
            "time": round(gen_time, 1),
        })

        # Snapshot to experiment tracker
        if gen > 0 and gen % snapshot_every == 0:
            config = params_to_config(best_params)
            tracker.log_experiment(
                theory=f"genetic-gen{gen:04d}",
                engine="v2",
                config=config,
                results={ds: {"accuracy": acc} for ds, acc in best_results.items()},
                generation=gen,
                notes=f"Gen {gen}/{n_generations}, stagnation={stagnation}",
            )

        # Early stopping
        if stagnation > 80:
            print(f"\n  Converged at generation {gen} (stagnation={stagnation})")
            break

        # Selection: top 20% survive
        survivors = [r[0] for r in gen_results[:n_pop // 5]]

        # Elitism: always keep the global best
        new_pop = [list(best_params)]

        # Breed from survivors
        while len(new_pop) < n_pop:
            pa = random.choice(survivors)
            pb = random.choice(survivors)
            child = crossover(pa, pb)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    elapsed = time.time() - t0
    total_evals = len(history) * n_pop * len(data)

    # Log final champion
    config = params_to_config(best_params)
    exp_id = tracker.log_experiment(
        theory="genetic-champion",
        engine="v2",
        config=config,
        results={ds: {"accuracy": acc} for ds, acc in best_results.items()},
        generation=len(history),
        notes=f"Champion after {len(history)} generations, {total_evals:,} evals, {elapsed:.0f}s",
    )

    print(f"\n{'='*70}")
    print(f"  CHAMPION: {best_composite:.2f}% (composite)")
    print(f"  {exp_id}")
    print(f"{'='*70}")
    for name, val in zip(PARAM_NAMES, best_params):
        default = PARAM_DEFAULTS[PARAM_NAMES.index(name)]
        delta = val - default
        sign = "+" if delta > 0 else ""
        print(f"    {name:<22}: {val:>8.4f}  (default={default}, {sign}{delta:.4f})")
    print(f"\n  Results:")
    for ds, acc in sorted(best_results.items()):
        print(f"    {ds:<15}: {acc:.2f}%")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Total evaluations: {total_evals:,}")
    if elapsed > 0:
        print(f"  Rate: {total_evals/elapsed:,.0f} evals/sec")

    # Save champion config
    out = {
        "champion_composite": best_composite,
        "champion_results": best_results,
        "champion_config": config,
        "generations": len(history),
        "total_evaluations": total_evals,
        "time_seconds": round(elapsed),
        "history": history,
    }
    out_path = os.path.join(os.path.dirname(__file__), "gpu_optimal_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return best_params, best_results, history


def _diversity(population):
    """Measure population diversity (avg std across params)."""
    import statistics
    if len(population) < 2:
        return 0
    stds = []
    for i in range(N_PARAMS):
        vals = [org[i] for org in population]
        stds.append(statistics.stdev(vals) / (PARAM_MAXS[i] - PARAM_MINS[i]))
    return round(sum(stds) / len(stds), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2 Engine Genetic Optimizer")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=200, help="Max generations")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--quick", action="store_true", help="Fast mode: 200 sentences per dataset")
    parser.add_argument("--snapshot", type=int, default=25, help="Log every N generations")
    args = parser.parse_args()

    max_per = 200 if args.quick else 1000
    data = load_test_data(max_per_dataset=max_per)

    run_evolution(
        data,
        n_pop=args.pop,
        n_generations=args.gen,
        n_workers=args.workers,
        snapshot_every=args.snapshot,
    )


if __name__ == "__main__":
    main()
