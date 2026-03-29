#!/usr/bin/env python3
"""EmoBank Optimizer — tune engine for human V/A/D correlation.

Instead of optimizing for binary pos/neg accuracy, this optimizes
for Pearson correlation with human dimensional ratings from EmoBank.

This is THE test: can the engine produce numbers that agree with
how humans actually perceive emotional dimensions?

Usage:
    python3 benchmarks/emobank_optimizer.py
    python3 benchmarks/emobank_optimizer.py --pop 80 --gen 100
    python3 benchmarks/emobank_optimizer.py --quick
"""

import csv
import json
import os
import sys
import time
import random
import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Parameter space — same 27 params as the main optimizer
# ---------------------------------------------------------------------------

PARAM_SPEC = [
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
    ("negation_decay_operator", 0.70, 0.99, 0.92),
    ("negation_decay_neutral",  0.60, 0.95, 0.85),
    ("negation_decay_payload",  0.10, 0.60, 0.35),
    ("evoker_decay",            0.70, 0.99, 0.88),
    ("conditional_dampener",    0.20, 0.80, 0.40),
    ("clinical_dampener",       0.10, 0.60, 0.30),
    ("passive_d_offset",       -30.0, 0.0, -15.0),
    ("comparative_amplifier",   1.0,  2.0,  1.30),
    ("superlative_amplifier",   1.0,  2.5,  1.50),
    ("filler_d_offset",        -15.0, 0.0,  -5.0),
    ("litotes_dampener",        0.40, 0.90, 0.70),
    ("weaponized_d_boost",      5.0,  30.0, 15.0),
    ("tag_d_offset",           -20.0, 0.0, -10.0),
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_MINS  = [p[1] for p in PARAM_SPEC]
PARAM_MAXS  = [p[2] for p in PARAM_SPEC]
PARAM_DEFAULTS = [p[3] for p in PARAM_SPEC]
N_PARAMS = len(PARAM_SPEC)


def params_to_config(params):
    return {name: val for name, val in zip(PARAM_NAMES, params)}


# ---------------------------------------------------------------------------
# Load EmoBank data
# ---------------------------------------------------------------------------

_EMOBANK = None

def load_emobank(max_n=None):
    global _EMOBANK
    if _EMOBANK is not None:
        return _EMOBANK

    data = []
    emobank_path = os.path.join(os.path.dirname(__file__), "emobank.csv")
    with open(emobank_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip('"')
            if not text or len(text) < 5:
                continue
            data.append((text, float(row["V"]), float(row["A"]), float(row["D"])))

    if max_n and len(data) > max_n:
        random.seed(42)
        data = random.sample(data, max_n)

    _EMOBANK = data
    return data


# ---------------------------------------------------------------------------
# Fitness function: Pearson correlation with human V/A/D
# ---------------------------------------------------------------------------

def pearson(x, y):
    n = len(x)
    if n < 3:
        return 0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0


def evaluate_organism(args):
    """Evaluate one config against EmoBank. Returns (params, r_v, r_a, r_d, composite)."""
    params, data = args

    from demo.pendulum_v2 import PendulumV2
    config = params_to_config(params)
    engine = PendulumV2(**config)

    human_v, engine_v = [], []
    human_a, engine_a = [], []
    human_d, engine_d = [], []

    for text, hv, ha, hd in data:
        vadug, _ = engine.process_text(text)
        # Convert 0-255 to 1-5 scale
        engine_v.append(1 + (vadug.v / 255) * 4)
        engine_a.append(1 + (vadug.a / 255) * 4)
        engine_d.append(1 + (vadug.d / 255) * 4)
        human_v.append(hv)
        human_a.append(ha)
        human_d.append(hd)

    r_v = pearson(human_v, engine_v)
    r_a = pearson(human_a, engine_a)
    r_d = pearson(human_d, engine_d)

    # Composite: weighted average (V matters most for comparison)
    composite = r_v * 0.5 + r_a * 0.25 + r_d * 0.25

    return params, r_v, r_a, r_d, composite


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def random_organism():
    return [random.uniform(mn, mx) for mn, mx in zip(PARAM_MINS, PARAM_MAXS)]

def crossover(a, b):
    point = random.randint(1, N_PARAMS - 1)
    return a[:point] + b[point:]

def mutate(org, rate=0.15):
    m = list(org)
    for i in range(N_PARAMS):
        if random.random() < rate:
            span = PARAM_MAXS[i] - PARAM_MINS[i]
            delta = random.gauss(0, span * 0.1)
            m[i] = max(PARAM_MINS[i], min(PARAM_MAXS[i], m[i] + delta))
    return m


def run_evolution(data, n_pop=80, n_gen=100, n_workers=8):
    # Seed with defaults + previous champion
    champion_prev = [
        0.903, 2.894, 0.085, 158.7, 68.9, 95.5, 0.529,
        1.920, 0.789, 2.901, 0.30, 0.707,
        116.7, 155.2,
        0.850, 0.60, 0.523,
        0.704, 0.339, 0.319, -21.6, 1.152, 2.313, -6.0, 0.825, 9.6, -7.8,
    ]
    population = [list(PARAM_DEFAULTS), champion_prev]
    population += [random_organism() for _ in range(n_pop - 2)]

    best_params = None
    best_composite = -1
    best_rv = best_ra = best_rd = 0
    stagnation = 0

    print(f"\n{'='*70}")
    print(f"  EMOBANK OPTIMIZER — Human Agreement Tuning")
    print(f"  {n_pop} organisms, {n_gen} generations, {len(data)} sentences")
    print(f"  Target: Pearson r > 0.7 on V, > 0.5 on A/D")
    print(f"{'='*70}\n")

    t0 = time.time()

    for gen in range(n_gen):
        gt0 = time.time()

        # Evaluate
        work = [(org, data) for org in population]
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for res in pool.map(evaluate_organism, work):
                results.append(res)

        # Sort by composite correlation
        results.sort(key=lambda x: x[4], reverse=True)

        gen_best = results[0]
        if gen_best[4] > best_composite:
            best_params = gen_best[0]
            best_rv, best_ra, best_rd = gen_best[1], gen_best[2], gen_best[3]
            best_composite = gen_best[4]
            stagnation = 0
            elapsed = time.time() - gt0
            print(f"  Gen {gen:>4d}: composite={best_composite:.4f}  "
                  f"V={best_rv:.4f} A={best_ra:.4f} D={best_rd:.4f}  "
                  f"*** NEW BEST  ({elapsed:.1f}s)")
        else:
            stagnation += 1
            if gen % 10 == 0:
                elapsed = time.time() - gt0
                print(f"  Gen {gen:>4d}: best={best_composite:.4f}  stagnation={stagnation}  ({elapsed:.1f}s)")

        # Snapshot logging
        if gen > 0 and gen % 25 == 0:
            try:
                from benchmarks.experiment_tracker import ExperimentTracker
                tracker = ExperimentTracker()
                tracker.log_experiment(
                    theory=f"emobank-gen{gen:04d}",
                    engine="v2",
                    config=params_to_config(best_params),
                    results={"emobank_r_v": round(best_rv, 4),
                             "emobank_r_a": round(best_ra, 4),
                             "emobank_r_d": round(best_rd, 4),
                             "emobank_composite": round(best_composite, 4)},
                    notes=f"Gen {gen}/{n_gen}, stagnation={stagnation}",
                )
            except Exception:
                pass

        # Selection: top 20% survive
        survivors = [r[0] for r in results[:n_pop // 5]]

        # Breed next generation
        new_pop = list(survivors)
        while len(new_pop) < n_pop:
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    total_time = time.time() - t0

    # Save champion
    output = {
        "champion_composite": round(best_composite, 4),
        "champion_r_v": round(best_rv, 4),
        "champion_r_a": round(best_ra, 4),
        "champion_r_d": round(best_rd, 4),
        "champion_config": params_to_config(best_params),
        "generations": n_gen,
        "sentences": len(data),
        "time_seconds": round(total_time),
    }

    out_path = os.path.join(os.path.dirname(__file__), "emobank_optimal.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CHAMPION: composite={best_composite:.4f}")
    print(f"  Valence:   r={best_rv:.4f}")
    print(f"  Arousal:   r={best_ra:.4f}")
    print(f"  Dominance: r={best_rd:.4f}")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Saved to {out_path}")
    print(f"{'='*70}")

    # Log final
    try:
        from benchmarks.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.log_experiment(
            theory="emobank-champion",
            engine="v2",
            config=params_to_config(best_params),
            results={"emobank_r_v": round(best_rv, 4),
                     "emobank_r_a": round(best_ra, 4),
                     "emobank_r_d": round(best_rd, 4),
                     "emobank_composite": round(best_composite, 4)},
            notes=f"Champion after {n_gen} gen, {total_time:.0f}s",
        )
    except Exception:
        pass

    return best_params, best_rv, best_ra, best_rd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    max_n = 1000 if args.quick else None
    data = load_emobank(max_n)
    print(f"  Loaded {len(data)} EmoBank sentences")

    run_evolution(data, n_pop=args.pop, n_gen=args.gen, n_workers=args.workers)


if __name__ == "__main__":
    main()
