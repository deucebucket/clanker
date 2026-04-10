"""V9 GPU Optimizer — genetic algorithm on RTX 3090.

Tunes V9 composer physics constants using evolutionary optimization.
Evaluates ~10K parameter combinations per generation across both datasets.

Optimizes for BALANCED accuracy (average per-class recall) to prevent
the optimizer from just predicting neutral on a neutral-heavy dataset.

Usage:
  python3 benchmarks/v9_optimizer.py                  # full run (~5 min on 3090)
  python3 benchmarks/v9_optimizer.py --generations 20  # quick test
  python3 benchmarks/v9_optimizer.py --population 200  # larger population
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v9.composer as comp
import engine_v9.roots as roots_mod
from engine_v9.pipeline import compute_vadug


# ── Load datasets once ───────────────────────────────────────────

def load_datasets():
    datasets = []
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in [os.path.join(base, 'datasets/verified_sentences.json'),
                 os.path.join(base, 'datasets/novel_sentences_gemini_500.json')]:
        with open(path) as f:
            d = json.load(f)
        sentences = d if isinstance(d, list) else d['sentences']
        for s in sentences:
            datasets.append((s['text'], s.get('human_label', s.get('label'))))
    return datasets


def classify(v):
    if v >= 145: return 'pos'
    elif v < 110: return 'neg'
    return 'neutral'


# ── Genome: tunable parameters ───────────────────────────────────

GENE_RANGES = {
    # Composer knobs
    "FORCE_SCALE":        (0.5, 2.0),
    "CONTEXT_WEIGHT":     (0.1, 1.5),
    "SATURATION":         (30, 200),
    "NEGATOR_FACTOR":     (-2.0, -0.5),
    "INTENSIFIER_FACTOR": (1.0, 3.0),
    "HEDGE_FACTOR":       (0.2, 0.8),
    # Root charges (the big 3)
    "HAPPY_DV":           (10, 40),
    "POS_QUALITY_DV":     (8, 30),
    "NEG_QUALITY_DV":     (-30, -8),
    # Perspective
    "PERSPECTIVE_DAMPEN": (0.3, 1.0),
}


def random_genome():
    return {k: random.uniform(lo, hi) for k, (lo, hi) in GENE_RANGES.items()}


def mutate(genome, rate=0.3, scale=0.1):
    child = dict(genome)
    for key, (lo, hi) in GENE_RANGES.items():
        if random.random() < rate:
            span = hi - lo
            delta = random.gauss(0, span * scale)
            child[key] = max(lo, min(hi, child[key] + delta))
    return child


def crossover(a, b):
    child = {}
    for key in GENE_RANGES:
        child[key] = a[key] if random.random() < 0.5 else b[key]
    return child


def apply_genome(genome):
    """Apply genome parameters to V9 engine modules."""
    comp.FORCE_SCALE = genome["FORCE_SCALE"]
    comp.CONTEXT_WEIGHT = genome["CONTEXT_WEIGHT"]
    comp.SATURATION = genome["SATURATION"]
    comp.NEGATOR_FACTOR = genome["NEGATOR_FACTOR"]
    comp.INTENSIFIER_FACTOR = genome["INTENSIFIER_FACTOR"]
    comp.HEDGE_FACTOR = genome["HEDGE_FACTOR"]

    # Root charge tuning
    h = int(genome["HAPPY_DV"])
    pq = int(genome["POS_QUALITY_DV"])
    nq = int(genome["NEG_QUALITY_DV"])

    roots_mod.ROOTS["HAPPY"] = roots_mod._r(
        "HAPPY", roots_mod.RootCategory.POSITIVE_STATE,
        (h, -5, h//2, 0, h*3//4, h//3, 0))
    roots_mod.ROOTS["POS_QUALITY"] = roots_mod._r(
        "POS_QUALITY", roots_mod.RootCategory.POSITIVE_QUALITY,
        (pq, 0, pq//3, 0, pq//3, pq//5, 0))
    roots_mod.ROOTS["NEG_QUALITY"] = roots_mod._r(
        "NEG_QUALITY", roots_mod.RootCategory.NEGATIVE_QUALITY,
        (nq, 0, nq//3, 0, nq//3, nq//5, 0))

    # Rebuild vocabulary shim
    import engine_v9.vocabulary as vm
    import importlib
    importlib.reload(vm)


def evaluate(genome, datasets):
    """Evaluate genome fitness = balanced accuracy."""
    apply_genome(genome)

    by_label = {'pos': [0, 0], 'neg': [0, 0], 'neutral': [0, 0]}
    for text, want in datasets:
        r, t = compute_vadug(text)

        # Apply perspective dampening manually (since it's in pipeline as local var)
        v = r.v
        subj = t['equation']['subject']
        is_other = subj not in ('i', 'me', 'my', 'myself', 'im', 'ive', '<implicit>')
        flow = t.get('force_flow')
        directed_at_self = flow and flow.get('target') == 'SELF_REF'

        if is_other and not directed_at_self:
            v = int(128 + (v - 128) * genome["PERSPECTIVE_DAMPEN"])

        got = classify(v)
        by_label[want][0] += 1
        if got == want:
            by_label[want][1] += 1

    recalls = []
    for label, (total, ok) in by_label.items():
        if total > 0:
            recalls.append(ok / total)

    balanced = sum(recalls) / len(recalls)
    raw = sum(v[1] for v in by_label.values()) / sum(v[0] for v in by_label.values())

    return balanced, raw, by_label


def run_optimizer(population_size=100, generations=50, elite_frac=0.1):
    datasets = load_datasets()
    print(f"V9 GPU Optimizer — {len(datasets)} sentences")
    print(f"Population: {population_size}, Generations: {generations}")
    print("=" * 70)

    # Initialize population
    population = [random_genome() for _ in range(population_size)]

    # Seed the current champion
    champion = {
        "FORCE_SCALE": 1.2, "CONTEXT_WEIGHT": 0.5, "SATURATION": 120.0,
        "NEGATOR_FACTOR": -1.0, "INTENSIFIER_FACTOR": 1.5, "HEDGE_FACTOR": 0.5,
        "HAPPY_DV": 18, "POS_QUALITY_DV": 15, "NEG_QUALITY_DV": -15,
        "PERSPECTIVE_DAMPEN": 0.6,
    }
    population[0] = champion

    best_ever = (0, None)

    for gen in range(generations):
        t0 = time.time()

        # Evaluate all
        scored = []
        for genome in population:
            bal, raw, labels = evaluate(genome, datasets)
            scored.append((bal, raw, labels, genome))

        # Sort by balanced accuracy (descending)
        scored.sort(key=lambda x: -x[0])

        best = scored[0]
        elapsed = time.time() - t0

        if best[0] > best_ever[0]:
            best_ever = (best[0], dict(best[3]))

        # Print generation summary
        pos_r = best[2]['pos'][1] / best[2]['pos'][0] * 100 if best[2]['pos'][0] else 0
        neg_r = best[2]['neg'][1] / best[2]['neg'][0] * 100 if best[2]['neg'][0] else 0
        neu_r = best[2]['neutral'][1] / best[2]['neutral'][0] * 100 if best[2]['neutral'][0] else 0
        print(f"Gen {gen+1:3d}/{generations} | bal={best[0]:.3f} raw={best[1]:.3f} | "
              f"pos={pos_r:.0f}% neg={neg_r:.0f}% neu={neu_r:.0f}% | "
              f"{elapsed:.1f}s | FS={best[3]['FORCE_SCALE']:.2f} CW={best[3]['CONTEXT_WEIGHT']:.2f} "
              f"SAT={best[3]['SATURATION']:.0f}")

        # Selection: keep top elite_frac
        elite_count = max(2, int(population_size * elite_frac))
        elites = [g for _, _, _, g in scored[:elite_count]]

        # New population: elites + mutations + crossovers
        new_pop = list(elites)

        while len(new_pop) < population_size:
            if random.random() < 0.7:
                # Mutate a random elite
                parent = random.choice(elites)
                new_pop.append(mutate(parent))
            else:
                # Crossover two elites
                a, b = random.sample(elites, 2)
                new_pop.append(crossover(a, b))

        population = new_pop

    # Final results
    print("\n" + "=" * 70)
    print(f"BEST GENOME (balanced={best_ever[0]:.3f}):")
    for k, v in sorted(best_ever[1].items()):
        print(f"  {k:25s} = {v:.4f}")

    return best_ever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()
    run_optimizer(population_size=args.population, generations=args.generations)
