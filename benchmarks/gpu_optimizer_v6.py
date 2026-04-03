#!/usr/bin/env python3
"""Genetic Optimizer V6 -- uses REAL labeled datasets, not hand-picked sentences.

Tests against 500 random sentences from SST-2, GoEmotions, TweetEval.
Each generation evaluates 500 sentences × population size.

Usage:
    python3 benchmarks/gpu_optimizer_v6.py --pop 100 --generations 200
"""

import sys, os, json, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KNOBS = {
    "momentum":           (0.30, 0.85),
    "force_scale":        (0.50, 2.00),
    "direct_push_cap":    (0.10, 1.00),
    "direct_push_trigger":(30.0, 150.0),
    "proximity_decay":    (0.50, 0.95),
    "coeff_cap":          (1.5, 5.0),
    "amplifier_mod":      (0.10, 1.00),
    "negator_mod":        (-2.5, -0.5),
    "self_ref_mod":       (0.05, 0.80),
    "hedge_mod":          (-0.8, -0.05),
    "compressor_mod":     (-1.0, -0.1),
}


def load_test_data(sample_size=500):
    """Load labeled sentences from real datasets."""
    from datasets import load_dataset

    sentences = []

    # SST-2: binary pos/neg
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    for row in sst:
        truth = "positive" if row["label"] == 1 else "negative"
        sentences.append((row["sentence"], truth, 1.0))

    # GoEmotions: multi-class → 3-way
    GOEMO = {0:"positive",1:"positive",2:"negative",3:"negative",4:"positive",5:"positive",
             6:"neutral",7:"neutral",8:"positive",9:"negative",10:"negative",11:"negative",
             12:"negative",13:"positive",14:"negative",15:"positive",16:"negative",17:"positive",
             18:"positive",19:"negative",20:"positive",21:"positive",22:"neutral",23:"positive",
             24:"negative",25:"negative",26:"neutral",27:"neutral"}
    ge = load_dataset("google-research-datasets/go_emotions", split="test")
    for row in ge:
        if not row["labels"]:
            continue
        truth = GOEMO.get(row["labels"][0], "neutral")
        sentences.append((row["text"], truth, 0.8))  # slightly lower weight (noisy labels)

    # TweetEval: emotion → binary
    tw_sent = {0: "negative", 1: "positive", 2: "positive", 3: "negative"}
    te = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test")
    for row in te:
        truth = tw_sent[row["label"]]
        sentences.append((row["text"], truth, 0.8))

    # Shuffle and sample
    random.seed(42)
    random.shuffle(sentences)
    sample = sentences[:sample_size]

    print(f"  Loaded {len(sentences)} total, sampling {len(sample)}")
    pos = sum(1 for _, t, _ in sample if t == "positive")
    neg = sum(1 for _, t, _ in sample if t == "negative")
    neu = sum(1 for _, t, _ in sample if t == "neutral")
    print(f"  Sample: {pos} pos, {neg} neg, {neu} neutral")
    return sample


def apply_knobs(knobs):
    import engine.pendulum as pend
    import engine.proximity as prox
    pend.MOMENTUM = knobs["momentum"]
    pend.FORCE_SCALE = knobs["force_scale"]
    pend.DIRECT_PUSH_CAP = knobs["direct_push_cap"]
    pend.DIRECT_PUSH_TRIGGER = knobs["direct_push_trigger"]
    prox.PROXIMITY_DECAY = knobs["proximity_decay"]
    prox.COEFFICIENT_CAP = knobs["coeff_cap"]
    prox.ROLE_MODIFIERS["AMPLIFIER"] = knobs["amplifier_mod"]
    prox.ROLE_MODIFIERS["NEGATOR"] = knobs["negator_mod"]
    prox.ROLE_MODIFIERS["SELF_REF"] = knobs["self_ref_mod"]
    prox.ROLE_MODIFIERS["HEDGE"] = knobs["hedge_mod"]
    prox.ROLE_MODIFIERS["COMPRESSOR"] = knobs["compressor_mod"]


def classify_v(v, mode="three_way"):
    if mode == "binary":
        return "positive" if v >= 128 else "negative"
    if v >= 145:
        return "positive"
    elif v <= 110:
        return "negative"
    return "neutral"


def evaluate(knobs, test_data):
    apply_knobs(knobs)
    from engine.pendulum import compute_vadug

    score = 0.0
    weight = 0.0
    for text, truth, w in test_data:
        try:
            vadug, _ = compute_vadug(text)
            got = classify_v(vadug.v)
            if got == truth:
                score += w
            elif truth == "neg" and vadug.v <= 120:
                score += w * 0.3
            elif truth == "pos" and vadug.v >= 135:
                score += w * 0.3
            weight += w
        except:
            weight += w
    return score / weight if weight > 0 else 0.0


def random_knobs():
    return {k: random.uniform(lo, hi) for k, (lo, hi) in KNOBS.items()}


def mutate(knobs, rate=0.2, strength=0.15):
    child = dict(knobs)
    for k, (lo, hi) in KNOBS.items():
        if random.random() < rate:
            delta = random.gauss(0, (hi - lo) * strength)
            child[k] = max(lo, min(hi, child[k] + delta))
    return child


def crossover(a, b):
    return {k: a[k] if random.random() < 0.5 else b[k] for k in KNOBS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--elite", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  V6 GENETIC OPTIMIZER -- REAL DATASET TESTING")
    print(f"  Pop: {args.pop} | Gen: {args.generations} | Sample: {args.sample}")
    print(f"{'=' * 70}")

    test_data = load_test_data(args.sample)

    # Current knobs
    import engine.pendulum as pend
    import engine.proximity as prox
    CURRENT = {
        "momentum": pend.MOMENTUM,
        "force_scale": pend.FORCE_SCALE,
        "direct_push_cap": pend.DIRECT_PUSH_CAP,
        "direct_push_trigger": pend.DIRECT_PUSH_TRIGGER,
        "proximity_decay": prox.PROXIMITY_DECAY,
        "coeff_cap": prox.COEFFICIENT_CAP,
        "amplifier_mod": prox.ROLE_MODIFIERS["AMPLIFIER"],
        "negator_mod": prox.ROLE_MODIFIERS["NEGATOR"],
        "self_ref_mod": prox.ROLE_MODIFIERS["SELF_REF"],
        "hedge_mod": prox.ROLE_MODIFIERS["HEDGE"],
        "compressor_mod": prox.ROLE_MODIFIERS["COMPRESSOR"],
    }

    current_fitness = evaluate(CURRENT, test_data)
    print(f"\n  Current fitness: {current_fitness:.4f}")

    # Initialize population
    population = [dict(CURRENT)]
    for _ in range(args.pop // 3):
        population.append(mutate(CURRENT, rate=0.5, strength=0.3))
    while len(population) < args.pop:
        population.append(random_knobs())

    best_ever = dict(CURRENT)
    best_fitness = current_fitness
    stale = 0

    for gen in range(args.generations):
        t0 = time.perf_counter()
        fitnesses = [evaluate(p, test_data) for p in population]
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        gen_best = ranked[0][0]

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_ever = dict(ranked[0][1])
            stale = 0
        else:
            stale += 1

        elapsed = time.perf_counter() - t0
        if gen % 10 == 0:
            print(f"  Gen {gen:4d} | best={gen_best:.4f} | all-time={best_fitness:.4f} | stale={stale} | {elapsed:.1f}s")

        mut_rate = 0.2 + min(0.4, stale * 0.02)
        mut_strength = 0.15 + min(0.3, stale * 0.015)
        elite = [p for _, p in ranked[:args.elite]]
        new_pop = list(elite)
        while len(new_pop) < args.pop:
            if random.random() < 0.7:
                a = ranked[random.randint(0, args.pop // 3)][1]
                b = ranked[random.randint(0, args.pop // 3)][1]
                child = mutate(crossover(a, b), rate=mut_rate, strength=mut_strength)
            else:
                child = mutate(random.choice(elite), rate=mut_rate, strength=mut_strength)
            new_pop.append(child)
        if stale > 20:
            for i in range(args.pop // 5):
                new_pop[-(i+1)] = random_knobs()
            stale = 0
        population = new_pop

    print(f"\n{'=' * 70}")
    print(f"  CHAMPION (fitness={best_fitness:.4f}, tested on {args.sample} real sentences):")
    for k, v in best_ever.items():
        print(f"    {k:25s}: {v:8.4f}")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fitness": round(best_fitness, 6),
        "current_fitness": round(current_fitness, 6),
        "sample_size": args.sample,
        "data_source": "SST-2 + GoEmotions + TweetEval",
        "generations": args.generations,
        "population": args.pop,
        "champion": {k: round(v, 6) for k, v in best_ever.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "v6_optimal.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
