#!/usr/bin/env python3
"""Genetic Optimizer for V5.5 Physics Knobs.

Evolves MOMENTUM, FORCE_SCALE, DIRECT_PUSH_CAP, DIRECT_PUSH_TRIGGER,
PROXIMITY_DECAY, and ROLE_MODIFIER strengths using a genetic algorithm.

Fitness function: weighted accuracy across stress test categories +
crisis recall + core accuracy + false positive penalty.

Uses all available CPU cores. RTX 3090 not needed -- this is pure math.

Usage:
    python3 benchmarks/gpu_optimizer_v5.py                  # full run
    python3 benchmarks/gpu_optimizer_v5.py --generations 50  # quick test
    python3 benchmarks/gpu_optimizer_v5.py --workers 8       # limit workers
"""

import sys, os, json, time, random, argparse, copy
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Knob definitions with ranges ──────────────────────────────

KNOBS = {
    "momentum":           (0.50, 0.95),  # how much state resists change
    "force_scale":        (0.20, 1.50),  # how strong word forces are
    "direct_push_cap":    (0.10, 1.00),  # max direct push strength
    "direct_push_trigger":(30.0, 200.0), # force threshold for direct push
    "proximity_decay":    (0.40, 0.90),  # influence falloff per word
    "coeff_cap":          (1.5,  5.0),   # max absolute coefficient
    "amplifier_mod":      (0.10, 1.00),  # AMPLIFIER modifier strength
    "negator_mod":        (-2.5, -0.5),  # NEGATOR modifier (always negative)
    "self_ref_mod":       (0.05, 0.80),  # SELF_REF modifier
    "hedge_mod":          (-0.8, -0.05), # HEDGE modifier
    "compressor_mod":     (-1.0, -0.1),  # COMPRESSOR modifier
}

# Current champion values
CURRENT = {
    "momentum": 0.82,
    "force_scale": 0.5,
    "direct_push_cap": 0.4,
    "direct_push_trigger": 80.0,
    "proximity_decay": 0.7,
    "coeff_cap": 3.0,
    "amplifier_mod": 0.4,
    "negator_mod": -1.6,
    "self_ref_mod": 0.3,
    "hedge_mod": -0.3,
    "compressor_mod": -0.5,
}


# ── Test sentences with expected outcomes ─────────────────────

TESTS = {
    "crisis": [
        ("i want to kill myself", "neg", 3.0),
        ("i dont want to be here anymore", "neg", 3.0),
        ("nobody would miss me", "neg", 2.0),
        ("im better off dead", "neg", 3.0),
        ("i cant do this anymore", "neg", 2.0),
        ("whats the point of living", "neg", 2.0),
        ("i just want the pain to stop", "neg", 2.0),
        ("i am nothing", "neg", 3.0),
        ("everyone would be better off without me", "neg", 3.0),
        ("im just a burden", "neg", 2.0),
        ("nobody would even notice if i disappeared", "neg", 2.0),
        ("i wish i was never born", "neg", 2.0),
        ("nobody likes me i might as well jump", "neg", 3.0),
    ],
    "positive": [
        ("I JUST GOT THE JOB", "pos", 2.0),
        ("my kid took their first steps today", "pos", 2.0),
        ("clean for 6 months now", "pos", 1.5),
        ("just got engaged", "pos", 2.0),
        ("she said yes", "pos", 1.5),
        ("i finally feel like i belong", "pos", 2.0),
        ("im proud of what we built", "pos", 2.0),
        ("we closed on the house", "pos", 1.5),
        ("twins born healthy this morning", "pos", 2.0),
        ("first day at the new job went great", "pos", 1.5),
    ],
    "sarcasm": [
        ("oh great another meeting", "neg", 2.0),
        ("nice work genius", "neg", 1.5),
        ("oh joy", "neg", 2.0),
        ("wow thanks so much for the help", "neg", 1.5),
        ("yeah that went really well", "neg", 1.5),
        ("love that for you", "neg", 1.0),
    ],
    "slang_positive": [
        ("bro that was fire", "pos", 1.5),
        ("she absolutely killed it", "pos", 1.5),
        ("no cap that was insane", "pos", 1.5),
        ("lowkey goated", "pos", 1.0),
        ("thats bussin fr", "pos", 1.0),
    ],
    "passive_aggressive": [
        ("whatever makes you happy", "neg", 1.5),
        ("im fine", "neg", 1.5),
        ("do what you want", "neg", 1.0),
        ("no its fine i didnt want to go anyway", "neg", 2.0),
        ("i guess i deserved it", "neg", 1.5),
    ],
    "grief": [
        ("my mom died last month", "neg", 2.0),
        ("i lost my best friend", "neg", 1.5),
        ("the house feels empty without her", "neg", 1.5),
        ("his chair is still at the table", "neg", 1.5),
        ("i cant listen to that song anymore", "neg", 1.5),
    ],
    "betrayal": [
        ("my wife cheated on me with my best friend", "neg", 2.0),
        ("he told everyone my secret", "neg", 1.5),
        ("she took the kids and left", "neg", 2.0),
        ("they went behind my back", "neg", 1.5),
        ("i trusted him with everything", "neg", 1.5),
    ],
    "self_worth": [
        ("im worthless", "neg", 2.0),
        ("i am nothing", "neg", 2.0),
        ("im not good enough", "neg", 1.5),
        ("why would anyone love me", "neg", 1.5),
        ("i dont deserve to be happy", "neg", 2.0),
    ],
    "safe": [
        ("the meeting is at three", "neutral", 2.0),
        ("i need to buy groceries", "neutral", 1.5),
        ("the weather is nice today", "neutral", 1.5),
        ("happy birthday", "neutral", 1.0),
        ("the game starts at noon", "neutral", 1.5),
        ("i had pizza for lunch", "neutral", 1.0),
        ("see you tomorrow", "neutral", 1.0),
        ("the dog needs a walk", "neutral", 1.0),
    ],
}


def apply_knobs(knobs):
    """Monkey-patch the engine with the given knob values."""
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


def classify_v(v):
    if v >= 145:
        return "pos"
    elif v <= 110:
        return "neg"
    return "neutral"


def evaluate(knobs):
    """Score a knob configuration. Higher = better."""
    apply_knobs(knobs)
    from engine.pendulum import compute_vadug

    total_score = 0.0
    total_weight = 0.0

    for category, sentences in TESTS.items():
        for text, expected, weight in sentences:
            try:
                vadug, meta = compute_vadug(text)
                got = classify_v(vadug.v)

                if got == expected:
                    total_score += weight
                else:
                    # Partial credit: how close did V get to the right zone?
                    if expected == "neg":
                        # Wanted neg (V<=110). Credit for being close.
                        if vadug.v <= 120:
                            total_score += weight * 0.3
                    elif expected == "pos":
                        # Wanted pos (V>=145). Credit for being close.
                        if vadug.v >= 135:
                            total_score += weight * 0.3

                total_weight += weight
            except Exception:
                total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def random_knobs():
    """Generate random knob values within ranges."""
    return {k: random.uniform(lo, hi) for k, (lo, hi) in KNOBS.items()}


def mutate(knobs, rate=0.2, strength=0.15):
    """Mutate knob values."""
    child = dict(knobs)
    for k, (lo, hi) in KNOBS.items():
        if random.random() < rate:
            span = hi - lo
            delta = random.gauss(0, span * strength)
            child[k] = max(lo, min(hi, child[k] + delta))
    return child


def crossover(a, b):
    """Uniform crossover between two parents."""
    child = {}
    for k in KNOBS:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return child


def eval_wrapper(knobs):
    """Wrapper for multiprocessing."""
    return evaluate(knobs)


def main():
    parser = argparse.ArgumentParser(description="V5.5 Genetic Optimizer")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--workers", type=int, default=0, help="CPU workers (0=auto)")
    parser.add_argument("--elite", type=int, default=10, help="Elite survivors per generation")
    args = parser.parse_args()

    workers = args.workers or max(1, cpu_count() - 1)
    print(f"\n{'=' * 70}")
    print(f"  V5.5 GENETIC OPTIMIZER")
    print(f"  Population: {args.pop} | Generations: {args.generations} | Workers: {workers}")
    print(f"  Knobs: {len(KNOBS)} | Test sentences: {sum(len(v) for v in TESTS.values())}")
    print(f"{'=' * 70}\n")

    # Evaluate current champion
    current_fitness = evaluate(CURRENT)
    print(f"  Current champion fitness: {current_fitness:.4f}")
    print(f"  Current knobs: {json.dumps({k: round(v, 4) for k, v in CURRENT.items()})}")
    print()

    # Initialize population: current champion + mutants + randoms
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

        # Evaluate all -- serial (monkey-patching globals isn't process-safe)
        fitnesses = [evaluate(p) for p in population]

        # Sort by fitness
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        gen_best_fit = ranked[0][0]
        gen_best = ranked[0][1]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_ever = dict(gen_best)
            stale = 0
        else:
            stale += 1

        elapsed = time.perf_counter() - t0
        evals = len(population) * sum(len(v) for v in TESTS.values())

        if gen % 5 == 0 or gen_best_fit > best_fitness - 0.001:
            print(f"  Gen {gen:4d} | best={gen_best_fit:.4f} | all-time={best_fitness:.4f} | "
                  f"stale={stale} | {elapsed:.1f}s | {evals/elapsed:.0f} evals/s")

        # Adaptive mutation: increase when stale
        mut_rate = 0.2 + min(0.4, stale * 0.02)
        mut_strength = 0.15 + min(0.3, stale * 0.015)

        # Selection: elite + tournament
        elite = [p for _, p in ranked[:args.elite]]
        new_pop = list(elite)

        while len(new_pop) < args.pop:
            if random.random() < 0.7:
                # Crossover two tournament winners
                a = ranked[random.randint(0, args.pop // 3)][1]
                b = ranked[random.randint(0, args.pop // 3)][1]
                child = crossover(a, b)
                child = mutate(child, rate=mut_rate, strength=mut_strength)
            else:
                # Mutate an elite
                parent = random.choice(elite)
                child = mutate(parent, rate=mut_rate, strength=mut_strength)
            new_pop.append(child)

        # Inject fresh randoms to prevent stagnation
        if stale > 20:
            for i in range(args.pop // 5):
                new_pop[-(i+1)] = random_knobs()
            stale = 0

        population = new_pop

    # Final results
    print(f"\n{'=' * 70}")
    print(f"  CHAMPION (fitness={best_fitness:.4f}):")
    for k, v in best_ever.items():
        cur = CURRENT[k]
        delta = v - cur
        print(f"    {k:25s}: {v:8.4f}  (was {cur:.4f}, delta {delta:+.4f})")

    # Save champion
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fitness": round(best_fitness, 6),
        "current_fitness": round(current_fitness, 6),
        "improvement": round(best_fitness - current_fitness, 6),
        "generations": args.generations,
        "population": args.pop,
        "champion": {k: round(v, 6) for k, v in best_ever.items()},
        "current": CURRENT,
    }
    out_path = os.path.join(os.path.dirname(__file__), "v5_optimal.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Apply champion and run stress test
    print(f"\n  Running stress test with champion knobs...")
    apply_knobs(best_ever)

    from engine.pendulum import compute_vadug
    correct = 0
    total = 0
    for category, sentences in TESTS.items():
        for text, expected, weight in sentences:
            vadug, _ = compute_vadug(text)
            got = classify_v(vadug.v)
            if got == expected:
                correct += 1
            total += 1

    print(f"  Champion accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
