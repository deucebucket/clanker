#!/usr/bin/env python3
"""Genetic Optimizer V7 — tunes against VERIFIED literary/game prose + conversational stress test.

Multi-objective: maximize literary accuracy WITHOUT regressing conversational accuracy.
Tunes both physics constants AND register detection thresholds.

Uses the 950+ verified sentences from datasets/verified_sentences.json
AND the 275 stress test sentences from benchmarks/stress_test.py.

Usage:
    python3 benchmarks/gpu_optimizer_v7.py --pop 80 --generations 300
    python3 benchmarks/gpu_optimizer_v7.py --pop 40 --generations 100 --quick
"""

import sys, os, json, time, random, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Tunable knobs: physics constants + register thresholds ──

KNOBS = {
    # Core physics (existing)
    "m_base":              (0.35, 0.75),
    "force_scale":         (0.80, 2.00),
    "direct_push_cap":     (0.50, 1.50),
    "direct_push_trigger": (50.0, 150.0),
    "saturation":          (80.0, 160.0),

    # Register detection (Council Round 7 — NEW)
    "lit_obs_threshold":   (0.20, 0.60),   # observation score threshold for LITERARY
    "lit_min_words":       (4, 15),         # minimum sentence length for LITERARY
    "lit_dampener":        (0.30, 0.70),    # base dampening for LITERARY register
    "lit_scatter_k":       (0.02, 0.10),    # mass-dependent scattering constant

    # Mundane dampening
    "mundane_alpha":       (0.02, 0.10),
    "mundane_epsilon":     (0.30, 2.00),

    # Static friction (Council Round 7 — NEW, currently disabled)
    "friction_threshold":  (15, 40),        # minimum |dV| to overcome friction
    "friction_enabled":    (0, 1),          # 0=off, 1=on (binary, rounded)

    # Sarcasm inversion
    "sarcasm_penalty":     (-25.0, -5.0),
}


def load_verified_data():
    """Load the hand-verified literary/game sentences."""
    with open("datasets/verified_sentences.json") as f:
        data = json.load(f)
    sentences = []
    for s in data["sentences"]:
        text = s["text"]
        human = s.get("human_label", "neutral")
        correct = s.get("correct", False)
        # Weight: correct sentences are validation, wrong ones are training targets
        sentences.append((text, human, 1.0))
    print(f"  Verified dataset: {len(sentences)} sentences")
    pos = sum(1 for _, h, _ in sentences if h == "pos")
    neg = sum(1 for _, h, _ in sentences if h == "neg")
    neu = sum(1 for _, h, _ in sentences if h == "neutral")
    print(f"  Distribution: {pos} pos, {neg} neg, {neu} neutral")
    return sentences


def load_stress_data():
    """Load the 275 conversational stress test sentences."""
    # Import the test data directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from stress_test import CATEGORIES
    sentences = []
    for cat, items in CATEGORIES.items():
        for text, expected in items:
            # Higher weight for conversational — must not regress
            sentences.append((text, expected, 2.0))
    print(f"  Stress test: {len(sentences)} sentences (weight 2.0x)")
    return sentences


def apply_knobs(knobs):
    """Apply knob values to the engine's global constants."""
    import engine.pendulum as pend
    import engine.proximity as prox

    pend.M_BASE = knobs["m_base"]
    pend.FORCE_SCALE = knobs["force_scale"]
    pend.DIRECT_PUSH_CAP = knobs["direct_push_cap"]
    pend.DIRECT_PUSH_TRIGGER = knobs["direct_push_trigger"]
    pend.SATURATION = knobs["saturation"]
    pend.MUNDANE_ALPHA = knobs["mundane_alpha"]
    pend.MUNDANE_EPSILON = knobs["mundane_epsilon"]

    # Sarcasm penalty is stored in a module-level variable we'll need to wire up
    # For now, store in a global that interpret_context can read
    pend._SARCASM_PENALTY_OVERRIDE = knobs["sarcasm_penalty"]

    # Register detection thresholds — these are harder to apply dynamically
    # since they're in interpret_context. We'll store as module globals.
    pend._LIT_OBS_THRESHOLD = knobs["lit_obs_threshold"]
    pend._LIT_MIN_WORDS = int(round(knobs["lit_min_words"]))
    pend._LIT_DAMPENER = knobs["lit_dampener"]
    pend._LIT_SCATTER_K = knobs["lit_scatter_k"]

    # Static friction
    pend.STATIC_FRICTION_THRESHOLD = knobs["friction_threshold"]
    # Enable/disable static friction in pipeline
    pend._FRICTION_ENABLED = knobs["friction_enabled"] >= 0.5


def classify_v(v):
    if v >= 145:
        return "pos"
    elif v < 110:
        return "neg"
    return "neutral"


def evaluate(knobs, verified_data, stress_data):
    """Multi-objective fitness: literary accuracy + conversational accuracy."""
    apply_knobs(knobs)
    from engine.pendulum import compute_vadug

    lit_score = 0.0
    lit_weight = 0.0
    conv_score = 0.0
    conv_weight = 0.0

    # Literary/game evaluation
    for text, truth, w in verified_data:
        try:
            vadug, _ = compute_vadug(text)
            got = classify_v(vadug.v)
            if got == truth:
                lit_score += w
            # Partial credit for near-misses
            elif truth == "neutral" and 100 <= vadug.v <= 155:
                lit_score += w * 0.3
            lit_weight += w
        except:
            lit_weight += w

    # Conversational evaluation (must not regress)
    for text, truth, w in stress_data:
        try:
            vadug, _ = compute_vadug(text)
            got = classify_v(vadug.v)
            if got == truth:
                conv_score += w
            conv_weight += w
        except:
            conv_weight += w

    lit_acc = lit_score / lit_weight if lit_weight > 0 else 0.0
    conv_acc = conv_score / conv_weight if conv_weight > 0 else 0.0

    # Multi-objective: conversational must stay >= 95%, literary is the optimization target
    # Heavy penalty if conversational drops below 95%
    if conv_acc < 0.95:
        penalty = (0.95 - conv_acc) * 5.0  # harsh penalty
        return lit_acc * 0.6 + conv_acc * 0.4 - penalty
    else:
        # Above 95% conv: optimize literary with conv as tiebreaker
        return lit_acc * 0.7 + conv_acc * 0.3


def random_knobs():
    result = {}
    for k, (lo, hi) in KNOBS.items():
        result[k] = random.uniform(lo, hi)
    return result


def mutate(knobs, rate=0.2, strength=0.15):
    child = dict(knobs)
    for k, (lo, hi) in KNOBS.items():
        if random.random() < rate:
            delta = random.gauss(0, (hi - lo) * strength)
            child[k] = max(lo, min(hi, child[k] + delta))
    return child


def crossover(a, b):
    return {k: a[k] if random.random() < 0.5 else b[k] for k in KNOBS}


def get_current_knobs():
    """Read current engine values as the starting point."""
    import engine.pendulum as pend
    return {
        "m_base": pend.M_BASE,
        "force_scale": pend.FORCE_SCALE,
        "direct_push_cap": pend.DIRECT_PUSH_CAP,
        "direct_push_trigger": pend.DIRECT_PUSH_TRIGGER,
        "saturation": pend.SATURATION,
        "mundane_alpha": pend.MUNDANE_ALPHA,
        "mundane_epsilon": pend.MUNDANE_EPSILON,
        "lit_obs_threshold": 0.50,   # current conservative value
        "lit_min_words": 10,
        "lit_dampener": 0.55,
        "lit_scatter_k": 0.05,
        "friction_threshold": 25,
        "friction_enabled": 0,       # currently disabled
        "sarcasm_penalty": -15.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.pop = 40
        args.generations = 50

    print(f"\n{'=' * 70}")
    print(f"  V7 GENETIC OPTIMIZER — LITERARY + CONVERSATIONAL MULTI-OBJECTIVE")
    print(f"  Pop: {args.pop} | Gen: {args.generations} | Knobs: {len(KNOBS)}")
    print(f"{'=' * 70}")

    verified_data = load_verified_data()
    stress_data = load_stress_data()

    # Sample verified data for speed (full dataset is slow per eval)
    if len(verified_data) > 500:
        random.seed(42)
        verified_sample = random.sample(verified_data, 500)
    else:
        verified_sample = verified_data

    current = get_current_knobs()
    current_fitness = evaluate(current, verified_sample, stress_data)
    print(f"\n  Current fitness: {current_fitness:.4f}")

    # Quick check: what's the current breakdown?
    apply_knobs(current)
    from engine.pendulum import compute_vadug
    lit_correct = sum(1 for text, truth, _ in verified_sample
                      if classify_v(compute_vadug(text)[0].v) == truth)
    conv_correct = sum(1 for text, truth, _ in stress_data
                       if classify_v(compute_vadug(text)[0].v) == truth)
    print(f"  Literary: {lit_correct}/{len(verified_sample)} ({100*lit_correct//len(verified_sample)}%)")
    print(f"  Conversational: {conv_correct}/{len(stress_data)} ({100*conv_correct//len(stress_data)}%)")

    # Initialize population: current + mutations + random
    population = [dict(current)]
    for _ in range(args.pop // 3):
        population.append(mutate(current, rate=0.5, strength=0.3))
    while len(population) < args.pop:
        population.append(random_knobs())

    best_ever = dict(current)
    best_fitness = current_fitness
    stale = 0

    for gen in range(args.generations):
        t0 = time.perf_counter()
        fitnesses = [evaluate(p, verified_sample, stress_data) for p in population]
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        gen_best = ranked[0][0]

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_ever = dict(ranked[0][1])
            stale = 0
        else:
            stale += 1

        elapsed = time.perf_counter() - t0

        if gen % 10 == 0 or gen_best > best_fitness - 0.001:
            print(f"  Gen {gen:3d} | best={gen_best:.4f} | all-time={best_fitness:.4f} | "
                  f"stale={stale} | {elapsed:.1f}s")

        # Adaptive mutation: increase when stale
        mut_rate = 0.2 + 0.3 * min(stale / 30, 1.0)
        mut_strength = 0.15 + 0.15 * min(stale / 30, 1.0)

        # Selection + reproduction
        elite = [p for _, p in ranked[:args.elite]]
        new_pop = list(elite)

        while len(new_pop) < args.pop:
            if random.random() < 0.7:
                # Crossover + mutate
                a, b = random.choices(elite, k=2)
                child = mutate(crossover(a, b), rate=mut_rate, strength=mut_strength)
            else:
                # Pure mutation of random elite
                parent = random.choice(elite)
                child = mutate(parent, rate=mut_rate, strength=mut_strength)
            new_pop.append(child)

        population = new_pop

        # Reset if very stale
        if stale >= 40:
            print(f"  ** Injecting fresh blood (stale={stale})")
            for i in range(args.pop // 4, args.pop):
                population[i] = mutate(best_ever, rate=0.8, strength=0.5)
            stale = 0

    # Final evaluation with full dataset
    print(f"\n{'=' * 70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best fitness: {best_fitness:.4f}")

    # Show champion knobs
    print(f"\n  Champion knobs:")
    for k, v in sorted(best_ever.items()):
        cur = current.get(k, "?")
        changed = " *" if abs(v - cur) > 0.01 else ""
        print(f"    {k:25s} = {v:8.4f}  (was {cur}){changed}")

    # Final accuracy check
    apply_knobs(best_ever)
    lit_correct = sum(1 for text, truth, _ in verified_data
                      if classify_v(compute_vadug(text)[0].v) == truth)
    conv_correct = sum(1 for text, truth, _ in stress_data
                       if classify_v(compute_vadug(text)[0].v) == truth)
    print(f"\n  Final literary:       {lit_correct}/{len(verified_data)} ({100*lit_correct//len(verified_data)}%)")
    print(f"  Final conversational: {conv_correct}/{len(stress_data)} ({100*conv_correct//len(stress_data)}%)")

    # Save champion
    out_path = os.path.join(os.path.dirname(__file__), "champion_v7.json")
    with open(out_path, "w") as f:
        json.dump({
            "knobs": best_ever,
            "fitness": best_fitness,
            "literary_accuracy": lit_correct / len(verified_data),
            "conversational_accuracy": conv_correct / len(stress_data),
            "verified_sentences": len(verified_data),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\n  Saved champion to {out_path}")


if __name__ == "__main__":
    main()
