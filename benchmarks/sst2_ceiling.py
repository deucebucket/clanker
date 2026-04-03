#!/usr/bin/env python3
"""Find SST-2 ceiling: what accuracy can the engine reach with domain-tuned knobs?"""
import sys, os, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

sst = load_dataset("stanfordnlp/sst2", split="validation")
test_data = [(row["sentence"], "positive" if row["label"] == 1 else "negative") for row in sst]
print(f"SST-2: {len(test_data)} sentences")

KNOBS = {
    "momentum": (0.30, 0.95), "force_scale": (0.50, 3.00),
    "direct_push_cap": (0.10, 1.00), "direct_push_trigger": (20.0, 200.0),
    "proximity_decay": (0.40, 0.95), "coeff_cap": (1.0, 5.0),
    "amplifier_mod": (0.10, 1.00), "negator_mod": (-2.5, -0.5),
    "self_ref_mod": (0.05, 0.80), "hedge_mod": (-0.8, -0.05),
    "compressor_mod": (-1.0, -0.1),
}

def apply_knobs(knobs):
    import engine.pendulum as p
    import engine.proximity as pr
    p.MOMENTUM = knobs["momentum"]; p.FORCE_SCALE = knobs["force_scale"]
    p.DIRECT_PUSH_CAP = knobs["direct_push_cap"]; p.DIRECT_PUSH_TRIGGER = knobs["direct_push_trigger"]
    pr.PROXIMITY_DECAY = knobs["proximity_decay"]; pr.COEFFICIENT_CAP = knobs["coeff_cap"]
    for k in ["amplifier_mod", "negator_mod", "self_ref_mod", "hedge_mod", "compressor_mod"]:
        role = k.replace("_mod", "").upper()
        pr.ROLE_MODIFIERS[role] = knobs[k]

def evaluate(knobs):
    apply_knobs(knobs)
    from engine.pendulum import compute_vadug
    correct = 0
    for text, truth in test_data:
        try:
            v, _ = compute_vadug(text)
            pred = "positive" if v.v >= 128 else "negative"
            if pred == truth: correct += 1
        except: pass
    return correct / len(test_data)

def random_knobs():
    return {k: random.uniform(lo, hi) for k, (lo, hi) in KNOBS.items()}

def mutate(k, rate=0.2, strength=0.15):
    c = dict(k)
    for key, (lo, hi) in KNOBS.items():
        if random.random() < rate:
            c[key] = max(lo, min(hi, c[key] + random.gauss(0, (hi-lo)*strength)))
    return c

def crossover(a, b):
    return {k: a[k] if random.random() < 0.5 else b[k] for k in KNOBS}

import engine.pendulum as p
import engine.proximity as pr
CURRENT = {
    "momentum": p.MOMENTUM, "force_scale": p.FORCE_SCALE,
    "direct_push_cap": p.DIRECT_PUSH_CAP, "direct_push_trigger": p.DIRECT_PUSH_TRIGGER,
    "proximity_decay": pr.PROXIMITY_DECAY, "coeff_cap": pr.COEFFICIENT_CAP,
    "amplifier_mod": pr.ROLE_MODIFIERS["AMPLIFIER"], "negator_mod": pr.ROLE_MODIFIERS["NEGATOR"],
    "self_ref_mod": pr.ROLE_MODIFIERS["SELF_REF"], "hedge_mod": pr.ROLE_MODIFIERS["HEDGE"],
    "compressor_mod": pr.ROLE_MODIFIERS["COMPRESSOR"],
}

cur_fit = evaluate(CURRENT)
print(f"Current SST-2: {cur_fit*100:.1f}%")

pop = [dict(CURRENT)]
for _ in range(50): pop.append(mutate(CURRENT, 0.5, 0.3))
while len(pop) < 150: pop.append(random_knobs())

best_fit = cur_fit
best = dict(CURRENT)

for gen in range(200):
    t0 = time.perf_counter()
    fits = [evaluate(p) for p in pop]
    ranked = sorted(zip(fits, pop), key=lambda x: -x[0])
    if ranked[0][0] > best_fit:
        best_fit = ranked[0][0]
        best = dict(ranked[0][1])
    elapsed = time.perf_counter() - t0
    if gen % 20 == 0:
        print(f"  Gen {gen:3d} | best={ranked[0][0]*100:.1f}% | all-time={best_fit*100:.1f}% | {elapsed:.1f}s")
    elite = [p for _, p in ranked[:10]]
    new = list(elite)
    while len(new) < 150:
        if random.random() < 0.7:
            a = ranked[random.randint(0, 30)][1]
            b = ranked[random.randint(0, 30)][1]
            new.append(mutate(crossover(a, b)))
        else:
            new.append(mutate(random.choice(elite)))
    pop = new

print(f"\nSST-2 CEILING: {best_fit*100:.1f}%")

with open(os.path.join(os.path.dirname(__file__), "sst2_ceiling.json"), "w") as f:
    json.dump({"sst2_accuracy": round(best_fit*100, 1), "champion": {k: round(v, 6) for k, v in best.items()}}, f, indent=2)
print("Saved to benchmarks/sst2_ceiling.json")
