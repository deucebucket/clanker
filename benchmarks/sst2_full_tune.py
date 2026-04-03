#!/usr/bin/env python3
"""SST-2 full tune: optimize knobs on TRAIN split, test on VALIDATION split.

67,349 train sentences for tuning. 872 validation for reporting.
Uses a random sample of train for speed (configurable).
"""
import sys, os, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

print("Loading SST-2...", flush=True)
sst_train = load_dataset("stanfordnlp/sst2", split="train")
sst_val = load_dataset("stanfordnlp/sst2", split="validation")

train_data = [(row["sentence"], "positive" if row["label"] == 1 else "negative") for row in sst_train]
val_data = [(row["sentence"], "positive" if row["label"] == 1 else "negative") for row in sst_val]
print(f"Train: {len(train_data)}, Validation: {len(val_data)}", flush=True)

# Use 2000 random train sentences per evaluation (speed vs coverage tradeoff)
random.seed(42)
train_sample = random.sample(train_data, 2000)
print(f"Train sample: {len(train_sample)}", flush=True)

KNOBS = {
    "momentum": (0.30, 0.95), "force_scale": (0.50, 3.00),
    "direct_push_cap": (0.10, 1.00), "direct_push_trigger": (20.0, 200.0),
    "proximity_decay": (0.40, 0.95), "coeff_cap": (1.0, 5.0),
    "amplifier_mod": (0.10, 1.00), "negator_mod": (-2.5, -0.5),
    "self_ref_mod": (0.05, 0.80), "hedge_mod": (-0.8, -0.05),
    "compressor_mod": (-1.0, -0.1),
}

def apply_knobs(knobs):
    import engine.pendulum as p; import engine.proximity as pr
    p.MOMENTUM=knobs["momentum"]; p.FORCE_SCALE=knobs["force_scale"]
    p.DIRECT_PUSH_CAP=knobs["direct_push_cap"]; p.DIRECT_PUSH_TRIGGER=knobs["direct_push_trigger"]
    pr.PROXIMITY_DECAY=knobs["proximity_decay"]; pr.COEFFICIENT_CAP=knobs["coeff_cap"]
    pr.ROLE_MODIFIERS["AMPLIFIER"]=knobs["amplifier_mod"]; pr.ROLE_MODIFIERS["NEGATOR"]=knobs["negator_mod"]
    pr.ROLE_MODIFIERS["SELF_REF"]=knobs["self_ref_mod"]; pr.ROLE_MODIFIERS["HEDGE"]=knobs["hedge_mod"]
    pr.ROLE_MODIFIERS["COMPRESSOR"]=knobs["compressor_mod"]

def evaluate(knobs, data):
    apply_knobs(knobs)
    from engine.pendulum import compute_vadug
    correct = 0
    for text, truth in data:
        try:
            v, _ = compute_vadug(text)
            if ("positive" if v.v >= 128 else "negative") == truth: correct += 1
        except: pass
    return correct / len(data)

def rand(): return {k: random.uniform(lo, hi) for k, (lo, hi) in KNOBS.items()}
def mut(k, rate=0.25, strength=0.2):
    c = dict(k)
    for key, (lo, hi) in KNOBS.items():
        if random.random() < rate: c[key] = max(lo, min(hi, c[key] + random.gauss(0, (hi-lo)*strength)))
    return c

import engine.pendulum as p; import engine.proximity as pr
CUR = {"momentum":p.MOMENTUM,"force_scale":p.FORCE_SCALE,"direct_push_cap":p.DIRECT_PUSH_CAP,
       "direct_push_trigger":p.DIRECT_PUSH_TRIGGER,"proximity_decay":pr.PROXIMITY_DECAY,
       "coeff_cap":pr.COEFFICIENT_CAP,"amplifier_mod":pr.ROLE_MODIFIERS["AMPLIFIER"],
       "negator_mod":pr.ROLE_MODIFIERS["NEGATOR"],"self_ref_mod":pr.ROLE_MODIFIERS["SELF_REF"],
       "hedge_mod":pr.ROLE_MODIFIERS["HEDGE"],"compressor_mod":pr.ROLE_MODIFIERS["COMPRESSOR"]}

# Evaluate current on both splits
cur_train = evaluate(CUR, train_sample)
cur_val = evaluate(CUR, val_data)
print(f"Current: train={cur_train*100:.1f}%, val={cur_val*100:.1f}%", flush=True)

# Evolve on train sample
pop = [dict(CUR)] + [mut(CUR) for _ in range(49)] + [rand() for _ in range(50)]
best_f = cur_train; best = dict(CUR)

for gen in range(100):
    t0 = time.perf_counter()
    fits = [evaluate(p, train_sample) for p in pop]
    ranked = sorted(zip(fits, pop), key=lambda x: -x[0])
    if ranked[0][0] > best_f: best_f = ranked[0][0]; best = dict(ranked[0][1])
    el = time.perf_counter() - t0

    # Every 10 gen, also check val to monitor overfitting
    if gen % 10 == 0:
        val_score = evaluate(best, val_data)
        print(f"Gen {gen:3d} | train={ranked[0][0]*100:.1f}% | val={val_score*100:.1f}% | best_train={best_f*100:.1f}% | {el:.1f}s", flush=True)

    elite = [p for _, p in ranked[:10]]
    new = list(elite)
    while len(new) < 100:
        a = ranked[random.randint(0,20)][1]; b = ranked[random.randint(0,20)][1]
        new.append(mut({k: a[k] if random.random()<0.5 else b[k] for k in KNOBS}))
    pop = new

# Final evaluation on FULL validation
final_val = evaluate(best, val_data)
final_train_full = evaluate(best, random.sample(train_data, 5000))  # bigger sample
print(f"\nFINAL: train(5K)={final_train_full*100:.1f}%, val={final_val*100:.1f}%", flush=True)
print(f"Champion knobs: {json.dumps({k:round(v,4) for k,v in best.items()})}", flush=True)

json.dump({
    "train_accuracy": round(final_train_full*100, 1),
    "val_accuracy": round(final_val*100, 1),
    "train_sample_size": 2000,
    "val_size": len(val_data),
    "champion": {k: round(v, 6) for k, v in best.items()},
}, open(os.path.join(os.path.dirname(__file__), "sst2_full_tune.json"), "w"), indent=2)
print("Saved to benchmarks/sst2_full_tune.json", flush=True)
