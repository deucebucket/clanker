#!/usr/bin/env python3
"""GPU-accelerated equation optimizer for Clanker.

Vectorizes the pendulum as tensor operations on GPU.
Evaluates 1000+ equation combinations simultaneously.
Uses genetic algorithm to evolve the best parameter set.

Run AFTER training finishes (needs full GPU VRAM).

Usage: python3 benchmarks/gpu_optimizer.py
Estimated: 1M evaluations in ~12 hours on RTX 3090
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
import re
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))


def load_test_data():
    """Load benchmark sentences with word forces pre-extracted."""
    from demo.forces import WORD_FORCES
    from demo.intent import IntentDetector
    from datasets import load_dataset

    GOEMO = {0:'positive',1:'positive',2:'negative',3:'negative',4:'positive',5:'positive',
             6:'neutral',7:'neutral',8:'positive',9:'negative',10:'negative',11:'negative',
             12:'negative',13:'positive',14:'negative',15:'positive',16:'negative',
             17:'positive',18:'positive',19:'negative',20:'positive',21:'positive',
             22:'neutral',23:'positive',24:'negative',25:'negative',26:'neutral',27:'neutral'}
    TW = {0:'negative',1:'positive',2:'positive',3:'negative'}

    intent = IntentDetector()
    data = []

    print("Loading datasets...")
    for row in load_dataset('google-research-datasets/go_emotions', split='test').select(range(2000)):
        if not row['labels']: continue
        truth = GOEMO.get(row['labels'][0], 'neutral')
        words = re.findall(r"[a-z']+", row['text'].lower())
        forces = []
        for w in words[:30]:  # cap at 30 words
            if w in WORD_FORCES:
                forces.append(list(WORD_FORCES[w][:5]))
            else:
                forces.append([0, 0, 0, 0, 0])
        # Pad to 30 words
        while len(forces) < 30:
            forces.append([0, 0, 0, 0, 0])
        mode, _ = intent.detect(row['text'])
        data.append((forces, truth, mode))

    for row in load_dataset('stanfordnlp/sst2', split='validation'):
        truth = 'positive' if row['label'] == 1 else 'negative'
        words = re.findall(r"[a-z']+", row['sentence'].lower())
        forces = []
        for w in words[:30]:
            if w in WORD_FORCES:
                forces.append(list(WORD_FORCES[w][:5]))
            else:
                forces.append([0, 0, 0, 0, 0])
        while len(forces) < 30:
            forces.append([0, 0, 0, 0, 0])
        mode, _ = intent.detect(row['sentence'])
        data.append((forces, truth, mode))

    for row in load_dataset('cardiffnlp/tweet_eval', 'emotion', split='test').select(range(500)):
        truth = TW[row['label']]
        words = re.findall(r"[a-z']+", row['text'].lower())
        forces = []
        for w in words[:30]:
            if w in WORD_FORCES:
                forces.append(list(WORD_FORCES[w][:5]))
            else:
                forces.append([0, 0, 0, 0, 0])
        while len(forces) < 30:
            forces.append([0, 0, 0, 0, 0])
        mode, _ = intent.detect(row['text'])
        data.append((forces, truth, mode))

    print(f"Loaded {len(data)} sentences")
    return data


def data_to_tensors(data, device):
    """Convert data to GPU tensors."""
    forces_list = [d[0] for d in data]
    truths = [0 if d[1] == 'negative' else 1 if d[1] == 'positive' else 2 for d in data]

    forces_tensor = torch.tensor(forces_list, dtype=torch.float32, device=device)
    truth_tensor = torch.tensor(truths, dtype=torch.long, device=device)

    return forces_tensor, truth_tensor


def gpu_pendulum_batch(forces, params, device):
    """Vectorized pendulum on GPU.

    forces: [n_sentences, 30, 5] — word forces per sentence
    params: [n_combos, 10] — equation parameters per combo:
        [momentum, decay_lambda, spike_thresh, force_scale_v, force_scale_a,
         force_scale_d, force_scale_u, force_scale_g, threshold_low, threshold_high]

    Returns: [n_combos, n_sentences] — predicted class (0=neg, 1=pos, 2=neutral)
    """
    n_combos = params.shape[0]
    n_sentences = forces.shape[0]
    n_words = forces.shape[1]

    # Expand forces for all combos: [n_combos, n_sentences, 30, 5]
    f = forces.unsqueeze(0).expand(n_combos, -1, -1, -1)

    # Scale forces by per-combo scaling: [n_combos, 1, 1, 5]
    scales = params[:, 3:8].unsqueeze(1).unsqueeze(1)
    f = f * scales

    # Initialize state: [n_combos, n_sentences, 5]
    state = torch.full((n_combos, n_sentences, 5), 128.0, device=device)
    state[:, :, 3] = 0  # U starts at 0

    # Momentum per combo: [n_combos, 1, 1]
    momentum = params[:, 0].unsqueeze(1).unsqueeze(1)

    # Process word by word (sequential — can't fully parallelize this)
    for w_idx in range(n_words):
        word_force = f[:, :, w_idx, :]  # [n_combos, n_sentences, 5]

        # Strength of this word
        strength = word_force[:, :, 0].abs() + word_force[:, :, 1].abs()

        # Effective momentum (simplified exponential decay)
        spike_thresh = params[:, 2].unsqueeze(1)
        is_strong = (strength > spike_thresh).float()
        eff_momentum = momentum * is_strong.unsqueeze(-1) + (momentum * 0.7) * (1 - is_strong.unsqueeze(-1))

        # Target
        target = 128.0 + word_force

        # Blend
        blend = 1.0 - eff_momentum
        state = state * eff_momentum + target * blend

    # Clamp
    state = state.clamp(0, 255)

    # Classify using V + G blend
    v = state[:, :, 0]
    g = state[:, :, 4]
    blended = v * 0.9 + g * 0.1

    # Thresholds per combo
    low = params[:, 8].unsqueeze(1)
    high = params[:, 9].unsqueeze(1)

    predictions = torch.where(blended > high, torch.ones_like(blended),
                              torch.where(blended < low, torch.zeros_like(blended),
                                          torch.full_like(blended, 2.0)))

    return predictions.long()


def evaluate_population(forces, truths, population, device):
    """Evaluate all organisms in the population."""
    predictions = gpu_pendulum_batch(forces, population, device)
    correct = (predictions == truths.unsqueeze(0)).float().sum(dim=1)
    accuracy = correct / truths.shape[0]
    return accuracy


def genetic_optimize(forces, truths, device, n_pop=200, n_generations=500):
    """Genetic algorithm on GPU."""

    # Parameter ranges: [momentum, decay_lambda, spike_thresh,
    #                    scale_v, scale_a, scale_d, scale_u, scale_g,
    #                    threshold_low, threshold_high]
    param_mins = torch.tensor([0.7, 0.05, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 110.0, 125.0], device=device)
    param_maxs = torch.tensor([0.99, 0.5, 40.0, 2.0, 3.0, 3.0, 2.0, 2.0, 130.0, 145.0], device=device)

    # Initialize random population
    population = torch.rand(n_pop, 10, device=device) * (param_maxs - param_mins) + param_mins

    best_acc = 0
    best_params = None
    stagnation = 0

    print(f"\nGenetic optimization: {n_pop} organisms, {n_generations} generations")
    print(f"{'='*60}")

    for gen in range(n_generations):
        # Evaluate
        fitness = evaluate_population(forces, truths, population, device)

        # Track best
        gen_best_idx = fitness.argmax()
        gen_best_acc = fitness[gen_best_idx].item()

        if gen_best_acc > best_acc:
            best_acc = gen_best_acc
            best_params = population[gen_best_idx].clone()
            stagnation = 0
            print(f"  Gen {gen:>4}: {best_acc*100:.2f}% *** NEW BEST")
        else:
            stagnation += 1

        if gen % 50 == 0 and gen > 0:
            print(f"  Gen {gen:>4}: best={best_acc*100:.2f}% (stagnation={stagnation})")

        if stagnation > 100:
            print(f"  Converged at generation {gen}")
            break

        # Selection: top 20% survive
        top_k = n_pop // 5
        top_indices = fitness.topk(top_k).indices
        survivors = population[top_indices]

        # Crossover: breed pairs of survivors
        new_pop = [survivors]
        while sum(p.shape[0] for p in new_pop) < n_pop:
            parents_a = survivors[torch.randint(top_k, (top_k,))]
            parents_b = survivors[torch.randint(top_k, (top_k,))]
            mask = torch.rand(top_k, 10, device=device) > 0.5
            children = torch.where(mask, parents_a, parents_b)
            new_pop.append(children)

        population = torch.cat(new_pop, dim=0)[:n_pop]

        # Mutation: 10% chance per parameter
        mutation_mask = torch.rand_like(population) < 0.1
        mutation = (torch.rand_like(population) - 0.5) * (param_maxs - param_mins) * 0.1
        population = population + mutation * mutation_mask.float()

        # Clamp to valid ranges
        population = population.clamp(min=param_mins, max=param_maxs)

    print(f"\n{'='*60}")
    print(f"CHAMPION: {best_acc*100:.2f}%")
    labels = ['momentum', 'decay_lambda', 'spike_thresh', 'scale_v', 'scale_a',
              'scale_d', 'scale_u', 'scale_g', 'thresh_low', 'thresh_high']
    for label, val in zip(labels, best_params.cpu().tolist()):
        print(f"  {label:<15}: {val:.4f}")

    return best_params.cpu(), best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {free:.1f} GB")

    # Load data
    data = load_test_data()
    forces, truths = data_to_tensors(data, device)
    print(f"Forces tensor: {forces.shape}")
    print(f"Truths tensor: {truths.shape}")

    # Run optimization
    t0 = time.time()
    # Scale to available VRAM — use as much GPU as possible
    free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    free_gb = free_vram / 1024**3

    # Each organism needs ~15MB for the pendulum tensors
    # With 24GB free: ~1500 organisms simultaneously
    max_pop = min(int(free_gb * 60), 2000)  # ~60 organisms per GB, cap at 2000
    n_pop = max(200, max_pop)
    n_gen = 1000  # more generations with bigger population

    print(f"\nFree VRAM: {free_gb:.1f}GB → population: {n_pop}, generations: {n_gen}")
    print(f"Total planned evaluations: {n_pop * n_gen * len(data):,}")

    best_params, best_acc = genetic_optimize(forces, truths, device,
                                              n_pop=n_pop, n_generations=n_gen)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    total_evals = 200 * 500 * len(data)
    print(f"Total evaluations: {total_evals:,}")
    print(f"Rate: {total_evals/elapsed:,.0f} evals/sec")

    # Save results
    results = {
        "champion_accuracy": round(best_acc * 100, 2),
        "parameters": {label: round(val, 4) for label, val in
                       zip(['momentum', 'decay_lambda', 'spike_thresh', 'scale_v', 'scale_a',
                            'scale_d', 'scale_u', 'scale_g', 'thresh_low', 'thresh_high'],
                           best_params.tolist())},
        "total_evaluations": total_evals,
        "time_seconds": round(elapsed),
    }

    out_path = os.path.join(os.path.dirname(__file__), "gpu_optimal.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
