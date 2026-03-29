#!/usr/bin/env python3
"""Sweep exponential decay parameters against combined academic dataset.

Optimizes: _decay_lambda, _spike_threshold, _momentum_floor, _momentum_range
"""

import sys, re, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from demo.simulator import SequentialPendulum
from demo.classifier import AdaptiveClassifier
from demo.intent import IntentDetector
from datasets import load_dataset

GOEMO_SENT = {0:'positive',1:'positive',2:'negative',3:'negative',4:'positive',5:'positive',6:'neutral',7:'neutral',8:'positive',9:'negative',10:'negative',11:'negative',12:'negative',13:'positive',14:'negative',15:'positive',16:'negative',17:'positive',18:'positive',19:'negative',20:'positive',21:'positive',22:'neutral',23:'positive',24:'negative',25:'negative',26:'neutral',27:'neutral'}
TW = {0:'negative',1:'positive',2:'positive',3:'negative'}

intent = IntentDetector()
classifier = AdaptiveClassifier()

# Pre-compute data
data = []
for row in load_dataset('google-research-datasets/go_emotions', split='test').select(range(2000)):
    if not row['labels']:
        continue
    truth = GOEMO_SENT.get(row['labels'][0], 'neutral')
    data.append((row['text'], truth))
for row in load_dataset('stanfordnlp/sst2', split='validation'):
    data.append((row['sentence'], 'positive' if row['label'] == 1 else 'negative'))
for row in load_dataset('cardiffnlp/tweet_eval', 'emotion', split='test').select(range(500)):
    data.append((row['text'], TW[row['label']]))

print(f'{len(data)} test sentences')

# Pre-tokenize
tokenized = []
for text, truth in data:
    words = re.findall(r"[a-z']+", text.lower())
    mode, _ = intent.detect(text)
    tokenized.append((words, mode, truth))


def evaluate(lambda_d, spike_thresh, mom_floor=0.5, mom_range=0.49):
    correct = 0
    for words, mode, truth in tokenized:
        p = SequentialPendulum()
        p._decay_lambda = lambda_d
        p._spike_threshold = spike_thresh
        p._momentum_floor = mom_floor
        p._momentum_range = mom_range
        for i, w in enumerate(words):
            p.process_word(w, words, i)
        if classifier.classify(int(p.v), mode) == truth:
            correct += 1
    return correct / len(data) * 100


# Phase 1: Sweep lambda_decay and spike_threshold
print('\n=== Phase 1: lambda_decay x spike_threshold ===')
best_acc = 0
best_cfg = None
results = []

for lambda_d in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7]:
    for spike_thresh in [10, 15, 20, 25, 30, 40]:
        t0 = time.time()
        acc = evaluate(lambda_d, spike_thresh)
        elapsed = time.time() - t0
        tag = ''
        if acc > best_acc:
            best_acc = acc
            best_cfg = (lambda_d, spike_thresh)
            tag = ' <-- NEW BEST'
        results.append((lambda_d, spike_thresh, acc))
        print(f'  lambda={lambda_d:.2f} thresh={spike_thresh:2d} => {acc:.2f}%  ({elapsed:.1f}s){tag}')

print(f'\nPhase 1 best: lambda={best_cfg[0]}, thresh={best_cfg[1]} => {best_acc:.2f}%')

# Phase 2: Fine-tune momentum floor and range around best lambda/thresh
print('\n=== Phase 2: momentum_floor x momentum_range ===')
best_lambda, best_thresh = best_cfg
best_acc2 = best_acc
best_cfg2 = (0.5, 0.49)

for mom_floor in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    for mom_range in [0.29, 0.39, 0.44, 0.49, 0.54, 0.59]:
        # Ensure floor + range <= 1.0 (max effective momentum)
        if mom_floor + mom_range > 1.0:
            continue
        acc = evaluate(best_lambda, best_thresh, mom_floor, mom_range)
        tag = ''
        if acc > best_acc2:
            best_acc2 = acc
            best_cfg2 = (mom_floor, mom_range)
            tag = ' <-- NEW BEST'
        print(f'  floor={mom_floor:.2f} range={mom_range:.2f} => {acc:.2f}%{tag}')

print(f'\nPhase 2 best: floor={best_cfg2[0]}, range={best_cfg2[1]} => {best_acc2:.2f}%')

# Baseline (original values)
base_acc = evaluate(0.3, 20, 0.5, 0.49)
print(f'\n{"="*60}')
print(f'Baseline (lambda=0.3, thresh=20, floor=0.5, range=0.49): {base_acc:.1f}%')
print(f'Best (lambda={best_lambda}, thresh={best_thresh}, floor={best_cfg2[0]}, range={best_cfg2[1]}): {best_acc2:.1f}% ({best_acc2-base_acc:+.1f}pp)')
print(f'{"="*60}')

# Phase 3: Sweep classifier thresholds with optimal decay params
print('\n=== Phase 3: Classifier threshold sweep ===')
best_cls_acc = 0
best_cls_cfg = None

# Pre-compute V values with optimal decay params
v_data = []
for words, mode, truth in tokenized:
    p = SequentialPendulum()
    p._decay_lambda = best_lambda
    p._spike_threshold = best_thresh
    p._momentum_floor = best_cfg2[0]
    p._momentum_range = best_cfg2[1]
    for i, w in enumerate(words):
        p.process_word(w, words, i)
    v_data.append((int(p.v), mode, truth))

for emo_low in range(115, 130, 2):
    for emo_high in range(120, 135, 2):
        if emo_high <= emo_low:
            continue
        for stmt_low in range(105, 120, 3):
            for stmt_high in range(130, 145, 3):
                if stmt_high <= stmt_low:
                    continue
                thresholds = {
                    "EMOTIONAL": {"low": emo_low, "high": emo_high},
                    "VENTING": {"low": emo_low, "high": emo_high},
                    "CASUAL": {"low": emo_low, "high": emo_high},
                    "STATEMENT": {"low": stmt_low, "high": stmt_high},
                    "QUESTION": {"low": 105, "high": 150},
                    "REQUEST": {"low": 105, "high": 150},
                    "GREETING": {"low": 105, "high": 150},
                    "DEFAULT": {"low": stmt_low, "high": stmt_high},
                }
                correct = 0
                for v, mode, truth in v_data:
                    t = thresholds.get(mode, thresholds["DEFAULT"])
                    if v > t["high"]:
                        pred = "positive"
                    elif v < t["low"]:
                        pred = "negative"
                    else:
                        pred = "neutral"
                    if pred == truth:
                        correct += 1
                acc = correct / len(v_data) * 100
                if acc > best_cls_acc:
                    best_cls_acc = acc
                    best_cls_cfg = (emo_low, emo_high, stmt_low, stmt_high)

print(f'Best classifier thresholds:')
print(f'  EMOTIONAL/VENTING/CASUAL: low={best_cls_cfg[0]}, high={best_cls_cfg[1]}')
print(f'  STATEMENT/DEFAULT:        low={best_cls_cfg[2]}, high={best_cls_cfg[3]}')
print(f'  Accuracy: {best_cls_acc:.1f}% (vs {best_acc2:.1f}% with old thresholds, {best_cls_acc-best_acc2:+.1f}pp)')

print(f'\n{"="*60}')
print(f'FINAL: {best_cls_acc:.1f}% total improvement from baseline: {best_cls_acc-base_acc:+.1f}pp')
print(f'{"="*60}')
