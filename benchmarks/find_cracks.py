"""Find where the engine fails -- run trace on academic data, save all misses."""
import sys
import json

sys.path.insert(0, '.')
from demo.trace import PipelineTrace
from datasets import load_dataset

GOEMO_SENT = {
    0: 'positive', 1: 'positive', 2: 'negative', 3: 'negative',
    4: 'positive', 5: 'positive', 6: 'neutral', 7: 'neutral',
    8: 'positive', 9: 'negative', 10: 'negative', 11: 'negative',
    12: 'negative', 13: 'positive', 14: 'negative', 15: 'positive',
    16: 'negative', 17: 'positive', 18: 'positive', 19: 'negative',
    20: 'positive', 21: 'positive', 22: 'neutral', 23: 'positive',
    24: 'negative', 25: 'negative', 26: 'neutral', 27: 'neutral',
}

trace = PipelineTrace()

# Run on 500 GoEmotions examples
ge = load_dataset("google-research-datasets/go_emotions", split="test").select(range(500))
for row in ge:
    if not row['labels']:
        continue
    truth = GOEMO_SENT.get(row['labels'][0], 'neutral')
    trace.trace_sentence(row['text'], truth)

print(f"Total: {len(trace.misses)} misses out of 500")

# Categorize misses
false_pos = [m for m in trace.misses if m['prediction'] == 'positive' and m['truth'] == 'negative']
false_neg = [m for m in trace.misses if m['prediction'] == 'negative' and m['truth'] == 'positive']
false_neut = [m for m in trace.misses if m['prediction'] == 'neutral' and m['truth'] != 'neutral']

print(f"False positives: {len(false_pos)}")
print(f"False negatives: {len(false_neg)}")
print(f"False neutral: {len(false_neut)}")

# Print top 10 most informative misses
print("\n=== TOP MISSES ===")
for m in trace.misses[:10]:
    trace.print_trace(m)

# Save all misses
trace.save_misses('benchmarks/misses.json')
print(f"\nSaved {len(trace.misses)} misses to benchmarks/misses.json")
