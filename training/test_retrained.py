#!/usr/bin/env python3
"""Test the retrained model on key sentences."""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from training.train import ClankerClassifier, BUCKET_NAMES, BUCKET_THRESHOLDS, NUM_BUCKETS
from transformers import GPT2Tokenizer

# Also run through engine + zone classifier for comparison
from demo.pendulum_v2 import PendulumV2
from demo.zones import ZoneClassifier

CHECKPOINT = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best")

TEST_SENTENCES = [
    "I am happy",
    "I am not happy",
    "Whatever I am fine",
    "I want to die",
    "Oh great another Monday",
]


def bucket_to_value(bucket_idx):
    """Convert bucket index to approximate VADUG value (midpoint of range)."""
    if bucket_idx == 0:
        return 18
    elif bucket_idx == 6:
        return 237
    else:
        low = BUCKET_THRESHOLDS[bucket_idx - 1]
        high = BUCKET_THRESHOLDS[bucket_idx]
        return (low + high) // 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ClankerClassifier.load_pretrained(CHECKPOINT, device=device)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sep_id = tokenizer.vocab_size  # 50257
    pad_id = sep_id + 1            # 50258

    # Engine + zone classifier for comparison
    engine = PendulumV2()
    zc = ZoneClassifier()

    print("=" * 80)
    print("Retrained Model Test Results")
    print("=" * 80)
    print()

    for sentence in TEST_SENTENCES:
        # --- Engine result ---
        vadug_engine, _ = engine.process_text(sentence)
        zone_result = zc.classify(vadug_engine)

        # --- Model result ---
        # Encode like ClankerDataset does: [BOS] text [SEP] pad...
        bos_id = tokenizer.bos_token_id or 50256
        text_ids = tokenizer.encode(sentence, add_special_tokens=False)
        input_ids = [bos_id] + text_ids + [sep_id]
        emotion_pos = len(input_ids) - 1  # position of SEP token

        # Pad to 256
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < 256:
            input_ids.append(pad_id)
            attention_mask.append(0)

        input_ids_t = torch.tensor([input_ids], device=device)
        attention_mask_t = torch.tensor([attention_mask], device=device)
        emotion_pos_t = torch.tensor([emotion_pos], device=device)

        with torch.no_grad():
            logits = model(input_ids_t, attention_mask_t, emotion_pos_t)

        # Get predicted buckets
        dims = ["V", "A", "D", "U", "G"]
        predicted = {}
        for dim in dims:
            bucket_idx = logits[dim].argmax(dim=-1).item()
            predicted[dim] = (BUCKET_NAMES[bucket_idx], bucket_to_value(bucket_idx))

        print(f'  "{sentence}"')
        print(f'    Engine:  V={vadug_engine.v:3d} A={vadug_engine.a:3d} D={vadug_engine.d:3d} U={vadug_engine.u:3d} G={vadug_engine.g:3d}  -> Zone: {zone_result.zone} ({zone_result.confidence:.2f})')
        print(f'    Model:   V={predicted["V"][1]:3d} A={predicted["A"][1]:3d} D={predicted["D"][1]:3d} U={predicted["U"][1]:3d} G={predicted["G"][1]:3d}  ({predicted["V"][0]}, {predicted["A"][0]}, {predicted["D"][0]}, {predicted["U"][0]}, {predicted["G"][0]})')
        print()


if __name__ == "__main__":
    main()
