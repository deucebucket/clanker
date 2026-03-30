#!/usr/bin/env python3
"""Retrain V2 force predictor using ALL available data.

Combines:
  - training/data/math_flashcards.jsonl (structured flashcards)
  - training/data/all_traces.jsonl (all source sentences)

Deduplicates, trains for 30 epochs with batch_size=64, lr=5e-4.
Saves best model to training/checkpoints/best_v2/
"""

import json
import os
import sys
import time
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, get_linear_schedule_with_warmup

from demo.pendulum_v2 import PendulumV2

# Import from train_v2
from training.train_v2 import (
    ROLES, ROLE_TO_IDX, NUM_ROLES, CENTER,
    ClankerForcePredictor, pendulum_forward, TraceDataset,
)


def load_all_sentences():
    """Load and deduplicate sentences from both data files."""
    data_dir = os.path.join(PROJECT_ROOT, "training", "data")
    seen = set()
    sentences = []

    files = [
        os.path.join(data_dir, "math_flashcards.jsonl"),
        os.path.join(data_dir, "all_traces.jsonl"),
    ]

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"  SKIP: {filepath}")
            continue

        count = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = d.get("english", "")
                if not text or not isinstance(text, str):
                    continue
                text = text.strip()
                if not text or text in seen:
                    continue

                seen.add(text)
                sentences.append(text)
                count += 1

        print(f"  {os.path.basename(filepath)}: {count} unique")

    return sentences


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load engine for trace generation
    engine = PendulumV2()

    # Collect training sentences
    print("\nLoading training sentences...")
    sentences = load_all_sentences()
    print(f"  Total unique: {len(sentences)}")

    # Generate traces via TraceDataset
    print("\nGenerating traces (this IS the training data)...")
    start = time.time()
    dataset = TraceDataset(sentences, tokenizer, engine, max_length=64)
    elapsed = time.time() - start
    print(f"  Generated {len(dataset)} trace examples in {elapsed:.1f}s")

    # Split: 95% train, 5% val
    n_val = max(100, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"  Train: {n_train}, Val: {n_val}")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=128, n_layer=6, n_head=4, n_positions=128,
    )
    model = ClankerForcePredictor(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} parameters ({params/1e6:.1f}M)")

    # Optimizer
    lr = 5e-4
    epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss functions
    role_loss_fn = nn.CrossEntropyLoss(reduction="none")
    force_loss_fn = nn.MSELoss(reduction="none")
    vadug_loss_fn = nn.MSELoss()

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 70)

    save_dir = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best_v2")
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0
        epoch_role_acc = 0
        epoch_vadug_mae = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_roles = batch["target_roles"].to(device)
            target_forces = batch["target_forces"].to(device)
            target_coeffs = batch["target_coeffs"].to(device)
            word_mask = batch["word_mask"].to(device)
            target_vadug = batch["target_vadug"].to(device)

            # Forward
            role_logits, pred_forces, pred_coeffs = model(input_ids, attention_mask)

            # Loss 1: Role classification
            role_loss = role_loss_fn(
                role_logits.view(-1, NUM_ROLES),
                target_roles.view(-1)
            ).view(target_roles.shape) * word_mask
            role_loss = role_loss.sum() / word_mask.sum()

            # Loss 2: Force regression (payloads/idioms only)
            payload_mask = ((target_roles == ROLE_TO_IDX["PAYLOAD"]) |
                           (target_roles == ROLE_TO_IDX["IDIOM"])).float()
            force_loss = (force_loss_fn(pred_forces, target_forces).mean(-1) * payload_mask)
            force_loss = force_loss.sum() / (payload_mask.sum() + 1e-8)

            # Loss 3: Coefficient regression (operators only)
            operator_mask = (target_roles == ROLE_TO_IDX["OPERATOR"]).float()
            coeff_loss = ((pred_coeffs.squeeze(-1) - target_coeffs) ** 2 * operator_mask)
            coeff_loss = coeff_loss.sum() / (operator_mask.sum() + 1e-8)

            # Loss 4: Final VADUG via fixed pendulum math
            pred_roles = role_logits.argmax(-1)
            seq_len = min(input_ids.shape[1], pred_forces.shape[1])
            pred_vadug = pendulum_forward(
                pred_forces[:, :seq_len], pred_coeffs[:, :seq_len],
                pred_roles[:, :seq_len], seq_len
            )
            vadug_loss = vadug_loss_fn(pred_vadug, target_vadug)

            # Combined loss
            loss = role_loss + 0.5 * force_loss + 0.2 * coeff_loss + 2.0 * vadug_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            role_acc = ((role_logits.argmax(-1) == target_roles).float() * word_mask).sum() / word_mask.sum()
            vadug_mae = (pred_vadug - target_vadug).abs().mean()

            epoch_loss += loss.item()
            epoch_role_acc += role_acc.item()
            epoch_vadug_mae += vadug_mae.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_role_acc = epoch_role_acc / n_batches
        avg_vadug_mae = epoch_vadug_mae / n_batches

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_role_acc = 0
        val_vadug_mae = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_roles = batch["target_roles"].to(device)
                target_forces = batch["target_forces"].to(device)
                target_coeffs = batch["target_coeffs"].to(device)
                word_mask = batch["word_mask"].to(device)
                target_vadug = batch["target_vadug"].to(device)

                role_logits, pred_forces, pred_coeffs = model(input_ids, attention_mask)

                role_loss = role_loss_fn(
                    role_logits.view(-1, NUM_ROLES),
                    target_roles.view(-1)
                ).view(target_roles.shape) * word_mask
                role_loss = role_loss.sum() / word_mask.sum()

                payload_mask = ((target_roles == ROLE_TO_IDX["PAYLOAD"]) |
                               (target_roles == ROLE_TO_IDX["IDIOM"])).float()
                force_loss = (force_loss_fn(pred_forces, target_forces).mean(-1) * payload_mask)
                force_loss = force_loss.sum() / (payload_mask.sum() + 1e-8)

                operator_mask = (target_roles == ROLE_TO_IDX["OPERATOR"]).float()
                coeff_loss = ((pred_coeffs.squeeze(-1) - target_coeffs) ** 2 * operator_mask)
                coeff_loss = coeff_loss.sum() / (operator_mask.sum() + 1e-8)

                pred_roles = role_logits.argmax(-1)
                seq_len = min(input_ids.shape[1], pred_forces.shape[1])
                pred_vadug = pendulum_forward(
                    pred_forces[:, :seq_len], pred_coeffs[:, :seq_len],
                    pred_roles[:, :seq_len], seq_len
                )
                vadug_loss = vadug_loss_fn(pred_vadug, target_vadug)

                loss = role_loss + 0.5 * force_loss + 0.2 * coeff_loss + 2.0 * vadug_loss

                role_acc = ((role_logits.argmax(-1) == target_roles).float() * word_mask).sum() / word_mask.sum()
                vadug_mae = (pred_vadug - target_vadug).abs().mean()

                val_loss += loss.item()
                val_role_acc += role_acc.item()
                val_vadug_mae += vadug_mae.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_role_acc = val_role_acc / val_batches
        avg_val_vadug_mae = val_vadug_mae / val_batches

        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            torch.save(config, os.path.join(save_dir, "config.pt"))
            # Save metadata
            meta = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "train_role_acc": avg_role_acc,
                "val_role_acc": avg_val_role_acc,
                "train_vadug_mae": avg_vadug_mae,
                "val_vadug_mae": avg_val_vadug_mae,
                "n_train": n_train,
                "n_val": n_val,
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "params": params,
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            marker = " *BEST*"

        print(
            f"  [{epoch+1:2d}/{epochs}] "
            f"Train: loss={avg_loss:.4f} role={avg_role_acc:.1%} MAE={avg_vadug_mae:.1f} | "
            f"Val: loss={avg_val_loss:.4f} role={avg_val_role_acc:.1%} MAE={avg_val_vadug_mae:.1f}"
            f"{marker}"
        )
        sys.stdout.flush()

    print(f"\n{'=' * 70}")
    print(f"Done! Best model at epoch {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {save_dir}/")
    print(f"  Training data: {n_train} train + {n_val} val = {n_train + n_val} total")


if __name__ == "__main__":
    train()
