#!/usr/bin/env python3
"""Train Clanker-Micro V2: Learn the MATH, not the answers.

Instead of: text → 5 bucket classifiers → VADUG
This does:  text → per-word force predictor → fixed pendulum math → VADUG

The model learns to predict for EACH word:
  1. Role: OPERATOR, PAYLOAD, NEGATOR, IDIOM, NEUTRAL, etc. (classification)
  2. Force tuple: (dV, dA, dD, dU, dG) (regression)
  3. Coefficient modifier: how much this word scales the next payload (regression)

Then the FIXED math layer (identical to pendulum_v2) computes the final VADUG.
The model learns the forces. The physics is constant.

Training signal: engine traces from PendulumV2.process_text()
"""

import json
import os
import sys
import time
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo.pendulum_v2 import PendulumV2

# ── Constants ─────────────────────────────────────────────────────

ROLES = ["NEUTRAL", "OPERATOR", "PAYLOAD", "NEGATOR", "IDIOM", "IDIOM_PART",
         "SCOPE", "BOUNDARY", "FILLER", "FUZZY", "SARCASM", "COMPARATIVE",
         "LITOTES"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
NUM_ROLES = len(ROLES)
CENTER = 128.0

# ── Model ─────────────────────────────────────────────────────────

class ClankerForcePredictor(nn.Module):
    """Predicts per-word emotional forces from text.

    For each word position, outputs:
      - role_logits: (NUM_ROLES,) — what role this word plays
      - force: (5,) — dV, dA, dD, dU, dG deltas
      - coefficient: (1,) — how much this word modifies the next payload

    The FIXED pendulum math then computes final VADUG from these predictions.
    """

    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        hidden = config.n_embd

        # Per-word prediction heads
        self.role_head = nn.Linear(hidden, NUM_ROLES)
        self.force_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 5),  # dV, dA, dD, dU, dG
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),  # coefficient is always positive
        )

    def forward(self, input_ids, attention_mask):
        """Returns per-position predictions.

        Returns:
            role_logits: (batch, seq_len, NUM_ROLES)
            forces: (batch, seq_len, 5)
            coefficients: (batch, seq_len, 1)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        role_logits = self.role_head(hidden)
        forces = self.force_head(hidden) * 50.0  # scale to force range (-128 to +127)
        coefficients = self.coeff_head(hidden)

        return role_logits, forces, coefficients


def pendulum_forward(forces, coefficients, roles, seq_len, momentum=0.82, force_scale=0.5):
    """Fixed pendulum math — identical physics to the engine.

    Takes per-word forces/coefficients/roles and computes final VADUG.
    This layer has NO learnable parameters.
    """
    batch_size = forces.shape[0]
    state = torch.full((batch_size, 5), CENTER, device=forces.device)

    for t in range(seq_len):
        force_t = forces[:, t, :]  # (batch, 5)
        coeff_t = coefficients[:, t, 0]  # (batch,)
        role_t = roles[:, t]  # (batch,) — role index

        # Only PAYLOAD and IDIOM roles apply force
        is_payload = (role_t == ROLE_TO_IDX["PAYLOAD"]) | (role_t == ROLE_TO_IDX["IDIOM"])
        is_payload = is_payload.float().unsqueeze(1)  # (batch, 1)

        # Target: where this word alone would place the pendulum
        target = CENTER + force_t * coeff_t.unsqueeze(1) * force_scale

        # Momentum blend (only for payloads)
        blend = 1.0 - momentum
        new_state = state * momentum + target * blend
        # Apply force only where word is a payload
        state = state * (1 - is_payload) + new_state * is_payload

    return state  # (batch, 5) — final VADUG


# ── Dataset ───────────────────────────────────────────────────────

class TraceDataset(Dataset):
    """Generate training pairs by running engine and capturing traces."""

    def __init__(self, sentences, tokenizer, engine, max_length=64):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for sent in sentences:
            vadug, trace = engine.process_text(sent)
            if not trace:
                continue

            # Extract per-word targets from trace
            word_roles = []
            word_forces = []
            word_coeffs = []

            for entry in trace:
                role_str = entry.get("role", "NEUTRAL")
                if role_str not in ROLE_TO_IDX:
                    role_str = "NEUTRAL"
                role_idx = ROLE_TO_IDX[role_str]

                # Parse force from note if it's a payload
                note = entry.get("note", "")
                coeff = 1.0
                coeff_match = re.search(r'coeff=([0-9.]+)', note)
                if coeff_match:
                    coeff = float(coeff_match.group(1))

                # Parse operator coefficient
                for op_name in ["self_high", "amplifiers", "diminishers", "hedging",
                                "present_tense", "past_tense", "hypothetical",
                                "articles", "demo_far", "demo_near", "possessive_self",
                                "self_medium", "abstract"]:
                    match = re.search(rf'{op_name}=([0-9.]+)', note)
                    if match:
                        coeff = float(match.group(1))
                        break

                # Compute the force delta this word applied
                # For payloads: delta = current_v - previous_v (approximately)
                # For operators/neutral: delta ≈ 0
                v, a, d, u, g = entry["v"], entry["a"], entry["d"], entry["u"], entry["g"]

                word_roles.append(role_idx)
                word_forces.append([0, 0, 0, 0, 0])  # will be computed below
                word_coeffs.append(coeff)

            # Compute actual force deltas between trace steps
            for i in range(len(trace)):
                if i == 0:
                    word_forces[i] = [
                        trace[i]["v"] - CENTER,
                        trace[i]["a"] - CENTER,
                        trace[i]["d"] - CENTER,
                        trace[i]["u"] - 0,
                        trace[i]["g"] - CENTER,
                    ]
                else:
                    word_forces[i] = [
                        trace[i]["v"] - trace[i-1]["v"],
                        trace[i]["a"] - trace[i-1]["a"],
                        trace[i]["d"] - trace[i-1]["d"],
                        trace[i]["u"] - trace[i-1]["u"],
                        trace[i]["g"] - trace[i-1]["g"],
                    ]

            self.examples.append({
                "text": sent,
                "target_vadug": [vadug.v, vadug.a, vadug.d, vadug.u, vadug.g],
                "word_roles": word_roles,
                "word_forces": word_forces,
                "word_coeffs": word_coeffs,
                "num_words": len(trace),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = self.tokenizer(
            ex["text"], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )

        # Pad word-level targets to max_length
        n = ex["num_words"]
        roles = ex["word_roles"][:self.max_length] + [0] * (self.max_length - n)
        forces = ex["word_forces"][:self.max_length] + [[0]*5] * (self.max_length - n)
        coeffs = ex["word_coeffs"][:self.max_length] + [1.0] * (self.max_length - n)
        mask = [1.0] * min(n, self.max_length) + [0.0] * (self.max_length - n)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target_roles": torch.tensor(roles, dtype=torch.long),
            "target_forces": torch.tensor(forces, dtype=torch.float),
            "target_coeffs": torch.tensor(coeffs, dtype=torch.float),
            "word_mask": torch.tensor(mask, dtype=torch.float),
            "target_vadug": torch.tensor(ex["target_vadug"], dtype=torch.float),
        }


# ── Training ──────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load engine for trace generation
    engine = PendulumV2()

    # Collect training sentences
    print("Loading training sentences...")
    sentences = []

    # Phase 1: Rosetta Stone
    p1_path = os.path.join(PROJECT_ROOT, "training", "data", "phase1.jsonl")
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            for line in f:
                d = json.loads(line)
                sentences.append(d["english"])
        print(f"  Phase 1: {len(sentences)}")

    # Phase 1 expanded
    p1e_path = os.path.join(PROJECT_ROOT, "training", "data", "phase1_expanded.jsonl")
    if os.path.exists(p1e_path):
        before = len(sentences)
        with open(p1e_path) as f:
            for line in f:
                d = json.loads(line)
                sentences.append(d["english"])
        print(f"  Phase 1 expanded: {len(sentences) - before}")

    # EmoBank
    emo_path = os.path.join(PROJECT_ROOT, "training", "data", "emobank_significant.jsonl")
    if os.path.exists(emo_path):
        before = len(sentences)
        with open(emo_path) as f:
            for line in f:
                d = json.loads(line)
                sentences.append(d.get("english", d.get("text", "")))
        print(f"  EmoBank: {len(sentences) - before}")

    print(f"  Total sentences: {len(sentences)}")

    # Generate traces
    print("Generating traces (this IS the training data)...")
    start = time.time()
    dataset = TraceDataset(sentences, tokenizer, engine, max_length=64)
    elapsed = time.time() - start
    print(f"  Generated {len(dataset)} trace examples in {elapsed:.1f}s")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Build model
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=128, n_layer=6, n_head=4, n_positions=128,
    )
    model = ClankerForcePredictor(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} parameters ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    epochs = 20
    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 200, total_steps)

    # Loss functions
    role_loss_fn = nn.CrossEntropyLoss(reduction="none")
    force_loss_fn = nn.MSELoss(reduction="none")
    vadug_loss_fn = nn.MSELoss()

    print(f"\nTraining: {epochs} epochs, {total_steps} steps")
    print("=" * 60)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_role_acc = 0
        epoch_vadug_mae = 0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_roles = batch["target_roles"].to(device)
            target_forces = batch["target_forces"].to(device)
            target_coeffs = batch["target_coeffs"].to(device)
            word_mask = batch["word_mask"].to(device)
            target_vadug = batch["target_vadug"].to(device)

            # Forward
            role_logits, pred_forces, pred_coeffs = model(input_ids, attention_mask)

            # Loss 1: Role classification (per word)
            role_loss = role_loss_fn(
                role_logits.view(-1, NUM_ROLES),
                target_roles.view(-1)
            ).view(target_roles.shape) * word_mask
            role_loss = role_loss.sum() / word_mask.sum()

            # Loss 2: Force regression (per word, only for payload/idiom words)
            payload_mask = ((target_roles == ROLE_TO_IDX["PAYLOAD"]) |
                           (target_roles == ROLE_TO_IDX["IDIOM"])).float()
            force_loss = (force_loss_fn(pred_forces, target_forces).mean(-1) * payload_mask)
            force_loss = force_loss.sum() / (payload_mask.sum() + 1e-8)

            # Loss 3: Coefficient regression (per word, for operators)
            operator_mask = (target_roles == ROLE_TO_IDX["OPERATOR"]).float()
            coeff_loss = ((pred_coeffs.squeeze(-1) - target_coeffs) ** 2 * operator_mask)
            coeff_loss = coeff_loss.sum() / (operator_mask.sum() + 1e-8)

            # Loss 4: Final VADUG (run fixed math, compare to engine output)
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

        print(f"  [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | Role acc: {avg_role_acc:.1%} | VADUG MAE: {avg_vadug_mae:.1f}")
        sys.stdout.flush()

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best_v2")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            torch.save(config, os.path.join(save_dir, "config.pt"))

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Saved: training/checkpoints/best_v2/")


if __name__ == "__main__":
    train()
