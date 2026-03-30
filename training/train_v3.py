#!/usr/bin/env python3
"""Train Clanker-Micro V3: Learn structural roles + proximity + patterns.

V3 model learns:
  1. Per-word STRUCTURAL ROLE (18 classes: SELF_REF, EMOTIONAL, NEGATOR, etc.)
  2. Sentence-level STRUCTURAL PATTERNS (FAREWELL, CRISIS, SARCASM, etc.)
  3. Final VADUG from pooled representation

The model learns to READ STRUCTURE, not memorize emotional scores.
Stars have mass. Operators shape the field. Dark matter inherits from proximity.

Training signal: V3 engine traces (word roles + structures + VADUG).
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.word_classifier import ROLES

NUM_ROLES = len(ROLES)
PATTERN_NAMES = [
    "FAREWELL", "METHOD_ACQUISITION", "FINALITY", "BLANKET_APOLOGY",
    "SELF_REMOVAL", "SUSPICIOUS_CALM", "EXHAUSTION", "NO_EXIT",
    "SELF_NULLIFY", "SARCASM_INVERSION", "CHOPPER_SPLIT",
    "PURSUIT_OF_METHOD", "FLEEING", "POWER_OVER_SELF",
    "SELF_SUBMISSION", "D_INVERSION",
    "BETRAYAL", "BRAVADO", "VICTIMIZATION",
    "CALLING_OUT", "DIRECTED_POSITIVE", "MINIMIZER",
]
NUM_PATTERNS = len(PATTERN_NAMES)
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_NAMES)}


class ClankerV3Model(nn.Module):
    """V3 structural model — learns roles, patterns, VADUG.

    Three heads:
      1. Role head: per-word structural role (18 classes)
      2. Pattern head: sentence-level structures (11 binary labels)
      3. VADUG head: final 5D coordinates (regression)
    """

    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        hidden = config.n_embd

        self.role_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, NUM_ROLES),
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, NUM_PATTERNS),
        )
        self.vadug_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 5),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        role_logits = self.role_head(hidden)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        pattern_logits = self.pattern_head(pooled)
        vadug_pred = self.vadug_head(pooled) * 128.0 + 128.0

        return role_logits, pattern_logits, vadug_pred


class V3Dataset(Dataset):
    def __init__(self, trace_path, tokenizer, max_length=64):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        role_to_idx = {r: i for i, r in enumerate(ROLES)}

        with open(trace_path) as f:
            for line in f:
                d = json.loads(line)
                text = d["english"]
                vadug = d["vadug"]
                word_roles = d.get("word_roles", [])
                structures = d.get("structures", [])

                roles = []
                for wr in word_roles:
                    rname = wr.get("role", "NEUTRAL")
                    roles.append(role_to_idx.get(rname, role_to_idx["NEUTRAL"]))

                pattern_vec = [0.0] * NUM_PATTERNS
                for s in structures:
                    pname = s.get("pattern", "")
                    if pname in PATTERN_TO_IDX:
                        pattern_vec[PATTERN_TO_IDX[pname]] = 1.0

                self.examples.append({
                    "text": text, "vadug": vadug, "roles": roles,
                    "patterns": pattern_vec, "num_words": len(word_roles),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = self.tokenizer(
            ex["text"], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        n = min(ex["num_words"], self.max_length)
        roles = ex["roles"][:self.max_length] + [0] * (self.max_length - n)
        word_mask = [1.0] * n + [0.0] * (self.max_length - n)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target_roles": torch.tensor(roles, dtype=torch.long),
            "word_mask": torch.tensor(word_mask, dtype=torch.float),
            "target_patterns": torch.tensor(ex["patterns"], dtype=torch.float),
            "target_vadug": torch.tensor(ex["vadug"], dtype=torch.float),
        }


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    trace_path = os.path.join(PROJECT_ROOT, "training", "data", "v3_traces.jsonl")
    if not os.path.exists(trace_path):
        print(f"ERROR: {trace_path} not found")
        sys.exit(1)

    print("Loading V3 traces...")
    dataset = V3Dataset(trace_path, tokenizer, max_length=64)
    print(f"  {len(dataset)} examples")

    val_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"  Train: {train_size}, Val: {val_size}")

    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=128, n_layer=6, n_head=4, n_positions=128)
    model = ClankerV3Model(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params ({params/1e6:.1f}M)")

    role_loss_fn = nn.CrossEntropyLoss(reduction="none")
    pattern_loss_fn = nn.BCEWithLogitsLoss()
    vadug_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    epochs = 30
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 300, total_steps)

    print(f"\nTraining: {epochs} epochs, {total_steps} steps")
    print("  Learning: word ROLES + sentence PATTERNS + VADUG")
    print("=" * 60)
    sys.stdout.flush()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        t_ra, t_pa, t_vm, t_l, t_n = 0, 0, 0, 0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tr = batch["target_roles"].to(device)
            tw = batch["word_mask"].to(device)
            tp = batch["target_patterns"].to(device)
            tv = batch["target_vadug"].to(device)

            rl, pl, vp = model(ids, mask)

            r_loss = (role_loss_fn(rl.view(-1, NUM_ROLES), tr.view(-1)).view(tr.shape) * tw).sum() / tw.sum()
            p_loss = pattern_loss_fn(pl, tp)
            v_loss = vadug_loss_fn(vp, tv)
            loss = 1.0 * r_loss + 2.0 * p_loss + 1.0 * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ra = ((rl.argmax(-1) == tr).float() * tw).sum() / tw.sum()
            pa = ((pl.sigmoid() > 0.5).float() == tp).float().mean()
            vm = (vp - tv).abs().mean()
            t_ra += ra.item(); t_pa += pa.item(); t_vm += vm.item(); t_l += loss.item(); t_n += 1

        model.eval()
        v_ra, v_pa, v_vm, v_l, v_n = 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tr = batch["target_roles"].to(device)
                tw = batch["word_mask"].to(device)
                tp = batch["target_patterns"].to(device)
                tv = batch["target_vadug"].to(device)

                rl, pl, vp = model(ids, mask)
                r_loss = (role_loss_fn(rl.view(-1, NUM_ROLES), tr.view(-1)).view(tr.shape) * tw).sum() / tw.sum()
                p_loss = pattern_loss_fn(pl, tp)
                v_loss = vadug_loss_fn(vp, tv)
                loss = 1.0 * r_loss + 2.0 * p_loss + 1.0 * v_loss

                ra = ((rl.argmax(-1) == tr).float() * tw).sum() / tw.sum()
                pa = ((pl.sigmoid() > 0.5).float() == tp).float().mean()
                vm = (vp - tv).abs().mean()
                v_ra += ra.item(); v_pa += pa.item(); v_vm += vm.item(); v_l += loss.item(); v_n += 1

        bm = ""
        vl_avg = v_l / v_n
        if vl_avg < best_val_loss:
            best_val_loss = vl_avg
            sd = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best_v3")
            os.makedirs(sd, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(sd, "model.pt"))
            torch.save(config, os.path.join(sd, "config.pt"))
            with open(os.path.join(sd, "metadata.json"), "w") as f:
                json.dump({"epoch": epoch+1, "val_loss": vl_avg,
                           "val_role_acc": v_ra/v_n, "val_pattern_acc": v_pa/v_n,
                           "val_vadug_mae": v_vm/v_n, "train_examples": train_size}, f, indent=2)
            bm = " *BEST*"

        print(f"  [{epoch+1:2d}/{epochs}] "
              f"T: role={t_ra/t_n:.1%} pat={t_pa/t_n:.1%} MAE={t_vm/t_n:.1f} | "
              f"V: role={v_ra/v_n:.1%} pat={v_pa/v_n:.1%} MAE={v_vm/v_n:.1f}{bm}")
        sys.stdout.flush()

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Saved: training/checkpoints/best_v3/")


if __name__ == "__main__":
    train()
