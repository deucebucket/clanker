#!/usr/bin/env python3
"""Train Clanker-Micro: ~4.8M param model that thinks in emotional coordinates.

Architecture: GPT-2 backbone (6 layers, 128 dim, 4 heads) with 958-token
Clanker vocabulary. 5 independent classification heads predict VADUG bucket
indices from English text — no autoregressive generation, no degenerate collapse.

Training format:
    [BOS] english text [SEP] [EMOTION] → 5 heads each classify into 7 buckets

Data:
    Phase 1 (Rosetta Stone): 336 hand-crafted calibration examples (10x weight)
    Phase 2 (engine-labeled): 71K pendulum-engine labeled examples (1x weight)

Loss: Focal loss (gamma=2.0) per head, averaged across heads.
Sampling: Class-balanced WeightedRandomSampler on V-axis buckets.
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import GPT2Config, GPT2Model, get_linear_schedule_with_warmup

# Add project root so we can import the tokenizer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tokenizer"))

# VADUG dimension names and bucket config
VADUG_DIMS = ["V", "A", "D", "U", "G"]
NUM_BUCKETS = 7
BUCKET_THRESHOLDS = [37, 74, 110, 147, 183, 220]
BUCKET_NAMES = ["neg_high", "neg_med", "neg_low", "neutral", "pos_low", "pos_med", "pos_high"]


def value_to_bucket(val):
    """Map a VADUG value (0-255) to a bucket index (0-6)."""
    for i, thresh in enumerate(BUCKET_THRESHOLDS):
        if val < thresh:
            return i
    return 6


class ClankerClassifier(nn.Module):
    """GPT-2 encoder with 5 independent classification heads for VADUG prediction.

    Instead of autoregressive token generation (which causes degenerate collapse),
    this model uses the hidden state at the [EMOTION] token position and passes it
    through 5 separate Linear(hidden_dim, 7) heads. Each head independently
    classifies into 7 emotion buckets.

    This eliminates the sequential dependency between VADUG dimensions that caused
    the model to repeat the same token 5 times.
    """

    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        hidden_dim = config.n_embd

        # 5 independent classification heads — one per VADUG dimension
        self.heads = nn.ModuleDict({
            dim: nn.Linear(hidden_dim, NUM_BUCKETS)
            for dim in VADUG_DIMS
        })

    def forward(self, input_ids, attention_mask, emotion_positions):
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            emotion_positions: (batch,) index of [EMOTION] token in each sequence

        Returns:
            dict mapping dimension name -> (batch, NUM_BUCKETS) logits
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Extract hidden state at [EMOTION] token position for each example
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        emotion_hidden = hidden_states[batch_indices, emotion_positions]  # (batch, hidden_dim)

        # Pass through each classification head independently
        logits = {
            dim: head(emotion_hidden)  # (batch, NUM_BUCKETS)
            for dim, head in self.heads.items()
        }
        return logits

    def save_pretrained(self, save_dir):
        """Save model weights and config."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        # Save the transformer config for reconstruction
        self.transformer.config.save_pretrained(save_dir)

    @classmethod
    def load_pretrained(cls, load_dir, device="cpu"):
        """Load model from checkpoint directory."""
        config = GPT2Config.from_pretrained(load_dir)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(load_dir, "model.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model


class ClankerDataset(Dataset):
    """Training dataset: English text -> 5 VADUG bucket indices.

    Each example becomes:
        input_ids:  [BOS] char-encoded-text [SEP] [EMOTION] [PAD...]
        labels:     5 integers (0-6), one bucket index per VADUG dimension

    The model reads the text and predicts all 5 VADUG dimensions simultaneously
    via independent classification heads at the [EMOTION] token position.
    """

    def __init__(self, phase1_path, phase2_path, tokenizer, max_length=256,
                 phase1_expanded_path=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.v_bucket_indices = []  # V bucket index per example (for sampler)

        # Load Phase 1 (Rosetta Stone — perfect calibration, 10x weight)
        if os.path.exists(phase1_path):
            with open(phase1_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.examples.append({
                        "text": data["english"],
                        "vadug": data["vadug"],  # [V, A, D, U, G] each 0-255
                        "weight": 10.0,
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
            print(f"  Phase 1 (Rosetta Stone): {sum(1 for e in self.examples if e['weight'] == 10.0)} examples")

        # Load Phase 1 Expanded (hand-scored + engine-scored expansion, 8x weight)
        # Slightly lower than original Rosetta Stone but much higher than Phase 2
        if phase1_expanded_path and os.path.exists(phase1_expanded_path):
            expanded_count = 0
            with open(phase1_expanded_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    # Hand-scored sources get higher weight
                    source = data.get("source", "")
                    if source in ("short_calibration", "positive_expansion",
                                  "tci_trajectory", "response_outcome"):
                        weight = 10.0  # hand-scored = same as Rosetta Stone
                    else:
                        weight = 5.0   # engine-scored expansion
                    self.examples.append({
                        "text": data["english"],
                        "vadug": data["vadug"],
                        "weight": weight,
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                    expanded_count += 1
            print(f"  Phase 1 Expanded: {expanded_count} examples (hand-scored + expansion)")

        # Load pangram sentences (context-diverse, hand-scored)
        pangram_path = os.path.join(os.path.dirname(phase1_path), "pangram_sentences.jsonl")
        if os.path.exists(pangram_path):
            pangram_count = 0
            with open(pangram_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.examples.append({
                        "text": data["english"],
                        "vadug": data["vadug"],
                        "weight": 10.0,  # hand-scored = highest weight
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                    pangram_count += 1
            print(f"  Pangram (context-diverse): {pangram_count} examples")

        # Load balance data (fill underrepresented bands)
        balance_files = [
            "balance_neutral.jsonl",
            "balance_positive.jsonl",
            "balance_pos_high.jsonl",
            "balance_pos_mild.jsonl",
            "emobank_vadug.jsonl",  # all 10K — human-rated VAD, engine U/G
        ]
        balance_total = 0
        for bf in balance_files:
            bp = os.path.join(os.path.dirname(phase1_path), bf)
            if os.path.exists(bp):
                with open(bp) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        source = data.get("source", "")
                        # EmoBank: human V/A/D but engine U/G — slightly lower weight
                        if "emobank" in source:
                            weight = 6.0
                        else:
                            weight = 10.0  # hand-scored balance data
                        self.examples.append({
                            "text": data["english"],
                            "vadug": data["vadug"],
                            "weight": weight,
                        })
                        self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                        balance_total += 1
        if balance_total:
            print(f"  Balance + EmoBank: {balance_total} examples")

        # Load triplet training data (consensus + Claude-scored outcomes)
        triplets_path = os.path.join(os.path.dirname(phase1_path), "triplets_training.jsonl")
        if os.path.exists(triplets_path):
            triplet_count = 0
            with open(triplets_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    source = data.get("source", "")
                    if source == "consensus_triplet":
                        weight = 8.0  # two models agreed
                    elif source == "claude_triplet_unverified":
                        weight = 4.0  # only Claude scored, lower confidence
                    else:
                        weight = 6.0  # triplet inputs
                    self.examples.append({
                        "text": data["english"],
                        "vadug": data["vadug"],
                        "weight": weight,
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                    triplet_count += 1
            print(f"  Triplets (A+B=C outcomes): {triplet_count} examples")

        # Load engine error corrections (pangram gaps — context the engine misses)
        # These are the MOST VALUABLE training examples: they teach the model
        # exactly where the engine fails (sarcasm, negation, hedging, clinical).
        error_path = os.path.join(os.path.dirname(phase1_path), "error_corrections.jsonl")
        if os.path.exists(error_path):
            error_count = 0
            with open(error_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.examples.append({
                        "text": data["text"],
                        "vadug": [data["v"], data["a"], data["d"], data["u"], data["g"]],
                        "weight": 15.0,  # HIGHEST weight — these are the gaps
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["v"]))
                    error_count += 1
            print(f"  Error corrections (engine gaps): {error_count} examples (weight=15)")

        # Load calibration data (sarcasm, negation — hand-scored truth, highest weight)
        calibration_dir = os.path.dirname(phase1_path)
        for cal_file in ["sarcasm_calibration.jsonl", "negation_calibration.jsonl"]:
            cal_path = os.path.join(calibration_dir, cal_file)
            if os.path.exists(cal_path):
                cal_count = 0
                with open(cal_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        self.examples.append({
                            "text": data["english"],
                            "vadug": data["vadug"],
                            "weight": 20.0,  # HIGHEST weight — hand-scored truth
                        })
                        self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                        cal_count += 1
                print(f"  Calibration ({cal_file}): {cal_count} examples (weight=20)")

        # Phase 2 DISABLED — engine-labeled book sentences are 43% neutral noise
        # and teach the model to replicate engine mistakes at 71K scale.
        # The model learns better from 3K quality examples than 74K noisy ones.
        # The word force dictionary stays as the ENGINE's runtime lookup,
        # not as training data for the model.
        #
        # To re-enable: uncomment and set skip_phase2=False
        skip_phase2 = True
        phase2_count = 0
        if not skip_phase2 and os.path.exists(phase2_path):
            with open(phase2_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.examples.append({
                        "text": data["english"],
                        "vadug": data["vadug"],
                        "weight": 1.0,
                    })
                    self.v_bucket_indices.append(value_to_bucket(data["vadug"][0]))
                    phase2_count += 1
            print(f"  Phase 2 (engine-labeled): {phase2_count} examples")
        else:
            print(f"  Phase 2: SKIPPED (engine-labeled noise, {phase2_path})")

        print(f"  Total: {len(self.examples)} training examples")
        self._print_v_distribution()

    def _print_v_distribution(self):
        """Print V-axis class distribution."""
        from collections import Counter
        counts = Counter(self.v_bucket_indices)
        total = len(self.v_bucket_indices)
        print(f"  V-axis distribution:")
        for i, name in enumerate(BUCKET_NAMES):
            ct = counts.get(i, 0)
            print(f"    {name:10s}: {ct:6d} ({100*ct/total:.1f}%)")

    def get_balanced_sampler(self):
        """Create a WeightedRandomSampler that equalizes V-axis bucket frequency.

        Inverse-frequency weighting: rarer buckets get sampled more often.
        Rosetta Stone examples get an additional 10x boost.
        """
        from collections import Counter
        counts = Counter(self.v_bucket_indices)
        total = len(self.v_bucket_indices)

        # Inverse frequency weight per bucket
        bucket_weights = {}
        for bucket_idx, count in counts.items():
            bucket_weights[bucket_idx] = total / (len(counts) * count)

        # Per-example sample weight = bucket_weight * data_weight
        sample_weights = []
        for i, ex in enumerate(self.examples):
            bw = bucket_weights[self.v_bucket_indices[i]]
            dw = ex["weight"]  # 10.0 for Rosetta, 1.0 for engine
            sample_weights.append(bw * dw)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _encode_english(self, text):
        """Encode English text using GPT-2 tokenizer (subword BPE)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Special token IDs
        bos_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        emotion_id = self.tokenizer.convert_tokens_to_ids("<emotion>")
        pad_id = self.tokenizer.pad_token_id

        # Encode English text as characters
        text_ids = self._encode_english(ex["text"])

        # Build sequence: [BOS] text [SEP] [EMOTION] [PAD...]
        # Reserve space for: BOS(1) + SEP(1) + EMOTION(1) = 3 tokens
        max_text_len = self.max_length - 3
        text_ids = text_ids[:max_text_len]

        input_ids = [bos_id] + text_ids + [sep_id, emotion_id]
        seq_len = len(input_ids)

        # Position of [EMOTION] token — this is where we extract features
        emotion_pos = seq_len - 1

        # Pad to max_length
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length
            # Ensure emotion_pos is within bounds
            emotion_pos = min(emotion_pos, self.max_length - 1)

        # Labels: 5 bucket indices (0-6), one per VADUG dimension
        v, a, d, u, g = ex["vadug"]
        labels = [
            value_to_bucket(v),
            value_to_bucket(a),
            value_to_bucket(d),
            value_to_bucket(u),
            value_to_bucket(g),
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "emotion_pos": torch.tensor(emotion_pos, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),  # (5,) bucket indices
            "weight": torch.tensor(ex["weight"], dtype=torch.float32),
        }


def focal_loss(logits, labels, gamma=2.0):
    """Focal loss for 7-class classification (one VADUG head).

    Standard CE: -log(p)
    Focal:       -(1-p)^gamma * log(p)

    When the model confidently predicts neutral (p=0.9), the loss multiplier
    is (1-0.9)^2 = 0.01 — nearly zero. Forces the model to focus on examples
    it's getting WRONG (positive sentiment, extreme emotions).

    gamma=0 is standard CE. gamma=2 is the sweet spot for imbalanced data.

    Args:
        logits: (batch, NUM_BUCKETS) raw logits from one classification head
        labels: (batch,) integer bucket indices (0-6)
        gamma: focal exponent

    Returns:
        Scalar loss for this head.
    """
    # Standard CE per example (no reduction)
    ce_loss = F.cross_entropy(logits, labels, reduction="none")

    # Get predicted probability for the correct class
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    p_correct = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    # Focal weight: (1 - p_correct)^gamma
    focal_weight = (1.0 - p_correct) ** gamma

    # Weighted loss
    loss = (focal_weight * ce_loss).mean()
    return loss


def predict_vadug(model, tokenizer, text, device="cpu"):
    """Predict VADUG bucket indices for a given text.

    Args:
        model: ClankerClassifier instance
        tokenizer: ClankerTokenizer instance
        text: English text string
        device: torch device

    Returns:
        dict with keys 'buckets' (5 bucket indices), 'bucket_names' (5 names),
        'raw_logits' (dict of dim -> logits array)
    """
    model.eval()

    # Build input sequence: [BOS] text [SEP] [EMOTION]
    bos_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    emotion_id = tokenizer.convert_tokens_to_ids("<emotion>")
    pad_id = tokenizer.pad_token_id

    text_ids = tokenizer.encode(text, add_special_tokens=False)

    max_length = 256
    max_text_len = max_length - 3
    text_ids = text_ids[:max_text_len]

    input_ids = [bos_id] + text_ids + [sep_id, emotion_id]
    seq_len = len(input_ids)
    emotion_pos = seq_len - 1

    pad_len = max_length - seq_len
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
    else:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
        emotion_pos = min(emotion_pos, max_length - 1)

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=device)
    emotion_pos_t = torch.tensor([emotion_pos], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids_t, attention_mask_t, emotion_pos_t)

    buckets = []
    bucket_names = []
    raw_logits = {}
    for dim in VADUG_DIMS:
        dim_logits = logits[dim][0]  # (NUM_BUCKETS,)
        bucket_idx = dim_logits.argmax().item()
        buckets.append(bucket_idx)
        bucket_names.append(BUCKET_NAMES[bucket_idx])
        raw_logits[dim] = dim_logits.cpu().numpy()

    return {
        "buckets": buckets,
        "bucket_names": bucket_names,
        "raw_logits": raw_logits,
    }


def train():
    """Main training loop for Clanker-Micro."""

    print("=" * 60)
    print("Clanker-Micro Training (5-Head Classifier)")
    print("=" * 60)

    # -----------------------------------------------------------
    # Device setup
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")
    else:
        print("WARNING: No CUDA device found. Training on CPU will be very slow.")

    # -----------------------------------------------------------
    # Load tokenizer — GPT-2 tokenizer for ENGLISH text
    # The Clanker 958-token tokenizer is for bytecode IR, not English.
    # "I love you" and "I hate you" mapped to identical tokens with it.
    # GPT-2 tokenizer (50,257 tokens) can actually READ English.
    # -----------------------------------------------------------
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Add special tokens the model needs
    special_tokens = {"sep_token": "<sep>", "pad_token": "<pad>",
                      "additional_special_tokens": ["<emotion>"]}
    tokenizer.add_special_tokens(special_tokens)
    print(f"\nTokenizer: GPT-2 ({tokenizer.vocab_size} tokens)")
    print(f"Special tokens: sep={tokenizer.sep_token_id}, pad={tokenizer.pad_token_id}")

    # -----------------------------------------------------------
    # Model configuration — Clanker-Nano with GPT-2 tokenizer
    # Larger vocab = larger embedding table, but same tiny transformer.
    # 50,260 vocab * 128 dim = ~6.4M embedding params + 1.3M transformer
    # = ~7.7M total. Still tiny. RTX 3090 eats this for breakfast.
    # -----------------------------------------------------------
    config = GPT2Config(
        vocab_size=len(tokenizer),  # 50,260 with special tokens
        n_embd=128,
        n_layer=6,
        n_head=4,
        n_positions=256,
        n_inner=512,
        activation_function="gelu_new",
        resid_pdrop=0.15,
        embd_pdrop=0.15,
        attn_pdrop=0.15,
        layer_norm_epsilon=1e-5,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Store tokenizer type in config for inference
    config.tokenizer_type = "gpt2"

    model = ClankerClassifier(config)
    num_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for head in model.heads.values() for p in head.parameters())
    print(f"\nModel: Clanker-Nano (5-Head Classifier)")
    print(f"  Parameters: {num_params / 1e6:.1f}M total ({head_params} in heads)")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embed dim: {config.n_embd}")
    print(f"  Heads: {config.n_head}")
    print(f"  Context: {config.n_positions}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Classification heads: 5 x Linear({config.n_embd}, {NUM_BUCKETS})")
    model = model.to(device)

    # -----------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    phase1_path = os.path.join(data_dir, "phase1.jsonl")
    phase1_expanded_path = os.path.join(data_dir, "phase1_expanded.jsonl")
    phase2_path = os.path.join(data_dir, "phase2.jsonl")

    print(f"\nLoading data...")
    dataset = ClankerDataset(
        phase1_path, phase2_path, tokenizer, max_length=256,
        phase1_expanded_path=phase1_expanded_path,
    )

    # Batch size tuned for RTX 3090 24GB
    batch_size = 64

    # Class-balanced sampler — equalizes V-axis bucket frequency
    sampler = dataset.get_balanced_sampler()
    print(f"\n  Using class-balanced sampler (inverse V-frequency weighting)")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # replaces shuffle=True
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    # -----------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------
    focal_gamma = 0.5  # 2.0 was too aggressive — loss hit 0 by epoch 2
    print(f"  Focal loss gamma: {focal_gamma}")
    num_epochs = 30  # more epochs for smaller quality dataset
    total_steps = len(dataloader) * num_epochs
    warmup_steps = min(200, total_steps // 10)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,  # lower LR for small dataset — 1e-3 caused degenerate shortcuts
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision for RTX 3090
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: 3e-4")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Architecture: 5 independent classification heads (no autoregressive)")
    print(f"{'=' * 60}\n")

    model.train()
    global_step = 0
    best_loss = float("inf")
    log_interval = 100
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        num_batches = 0

        # Track per-head accuracy within epoch
        head_correct = {dim: 0 for dim in VADUG_DIMS}
        head_total = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_pos = batch["emotion_pos"].to(device)
            labels = batch["labels"].to(device)  # (batch, 5)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask, emotion_pos)
                    # Compute focal loss for each head, average across heads
                    loss = torch.tensor(0.0, device=device)
                    for i, dim in enumerate(VADUG_DIMS):
                        loss = loss + focal_loss(logits[dim], labels[:, i], gamma=focal_gamma)
                    loss = loss / len(VADUG_DIMS)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, emotion_pos)
                loss = torch.tensor(0.0, device=device)
                for i, dim in enumerate(VADUG_DIMS):
                    loss = loss + focal_loss(logits[dim], labels[:, i], gamma=focal_gamma)
                loss = loss / len(VADUG_DIMS)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Track accuracy
            with torch.no_grad():
                for i, dim in enumerate(VADUG_DIMS):
                    preds = logits[dim].argmax(dim=-1)
                    head_correct[dim] += (preds == labels[:, i]).sum().item()
                head_total += labels.size(0)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                steps_per_sec = num_batches / elapsed
                # Per-head accuracy string
                acc_str = " ".join(
                    f"{dim}:{100*head_correct[dim]/head_total:.0f}%"
                    for dim in VADUG_DIMS
                )
                print(
                    f"  [{epoch+1}/{num_epochs}] "
                    f"Step {global_step:>6d}/{total_steps} | "
                    f"Loss: {avg:.4f} | "
                    f"Acc: {acc_str} | "
                    f"LR: {lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start
        acc_str = " ".join(
            f"{dim}:{100*head_correct[dim]/max(head_total,1):.1f}%"
            for dim in VADUG_DIMS
        )
        print(
            f"\nEpoch {epoch+1}/{num_epochs} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Acc: {acc_str} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save epoch checkpoint
        ckpt_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_dir)
        print(f"  Saved checkpoint: {ckpt_dir}")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(checkpoint_dir, "best")
            model.save_pretrained(best_dir)
            # Save tokenizer alongside model
            tokenizer.save_pretrained(best_dir)
            print(f"  New best model! Loss: {best_loss:.4f} -> {best_dir}")

    # -----------------------------------------------------------
    # Final test: load best model and predict
    # -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Best model: {os.path.join(checkpoint_dir, 'best')}")
    print(f"  Total steps: {global_step}")

    # Quick sanity check: load and predict
    best_dir = os.path.join(checkpoint_dir, "best")
    if os.path.exists(os.path.join(best_dir, "model.pt")):
        print(f"\nSanity check -- loading best model and predicting...")
        test_model = ClankerClassifier.load_pretrained(best_dir, device=str(device))
        test_model = test_model.to(device)

        test_sentences = [
            "I love you so much",
            "I want to kill myself",
            "The weather is okay today",
            "You make me so angry I could scream",
        ]
        for sent in test_sentences:
            result = predict_vadug(test_model, tokenizer, sent, device=device)
            dims_str = " ".join(
                f"{dim}={name}"
                for dim, name in zip(VADUG_DIMS, result["bucket_names"])
            )
            print(f"  \"{sent}\" -> {dims_str}")

    print(f"\nNext steps:")
    print(f"  1. Check per-head accuracy -- all 5 should learn independently")
    print(f"  2. No more degenerate collapse -- each head picks its own bucket")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
