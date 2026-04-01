#!/usr/bin/env python3
"""
Fine-tune Phi-4-mini-instruct with Clanker VADUGWI emotional intelligence.

The engine's knowledge gets baked into the model weights via LoRA.
Engine handles the structured 75%. Model handles conversation + the hard 25%.

Architecture:
  Phi-4-mini (3.8B, MIT) + LoRA adapter trained on 200K VADUGWI traces
  = emotionally intelligent small LLM in a single distributable file

Usage:
  python3 training/train_phi4_lora.py
  python3 training/train_phi4_lora.py --quick   # small subset for testing
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODEL_PATH = os.path.expanduser("~/ai-drive/models/phi-4-mini-instruct")
OUTPUT_DIR = str(PROJECT_ROOT / "training" / "checkpoints" / "phi4-clanker-lora")
TRACES_PATH = str(PROJECT_ROOT / "training" / "data" / "v3_traces.jsonl")
GAP_PATH = str(PROJECT_ROOT / "training" / "data" / "gap_filling_dataset.jsonl")


def load_training_data(max_samples=None, include_gaps=True):
    """Load VADUGWI traces and format for instruction fine-tuning.

    Each trace becomes a conversation:
      User: Score this text emotionally: "i hate myself"
      Assistant: V=85 A=143 D=92 U=6 G=123 W=92 I=19
                 Patterns: SELF_NULLIFY
                 Reading: deeply negative, low self-worth, withdrawing
    """
    samples = []

    # Load main traces
    with open(TRACES_PATH) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                t = json.loads(line)
                samples.append(t)
            except json.JSONDecodeError:
                continue

    # Load gap-filling data (these OVERRIDE engine -- critical for slang/sarcasm)
    if include_gaps and os.path.exists(GAP_PATH):
        gap_samples = []
        with open(GAP_PATH) as f:
            for line in f:
                try:
                    gap_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Oversample gaps 50x relative to regular data (they're the most important)
        for _ in range(50):
            samples.extend(gap_samples)
        print(f"  Gap-filling samples: {len(gap_samples)} x50 = {len(gap_samples)*50}")

    print(f"  Total training samples: {len(samples)}")
    return samples


def format_as_chat(trace):
    """Convert a VADUGWI trace to chat format for instruction tuning."""
    text = trace["english"]
    vadug = trace["vadug"]

    # Dimension names
    dims = ["V", "A", "D", "U", "G", "W", "I"]
    scores = " ".join(f"{d}={v}" for d, v in zip(dims, vadug))

    # Structures
    structs = trace.get("structures", [])
    struct_names = [s["pattern"] if isinstance(s, dict) else s for s in structs]
    struct_line = f"Patterns: {', '.join(struct_names)}" if struct_names else "No structural patterns detected."

    # Reading
    v = vadug[0]
    if v < 80:
        tone = "deeply negative"
    elif v < 110:
        tone = "negative"
    elif v < 120:
        tone = "slightly negative"
    elif v < 136:
        tone = "neutral"
    elif v < 160:
        tone = "positive"
    else:
        tone = "strongly positive"

    w = vadug[5] if len(vadug) > 5 else 128
    if w < 80:
        worth = ", shattered self-worth"
    elif w < 110:
        worth = ", low self-worth"
    elif w < 140:
        worth = ""
    else:
        worth = ", strong self-worth"

    i_val = vadug[6] if len(vadug) > 6 else 128
    if i_val < 30:
        intent = ", withdrawing"
    elif i_val < 80:
        intent = ", deflecting"
    elif i_val < 148:
        intent = ""
    elif i_val < 200:
        intent = ", connecting"
    else:
        intent = ", controlling"

    reading = f"{tone}{worth}{intent}"

    user_msg = f'Score this text emotionally: "{text}"'
    assistant_msg = f"{scores}\n{struct_line}\nReading: {reading}"

    return {"user": user_msg, "assistant": assistant_msg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test with small subset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Download with: huggingface-cli download microsoft/Phi-4-mini-instruct")
        sys.exit(1)

    print("=" * 60)
    print("Phi-4-mini + Clanker VADUGWI LoRA Fine-Tune")
    print(f"Model: {MODEL_PATH}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    max_samples = 5000 if args.quick else None
    traces = load_training_data(max_samples=max_samples)

    # Format as chat
    print("Formatting as instruction pairs...")
    chat_data = [format_as_chat(t) for t in traces]

    # Load model + tokenizer
    print("\nLoading Phi-4-mini...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try native first (transformers 5.4+ may support Phi-4 natively)
    # Fall back to trust_remote_code if needed
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        print("Loaded with native transformers support")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loaded with custom remote code")

    # LoRA config
    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    print("Tokenizing...")

    def tokenize_chat(example):
        prompt = f"<|user|>\n{example['user']}<|end|>\n<|assistant|>\n{example['assistant']}<|end|>"
        enc = tokenizer(prompt, truncation=True, max_length=256, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    from datasets import Dataset
    dataset = Dataset.from_list(chat_data)
    tokenized = dataset.map(tokenize_chat, remove_columns=dataset.column_names)

    # Split
    split = tokenized.train_test_split(test_size=0.05, seed=42)

    # Training
    print(f"\nTraining on {len(split['train'])} samples...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    print(f"\nSaving adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Merge and save full model for GGUF export
    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged_dir = OUTPUT_DIR + "-merged"
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
    print("Convert to GGUF with: python3 llama.cpp/convert_hf_to_gguf.py " + merged_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
