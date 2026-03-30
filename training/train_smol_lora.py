#!/usr/bin/env python3
"""LoRA fine-tune SmolLM2-1.7B-Instruct with VADUG emotional conditioning.

Teaches SmolLM2 to generate emotionally coherent responses by conditioning
on Clanker's VADUG state. The base model stays frozen -- only the LoRA
adapter weights (~15-30MB) are trained.

Input: training/data/smol_lora_train.jsonl (47K conversation pairs)
Output: training/checkpoints/smol_lora/ (adapter weights)

Requirements: pip install peft trl
"""

import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data(path):
    """Load JSONL chat data."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_chat(example, tokenizer):
    """Format a single example as chat template text."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def main():
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    data_path = os.path.join(PROJECT_ROOT, "training", "data", "smol_lora_train.jsonl")
    output_dir = os.path.join(PROJECT_ROOT, "training", "checkpoints", "smol_lora")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run format_smol_lora.py first.")
        sys.exit(1)

    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit for memory efficiency
    print("Loading model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    params = sum(p.numel() for p in model.parameters())
    print(f"  Base model: {params / 1e6:.0f}M params")

    # LoRA config
    lora_config = LoraConfig(
        r=16,                    # rank -- 16 is a good balance
        lora_alpha=32,           # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable: {trainable / 1e6:.1f}M / {total / 1e6:.0f}M ({trainable / total * 100:.1f}%)")

    # Load data
    print("\nLoading training data...")
    examples = load_data(data_path)
    print(f"  {len(examples)} examples")

    # Format as text
    texts = []
    for ex in examples:
        try:
            text = format_chat(ex, tokenizer)
            texts.append({"text": text})
        except Exception:
            continue
    print(f"  {len(texts)} formatted")

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        max_seq_length=256,
        dataset_text_field="text",
        report_to="none",
    )

    from datasets import Dataset
    train_dataset = Dataset.from_list(texts)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print(f"\nTraining: {training_args.num_train_epochs} epochs")
    print(f"  Batch: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  LR: {training_args.learning_rate}")
    print("=" * 50)
    sys.stdout.flush()

    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    adapter_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors") or f.endswith(".bin")
    )
    print(f"\nDone! Adapter saved: {output_dir}")
    print(f"  Adapter size: {adapter_size / 1e6:.1f} MB")
    print(f"  Base model: ~3.5 GB")
    print(f"  Total stack: ~{3500 + adapter_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
