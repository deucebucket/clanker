#!/usr/bin/env python3
"""Format EmpatheticDialogues into LoRA training data for SmolLM2.

Takes speaker->listener pairs, adds VADUG state conditioning,
and outputs in chat format for LoRA fine-tuning.

The model learns: given emotional state + what someone said,
generate an emotionally coherent response.

Output: training/data/smol_lora_train.jsonl
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug


def emotion_label(v):
    if v > 170: return "joyful"
    if v > 150: return "happy"
    if v > 140: return "content"
    if v > 132: return "okay"
    if v > 124: return "neutral"
    if v > 115: return "uneasy"
    if v > 105: return "hurt"
    if v > 90: return "distressed"
    if v > 70: return "anguished"
    if v > 50: return "devastated"
    return "shattered"


def arousal_label(a):
    if a > 180: return "agitated"
    if a > 150: return "alert"
    if a > 120: return "engaged"
    return "calm"


def dominance_label(d):
    if d > 170: return "commanding"
    if d > 140: return "confident"
    if d > 120: return "steady"
    if d > 100: return "neutral"
    return "yielding"


def format_state(vadug):
    """Format VADUG array into natural language state description."""
    v, a, d, u, g = vadug
    return (
        f"V={v} ({emotion_label(v)}), "
        f"A={a} ({arousal_label(a)}), "
        f"D={d} ({dominance_label(d)}), "
        f"G={g}"
    )


def main():
    data_dir = os.path.join(PROJECT_ROOT, "training", "data")
    ed_path = os.path.join(data_dir, "empathetic_dialogues.jsonl")

    if not os.path.exists(ed_path):
        print(f"ERROR: {ed_path} not found")
        sys.exit(1)

    # Load conversations
    print("Loading EmpatheticDialogues...")
    convos = {}
    with open(ed_path) as f:
        for line in f:
            d = json.loads(line)
            cid = d.get("conv_id", "")
            if cid not in convos:
                convos[cid] = []
            convos[cid].append(d)

    print(f"  {len(convos)} conversations")

    # Build training pairs
    # Format: system prompt with emotional state + user message + assistant response
    examples = []
    for cid, msgs in convos.items():
        for i in range(len(msgs) - 1):
            speaker = msgs[i]
            listener = msgs[i + 1]

            # Speaker says something, listener responds
            if "speaker" not in speaker.get("features", ""):
                continue
            if "listener" not in listener.get("features", ""):
                continue

            speaker_text = speaker["english"]
            listener_text = listener["english"]
            emotion = speaker.get("features", "").split("emotion:")[-1].split(",")[0]

            # Score the speaker's message to get the emotional impact on the listener
            try:
                vadug, _ = compute_vadug(speaker_text)
                state = format_state([vadug.v, vadug.a, vadug.d, vadug.u, vadug.g])
            except Exception:
                state = "V=128 (neutral), A=128 (calm), D=128 (neutral), G=128"

            # Chat format for LoRA training
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"You are in a conversation. "
                            f"The emotional state of this moment: {state}. "
                            f"The tone is {emotion}. "
                            f"Respond naturally and briefly, matching the emotional energy."
                        ),
                    },
                    {"role": "user", "content": speaker_text},
                    {"role": "assistant", "content": listener_text},
                ],
            }
            examples.append(example)

    print(f"  {len(examples)} training pairs")

    # Save
    out_path = os.path.join(data_dir, "smol_lora_train.jsonl")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  Saved: {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Show samples
    print("\nSamples:")
    for ex in examples[:3]:
        sys = ex["messages"][0]["content"]
        usr = ex["messages"][1]["content"]
        ast = ex["messages"][2]["content"]
        print(f"  SYS: {sys[:80]}...")
        print(f"  USR: {usr[:60]}")
        print(f"  AST: {ast[:60]}")
        print()


if __name__ == "__main__":
    main()
