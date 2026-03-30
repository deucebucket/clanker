#!/usr/bin/env python3
"""Two-character conversation powered by SmolLM2 + Clanker VADUG.

SmolLM2 generates dialogue. Clanker scores emotional state.
Each character's VADUG state conditions the next response.
The model speaks. Clanker feels.

Usage: python3 demo/conversation_loop.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug

# Path to GGUF model
MODEL_PATH = os.path.expanduser("~/Downloads/SmolLM2-1.7B-Instruct.F16.gguf")

# Character personalities
PERSONALITIES = {
    "hothead": {
        "name": "Marcus",
        "desc": "Short temper, speaks before thinking, loud, passionate. Gets angry fast but also cools down fast. Loyal underneath.",
        "baseline": {"v": 120, "a": 160, "d": 150, "u": 20, "g": 135},
    },
    "peacekeeper": {
        "name": "Sam",
        "desc": "Avoids conflict, people-pleaser, apologizes too much. Absorbs other people's emotions. Quietly resentful underneath.",
        "baseline": {"v": 135, "a": 100, "d": 90, "u": 5, "g": 140},
    },
    "ice": {
        "name": "Kai",
        "desc": "Emotionally guarded, speaks in short sentences, uses silence as a weapon. Hard to read. Protective of feelings.",
        "baseline": {"v": 128, "a": 90, "d": 160, "u": 0, "g": 120},
    },
    "empath": {
        "name": "River",
        "desc": "Feels everything deeply, cries easily, takes things personally. Warm and caring but gets overwhelmed fast.",
        "baseline": {"v": 140, "a": 140, "d": 100, "u": 10, "g": 170},
    },
    "joker": {
        "name": "Danny",
        "desc": "Uses humor to deflect, never serious, cracks jokes when uncomfortable. Mask over real feelings. Afraid of vulnerability.",
        "baseline": {"v": 145, "a": 130, "d": 130, "u": 5, "g": 110},
    },
}

TOPICS = [
    "You found out your partner lied about where they were last night.",
    "Your roommate keeps eating your food without asking.",
    "One of you forgot the other's birthday.",
    "You disagree about whether to move to a new city for a job.",
    "One of you said something hurtful during an argument last week and never apologized.",
    "You just found out a mutual friend has been talking behind both your backs.",
    "One of you wants kids and the other doesn't.",
    "Your best friend is dating your ex.",
    "You're stuck in traffic and late for something important together.",
    "One of you got a promotion and the other got passed over at the same company.",
]


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


def dominance_label(d):
    if d > 160: return "commanding"
    if d > 140: return "confident"
    if d > 120: return "steady"
    if d > 100: return "neutral"
    return "yielding"


def build_system_prompt(char_name, vadug, personality_desc="", relationship_desc=""):
    """Build a system prompt that conditions the model on emotional state + personality."""
    v, a, d, u, g = vadug.v, vadug.a, vadug.d, vadug.u, vadug.g
    emo = emotion_label(v)
    dom = dominance_label(d)

    prompt = f"You are {char_name}. "
    if personality_desc:
        prompt += f"Your personality: {personality_desc} "
    prompt += (
        f"Right now you feel {emo} (intensity {abs(v-128)}/128). "
        f"Your stance is {dom}. "
    )

    if v < 100:
        prompt += "You are in real pain. Your responses are short, raw, maybe broken. "
    elif v < 120:
        prompt += "You are hurting. You might deflect, get defensive, or go quiet. "
    elif v < 128:
        prompt += "You are uneasy. Something is off. You are guarded. "
    elif v > 160:
        prompt += "You are genuinely warm and open. "
    elif v > 140:
        prompt += "You are in a good place. Relaxed and present. "

    if d < 90:
        prompt += "You feel powerless in this conversation. You yield easily. "
    elif d > 160:
        prompt += "You are asserting yourself. You push back. "

    if u > 30:
        prompt += "This feels urgent -- you can't just let it go. "

    if g > 160:
        prompt += "This matters deeply to you. "

    if relationship_desc:
        prompt += relationship_desc + " "

    prompt += "Keep responses short (1-2 sentences). Be natural, not formal. No quotation marks."
    return prompt


def score_message(text):
    """Run Clanker engine on text, return VADUG."""
    vadug, trace = compute_vadug(text)
    structures = [s.pattern for s in trace["structures"]]
    return vadug, structures


class Character:
    def __init__(self, name):
        self.name = name
        self.v = 128
        self.a = 128
        self.d = 128
        self.u = 0
        self.g = 128
        self.history = []

    @property
    def vadug(self):
        from engine.shared import VADUG
        return VADUG(v=self.v, a=self.a, d=self.d, u=self.u, g=self.g)

    def receive(self, vadug):
        """Update state based on incoming message's emotional impact."""
        blend = 0.3  # how much the incoming message shifts baseline
        self.v = int(self.v * (1 - blend) + vadug.v * blend)
        self.a = int(self.a * (1 - blend) + vadug.a * blend)
        self.d = int(self.d * (1 - blend) + vadug.d * blend)
        self.u = max(0, int(vadug.u * 0.8))
        self.g = int(self.g * (1 - blend) + vadug.g * blend)

    def status(self):
        return f"V={self.v} ({emotion_label(self.v)}), D={self.d} ({dominance_label(self.d)}), G={self.g}"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Download SmolLM2-1.7B-Instruct.F16.gguf to ~/Downloads/")
        sys.exit(1)

    print("Loading SmolLM2-1.7B...")
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: pip install llama-cpp-python")
        sys.exit(1)

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,  # all layers on GPU
        verbose=False,
    )
    print("Model loaded.\n")

    # Pick personalities
    print("PERSONALITIES:")
    pkeys = list(PERSONALITIES.keys())
    for i, k in enumerate(pkeys):
        p = PERSONALITIES[k]
        print(f"  {i+1}. {p['name']} ({k}) -- {p['desc'][:60]}...")
    print()
    a_pick = int(input("Pick character A (1-5): ").strip() or "1") - 1
    b_pick = int(input("Pick character B (1-5): ").strip() or "2") - 1
    pers_a = PERSONALITIES[pkeys[min(a_pick, len(pkeys)-1)]]
    pers_b = PERSONALITIES[pkeys[min(b_pick, len(pkeys)-1)]]

    char_a = Character(pers_a["name"])
    char_b = Character(pers_b["name"])
    # Apply personality baselines
    for k in ["v", "a", "d", "u", "g"]:
        setattr(char_a, k, pers_a["baseline"][k])
        setattr(char_b, k, pers_b["baseline"][k])

    # Pick topic
    print("\nTOPICS:")
    for i, t in enumerate(TOPICS):
        print(f"  {i+1}. {t[:65]}...")
    print(f"  0. Custom topic")
    t_pick = int(input("\nPick topic (0-10): ").strip() or "1")
    if t_pick == 0:
        topic = input("Your topic: ").strip()
    else:
        topic = TOPICS[min(t_pick-1, len(TOPICS)-1)]

    print(f"\n{'='*60}")
    print(f"  {char_a.name} ({pkeys[a_pick]}) vs {char_b.name} ({pkeys[b_pick]})")
    print(f"  Topic: {topic}")
    print(f"  {char_a.name}: {char_a.status()}")
    print(f"  {char_b.name}: {char_b.status()}")
    print(f"{'='*60}")

    starter = topic

    print(f"\n  {char_a.name}: {starter}")
    vadug_a, structs_a = score_message(starter)
    char_b.receive(vadug_a)
    print(f"  [Clanker: V={vadug_a.v} {emotion_label(vadug_a.v)} | {structs_a}]")
    print(f"  [{char_b.name} state: {char_b.status()}]")

    # Conversation loop
    speakers = [(char_b, char_a), (char_a, char_b)]  # alternating
    last_message = starter

    for turn in range(20):
        speaker, listener = speakers[turn % 2]

        # Build prompt with personality
        spk_pers = pers_a if speaker.name == char_a.name else pers_b
        sys_prompt = build_system_prompt(
            speaker.name, speaker.vadug,
            personality_desc=spk_pers["desc"],
            relationship_desc=f"You are talking to {listener.name} about: {topic}",
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{listener.name} said: \"{last_message}\""},
        ]

        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=60,
            temperature=0.8,
            top_p=0.9,
        )

        reply = response["choices"][0]["message"]["content"].strip()
        # Clean up
        reply = reply.replace(f"{speaker.name}:", "").replace(f'"{reply}"', reply).strip()
        if not reply:
            reply = "..."

        # Score with Clanker
        vadug, structs = score_message(reply)
        listener.receive(vadug)

        # Display
        emo = emotion_label(vadug.v)
        print(f"\n  {speaker.name}: {reply}")
        print(f"  [Clanker: V={vadug.v} {emo} | {structs}]")
        print(f"  [{listener.name} state: {listener.status()}]")

        last_message = reply

        # Check for conversation end signals
        if vadug.v < 60 or vadug.u > 80:
            print(f"\n  --- CRISIS DETECTED: {emo} ---")

        input("  [Enter to continue, 'quit' to stop] ")

    print("\nConversation ended.")


if __name__ == "__main__":
    main()
