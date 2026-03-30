#!/usr/bin/env python3
"""Living Conversation -- two characters that just exist.

No turn limit. No forced topic. They talk, fight, make up, drift,
trigger each other, cool down, bring up old shit. The emotional
state carries through everything.

SmolLM2 speaks. Clanker feels. The conversation runs until you stop it.

Usage: python3 demo/living_conversation.py
"""

import os
import sys
import random
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.pendulum import compute_vadug

MODEL_PATH = os.path.expanduser("~/Downloads/SmolLM2-1.7B-Instruct.F16.gguf")

PERSONALITIES = {
    "hothead": {
        "name": "Marcus",
        "desc": "Short temper. Explosive. Swears. Says things he regrets. Loyal underneath but burns hot. Takes everything personally.",
        "baseline": {"v": 118, "a": 160, "d": 155, "u": 15, "g": 140},
    },
    "peacekeeper": {
        "name": "Sam",
        "desc": "Avoids conflict at all costs. Apologizes even when its not their fault. People-pleaser. Quietly resentful. Bottles up.",
        "baseline": {"v": 138, "a": 95, "d": 85, "u": 5, "g": 145},
    },
    "ice": {
        "name": "Kai",
        "desc": "Emotionally guarded. Short sentences. Uses silence and calm as weapons. Hard to read. Wont show vulnerability. Deflects with logic.",
        "baseline": {"v": 128, "a": 85, "d": 170, "u": 0, "g": 115},
    },
    "empath": {
        "name": "River",
        "desc": "Feels everything deeply. Cries easily. Takes things personally. Warm and caring but gets overwhelmed fast. Wears heart on sleeve.",
        "baseline": {"v": 142, "a": 145, "d": 95, "u": 10, "g": 175},
    },
    "joker": {
        "name": "Danny",
        "desc": "Uses humor to deflect everything. Cracks jokes when uncomfortable. Never serious. Mask over real feelings. Terrified of being vulnerable.",
        "baseline": {"v": 148, "a": 135, "d": 130, "u": 5, "g": 105},
    },
    "narcissist": {
        "name": "Victor",
        "desc": "Everything is about them. Gaslights. Flips blame. Charming on surface. Never apologizes genuinely. Makes you doubt yourself.",
        "baseline": {"v": 140, "a": 110, "d": 200, "u": 0, "g": 90},
    },
}

# Life events that can randomly interrupt the conversation
LIFE_EVENTS = [
    "Your phone buzzes with a notification.",
    "Someone walks by and waves.",
    "You both smell food from somewhere nearby.",
    "One of your phones rings.",
    "A song you both know starts playing.",
    "It starts raining.",
    "A mutual friend texts the group chat.",
    "Someone nearby is arguing loudly.",
    "A dog runs up to you both.",
    "You realize youre both hungry.",
    "An awkward silence falls.",
    "One of you checks the time.",
]

STARTERS = [
    "So... how have you been?",
    "We need to talk.",
    "I saw your post last night.",
    "You look different.",
    "Remember when we used to actually hang out?",
    "I heard something about you.",
    "Can I be honest with you for a second?",
    "I keep thinking about what you said.",
    "Why didnt you text me back?",
    "Whats been going on with you lately?",
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
    if d > 160: return "dominant"
    if d > 140: return "assertive"
    if d > 120: return "steady"
    if d > 100: return "neutral"
    if d > 80: return "yielding"
    return "powerless"


class Character:
    def __init__(self, key, config):
        self.key = key
        self.name = config["name"]
        self.desc = config["desc"]
        self.v = config["baseline"]["v"]
        self.a = config["baseline"]["a"]
        self.d = config["baseline"]["d"]
        self.u = config["baseline"]["u"]
        self.g = config["baseline"]["g"]
        self.baseline_v = config["baseline"]["v"]
        self.wound_count = 0
        self.deepest_v = 128
        self.topics_raised = []

    def receive(self, vadug):
        blend = 0.35
        self.v = int(self.v * (1 - blend) + vadug.v * blend)
        self.a = int(self.a * (1 - blend) + vadug.a * blend)
        self.d = int(self.d * (1 - blend) + vadug.d * blend)
        self.u = max(0, int(vadug.u * 0.7))
        self.g = int(self.g * (1 - blend) + vadug.g * blend)
        if vadug.v < self.deepest_v:
            self.deepest_v = vadug.v
        if vadug.v < 100:
            self.wound_count += 1

    def natural_recovery(self):
        """Slow drift back toward baseline between turns."""
        self.v = int(self.v * 0.95 + self.baseline_v * 0.05)
        self.a = max(80, self.a - 2)
        self.u = max(0, self.u - 3)

    def status_line(self):
        return f"V={self.v}({emotion_label(self.v)}) A={self.a} D={self.d} wounds={self.wound_count}"

    def build_prompt(self, other_name, last_msg, turn_num, life_event=None):
        emo = emotion_label(self.v)
        dom = dominance_label(self.d)

        prompt = f"You are {self.name}. {self.desc}\n"
        prompt += f"You feel {emo} right now. Your stance is {dom}.\n"

        if self.v < 80:
            prompt += "You are in real pain. You might break down, shut down, or lash out.\n"
        elif self.v < 105:
            prompt += "You are hurting. You might get defensive, deflect, or go quiet.\n"
        elif self.v < 120:
            prompt += "Something is bothering you. You are on edge.\n"
        elif self.v > 155:
            prompt += "You are genuinely warm and open right now.\n"

        if self.wound_count > 3:
            prompt += f"You have been hurt {self.wound_count} times in this conversation. You are wearing thin.\n"

        if self.d < 85:
            prompt += "You feel powerless. You might give in or shut down.\n"
        elif self.d > 165:
            prompt += "You are asserting dominance. You push back hard.\n"

        if life_event:
            prompt += f"Something just happened: {life_event}\n"

        prompt += f"You are talking to {other_name}.\n"
        prompt += "Respond in 1-2 short sentences. Be natural. No quotes. No narration. No actions in asterisks. Just what you would actually say out loud."

        return prompt


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    from llama_cpp import Llama

    print("Loading SmolLM2-1.7B...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_gpu_layers=-1, verbose=False)
    print("Ready.\n")

    # Pick characters
    print("CHARACTERS:")
    keys = list(PERSONALITIES.keys())
    for i, k in enumerate(keys):
        p = PERSONALITIES[k]
        print(f"  {i+1}. {p['name']} ({k}) -- {p['desc'][:55]}...")

    a_idx = int(input("\nCharacter A (1-6): ").strip() or "1") - 1
    b_idx = int(input("Character B (1-6): ").strip() or "3") - 1

    char_a = Character(keys[a_idx], PERSONALITIES[keys[a_idx]])
    char_b = Character(keys[b_idx], PERSONALITIES[keys[b_idx]])

    # Pick starter or random
    print(f"\nSTARTERS:")
    for i, s in enumerate(STARTERS):
        print(f"  {i+1}. {s}")
    print(f"  0. Random")
    s_idx = int(input("Pick (0-10): ").strip() or "0")
    starter = STARTERS[s_idx - 1] if s_idx > 0 else random.choice(STARTERS)

    print(f"\n{'='*60}")
    print(f"  {char_a.name} ({char_a.key}) vs {char_b.name} ({char_b.key})")
    print(f"  {char_a.name}: {char_a.status_line()}")
    print(f"  {char_b.name}: {char_b.status_line()}")
    print(f"{'='*60}")

    # First message from A
    print(f"\n  {char_a.name}: {starter}")
    vadug, trace = compute_vadug(starter)
    structs = [s.pattern for s in trace["structures"]]
    char_b.receive(vadug)
    print(f"  [V={vadug.v} {emotion_label(vadug.v)} {structs}]")

    last_msg = starter
    speakers = [(char_b, char_a), (char_a, char_b)]
    turn = 0

    while True:
        speaker, listener = speakers[turn % 2]

        # Random life event (10% chance after turn 5)
        life_event = None
        if turn > 5 and random.random() < 0.10:
            life_event = random.choice(LIFE_EVENTS)
            print(f"\n  --- {life_event} ---")

        # Natural emotional recovery between turns
        speaker.natural_recovery()

        # Generate response
        sys_prompt = speaker.build_prompt(listener.name, last_msg, turn, life_event)

        context = last_msg
        if life_event:
            context = f"{listener.name} said: \"{last_msg}\" (Also: {life_event})"

        r = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": context},
            ],
            max_tokens=60,
            temperature=0.85,
            top_p=0.88,
            repeat_penalty=1.2,
        )

        reply = r["choices"][0]["message"]["content"].strip()
        # Clean up model artifacts
        for prefix in [f"{speaker.name}:", f"{listener.name}:", '"', "*", "(", ">"]:
            reply = reply.lstrip(prefix).strip()
        reply = reply.rstrip('"').split("\n")[0].strip()
        if not reply:
            reply = "..."

        # Clanker scores
        vadug, trace = compute_vadug(reply)
        structs = [s.pattern for s in trace["structures"]]
        listener.receive(vadug)

        # Display
        emo = emotion_label(vadug.v)
        struct_str = f" [{', '.join(structs)}]" if structs else ""
        print(f"\n  {speaker.name}: {reply}")
        print(f"  [V={vadug.v} {emo}{struct_str}]")
        print(f"  {listener.name}: {listener.status_line()}")

        # Breaking point detection
        if listener.v < 50:
            print(f"\n  {'!'*40}")
            print(f"  {listener.name} HAS REACHED BREAKING POINT")
            print(f"  V={listener.v} -- {emotion_label(listener.v)}")
            print(f"  Wounds: {listener.wound_count}")
            print(f"  Deepest: {listener.deepest_v}")
            print(f"  {'!'*40}")

        if speaker.v < 50:
            print(f"\n  {'!'*40}")
            print(f"  {speaker.name} IS BREAKING")
            print(f"  {'!'*40}")

        last_msg = reply
        turn += 1

        # User can intervene, inject, or just watch
        try:
            user_input = input("  [Enter=continue | 'q'=quit | type=inject message] ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if user_input:
                # User injects a message as the next speaker
                last_msg = user_input
                vadug, trace = compute_vadug(user_input)
                other = speakers[(turn) % 2][1]
                other.receive(vadug)
                print(f"  [INJECTED: V={vadug.v} {emotion_label(vadug.v)}]")
                turn += 1
        except (EOFError, KeyboardInterrupt):
            break

    print(f"\n{'='*60}")
    print(f"  CONVERSATION ENDED ({turn} turns)")
    print(f"  {char_a.name}: V={char_a.v} ({emotion_label(char_a.v)}) wounds={char_a.wound_count} deepest={char_a.deepest_v}")
    print(f"  {char_b.name}: V={char_b.v} ({emotion_label(char_b.v)}) wounds={char_b.wound_count} deepest={char_b.deepest_v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
