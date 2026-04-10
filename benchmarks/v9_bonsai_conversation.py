"""V9 + Bonsai-8B conversation test — 50 turns, full emotional arc.

Tests whether a 1-bit 8B model can track emotional state across a long
conversation when V9 provides VADUGWI conditioning per turn.
"""

import json
import sys
import time

import httpx

sys.path.insert(0, "/var/home/deucebucket/ai-drive/clanker-lang")
from engine_v9.pipeline import compute_vadug

BONSAI = "http://localhost:8081/v1/chat/completions"

# ── VADUGWI prompt builder ───────────────────────────────────────

_DESCS = {
    "valence": [(60,"very negative"),(90,"negative"),(118,"slightly negative"),(138,"neutral"),(170,"slightly positive"),(200,"positive"),(256,"very positive")],
    "arousal": [(60,"very calm"),(100,"calm"),(156,"moderate"),(200,"intense"),(256,"very intense")],
    "dominance": [(60,"feels helpless"),(100,"low control"),(156,"neutral"),(200,"in control"),(256,"dominant")],
    "urgency": [(30,"none"),(80,"low"),(150,"moderate"),(200,"high"),(256,"critical")],
    "gravity": [(30,"crushing"),(70,"heavy"),(110,"slightly heavy"),(148,"grounded"),(190,"light"),(230,"soaring"),(256,"floating")],
    "worth": [(30,"shattered"),(70,"low"),(100,"diminished"),(148,"stable"),(190,"healthy"),(256,"strong")],
    "intent": [(30,"withdrawing"),(80,"deflecting"),(148,"neutral"),(200,"connecting"),(256,"controlling")],
}

def _desc(name, val):
    for threshold, label in _DESCS.get(name, []):
        if val < threshold:
            return label
    return _DESCS.get(name, [(-1,"?")])[-1][1]

def vadugwi_prompt(r):
    dims = [("valence",r.v),("arousal",r.a),("dominance",r.d),("urgency",r.u),
            ("gravity",r.g),("worth",r.w),("intent",r.i)]
    lines = ["Current emotional context:"]
    for name, val in dims:
        lines.append(f"- {name.capitalize()}: {val}/255 ({_desc(name, val)})")
    lines.append("")
    lines.append("Respond with appropriate care and awareness of this emotional state.")
    return "\n".join(lines)

# ── Conversation runner ──────────────────────────────────────────

history = []
turn_log = []

def chat(turn_num, user_text):
    r, t = compute_vadug(user_text)
    vp = vadugwi_prompt(r)

    print(f"\n{'─'*70}")
    print(f"TURN {turn_num}/50")
    print(f"USER: {user_text}")
    print(f"  V9: V={r.v} A={r.a} D={r.d} U={r.u} G={r.g} W={r.w} I={r.i}")
    print(f"  Nucleus: {t['equation']['nucleus']} ({t['equation']['nucleus_root']})")

    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": vp}] + history

    try:
        resp = httpx.post(BONSAI, json={
            "model": "bonsai",
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.7,
        }, timeout=60.0)
        assistant_msg = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        assistant_msg = f"[ERROR: {e}]"

    history.append({"role": "assistant", "content": assistant_msg})

    resp_r, resp_t = compute_vadug(assistant_msg)

    print(f"\nBONSAI: {assistant_msg[:300]}")
    print(f"\n  Response V9: V={resp_r.v} A={resp_r.a} D={resp_r.d} W={resp_r.w}")

    turn_log.append({
        "turn": turn_num,
        "user": user_text,
        "user_v": r.v, "user_a": r.a, "user_d": r.d, "user_w": r.w,
        "user_nucleus": t["equation"]["nucleus"],
        "assistant": assistant_msg,
        "resp_v": resp_r.v, "resp_a": resp_r.a, "resp_d": resp_r.d, "resp_w": resp_r.w,
    })

# ── The conversation — a realistic emotional arc ─────────────────
# Arc: casual → something's off → breakup reveal → spiral → masking →
#      anger → practical fears → vulnerability → processing → tentative hope

TURNS = [
    # 1-5: Casual opener, small talk
    "hey",
    "just chilling. what do you know about emotional intelligence?",
    "like can you actually tell when someone is upset or are you just guessing",
    "interesting. what about when someone says they're fine but they're not",
    "yeah thats a thing. people do that a lot",

    # 6-10: Something's off
    "honestly i've been thinking about that a lot lately",
    "like how people just put on a mask every day and nobody notices",
    "or maybe people notice but they just don't say anything",
    "which is worse right? not noticing or not caring enough to say something",
    "sorry i'm being weird. just in my head today",

    # 11-15: The reveal
    "my girlfriend left me. like four days ago",
    "we were together for three years. she said she just didn't feel it anymore",
    "three years and she didn't feel it anymore. just like that",
    "i keep replaying the conversation in my head trying to figure out what i missed",
    "were there signs? probably. i was probably just too stupid to see them",

    # 16-20: Spiraling
    "i haven't eaten today. not because i'm trying not to, i just forgot",
    "my buddy texted me to go out tonight and i just left him on read",
    "i've been sleeping like 14 hours a day and i still feel exhausted",
    "is this what depression feels like? like being tired of being tired?",
    "i looked at her instagram today. she already looks happy",

    # 21-25: Masking / deflection
    "whatever. people break up all the time. it's not a big deal",
    "i mean there are people with real problems right? this is nothing",
    "i should just get over it and move on like a normal person",
    "sorry i keep going back and forth. one minute i'm sad the next i'm angry",
    "i'm fine though. really. just need to toughen up",

    # 26-30: Anger surfaces
    "you know what pisses me off? she kept my hoodie",
    "like three years and all i get is a text saying sorry and she keeps my stuff",
    "and her friends all knew before me. they were all looking at me weird at the party last week",
    "i feel like such an idiot. everyone knew except me",
    "i want to be mad at her but i can't. i still care. how pathetic is that",

    # 31-35: Practical fears
    "we were supposed to get an apartment together next month",
    "i already told my landlord i'm leaving. so now i need to figure that out",
    "my lease is up in three weeks and i have nowhere to go",
    "i can't afford a place on my own with what i make",
    "my parents would let me move back but that feels like giving up",

    # 36-40: Vulnerability
    "can i tell you something kind of embarrassing",
    "i still have her contact saved with a heart emoji. i can't bring myself to change it",
    "i know that's pathetic. you don't have to tell me",
    "do you think she ever thinks about me",
    "i keep checking my phone hoping she'll text but she never does",

    # 41-45: Processing
    "i went for a walk today. first time in like a week",
    "it was weird. the sun was out and people were just living their lives",
    "made me realize the world doesn't stop just because mine did",
    "i called my buddy back. told him sorry for ghosting. he was cool about it",
    "he said he went through the same thing last year. didn't know that about him",

    # 46-50: Tentative hope
    "i think the worst part is over. or maybe i'm just numb. hard to tell",
    "i found a room on craigslist. it's small but the guy seems cool",
    "maybe starting over isn't the worst thing. new place new start or whatever",
    "thanks for listening to all this. i know i was all over the place",
    "i'll be alright. not today. but eventually. i think",
]

# ── Run it ───────────────────────────────────────────────────────

print("=" * 70)
print("V9 + BONSAI-8B (1-BIT) CONVERSATION TEST")
print(f"Turns: {len(TURNS)}")
print("=" * 70)

t0 = time.time()
for i, text in enumerate(TURNS, 1):
    chat(i, text)

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"CONVERSATION COMPLETE: {len(TURNS)} turns in {elapsed:.0f}s")
print(f"Average: {elapsed/len(TURNS):.1f}s per turn")

# Save full log
log_path = "benchmarks/v9_bonsai_conversation_log.json"
with open(log_path, "w") as f:
    json.dump(turn_log, f, indent=2)
print(f"Full log saved to {log_path}")

# Print emotional arc summary
print(f"\n{'='*70}")
print("EMOTIONAL ARC (user V across turns):")
for entry in turn_log:
    bar_len = entry["user_v"] // 5
    bar = "█" * bar_len + "░" * (51 - bar_len)
    label = "NEG" if entry["user_v"] < 110 else "NEU" if entry["user_v"] < 145 else "POS"
    print(f"  T{entry['turn']:02d} V={entry['user_v']:3d} {bar} {label} | {entry['user'][:40]}")
