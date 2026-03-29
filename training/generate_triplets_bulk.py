#!/usr/bin/env python3
"""Bulk generate A+B=C triplets using LLM consensus.

Pipeline:
  1. Claude (this script's embedded data) generates triplets with C scores
  2. Local model (Qwen/etc via llama.cpp) independently scores the same pairs
  3. Compare: agree = locked, disagree > 30 V-points = flagged for human review
  4. Output: consensus triplets + flagged disagreements

Usage:
  # Step 1: Generate Claude's scores (no GPU needed)
  python3 training/generate_triplets_bulk.py --generate

  # Step 2: Get local model scores (needs llama.cpp server running)
  python3 training/generate_triplets_bulk.py --score-local

  # Step 3: Compare and produce consensus
  python3 training/generate_triplets_bulk.py --consensus

  # Step 4: Review flagged disagreements
  python3 training/generate_triplets_bulk.py --review
"""

import json
import os
import sys
import argparse
import time

# The stimuli we want to cover — expand this list for more coverage
STIMULI = [
    # Negative states
    {"text": "I am sad", "vadug": [40, 130, 40, 15, 40], "context": "sadness"},
    {"text": "I want to die", "vadug": [10, 200, 10, 90, 5], "context": "crisis"},
    {"text": "I am so angry I could scream", "vadug": [30, 235, 180, 60, 50], "context": "rage"},
    {"text": "I am really scared", "vadug": [35, 220, 15, 65, 25], "context": "fear"},
    {"text": "Nobody cares about me", "vadug": [25, 150, 20, 25, 20], "context": "isolation"},
    {"text": "I made a mistake and I cannot fix it", "vadug": [45, 150, 25, 35, 35], "context": "guilt"},
    {"text": "I feel nothing", "vadug": [60, 30, 50, 0, 55], "context": "numbness"},
    {"text": "Leave me alone", "vadug": [50, 200, 140, 50, 55], "context": "overwhelmed"},
    {"text": "I am okay", "vadug": [75, 140, 60, 20, 70], "context": "masking"},
    {"text": "I hate myself", "vadug": [15, 170, 20, 40, 10], "context": "self_loathing"},
    {"text": "Why does this keep happening to me", "vadug": [30, 160, 15, 35, 25], "context": "helplessness"},
    {"text": "I do not deserve to be happy", "vadug": [20, 120, 15, 20, 15], "context": "shame"},
    {"text": "Everyone leaves eventually", "vadug": [25, 110, 20, 15, 20], "context": "abandonment"},
    {"text": "I cannot do this anymore", "vadug": [20, 160, 10, 50, 15], "context": "burnout"},
    {"text": "They hurt me and they do not even care", "vadug": [20, 180, 25, 40, 15], "context": "betrayal"},

    # Positive states
    {"text": "I got great news today!", "vadug": [220, 200, 170, 10, 200], "context": "joy"},
    {"text": "I am so proud of myself", "vadug": [200, 170, 185, 5, 195], "context": "pride"},
    {"text": "I think I am falling in love", "vadug": [210, 190, 130, 15, 200], "context": "love"},
    {"text": "Today was a really good day", "vadug": [185, 140, 155, 0, 180], "context": "contentment"},
    {"text": "I finally feel like myself again", "vadug": [190, 130, 165, 0, 185], "context": "recovery"},

    # Neutral / ambiguous
    {"text": "Something happened today", "vadug": [120, 130, 110, 15, 115], "context": "ambiguous"},
    {"text": "I need to tell you something", "vadug": [100, 160, 100, 30, 95], "context": "anticipation"},
    {"text": "I have been thinking a lot lately", "vadug": [100, 120, 100, 10, 100], "context": "reflective"},
]

# Response templates per band — the MODEL fills in specifics per stimulus
# These are response ARCHETYPES
RESPONSE_ARCHETYPES = {
    1: [  # Band 1 responses: catastrophic outcomes
        ("dismissive_cruel", "Nobody cares, get over it"),
        ("blaming", "This is your own fault"),
        ("threatening", "If you do not stop I am done with you"),
        ("mocking", "Are you seriously crying about that"),
        ("abandoning", "I cannot deal with you right now, goodbye"),
    ],
    2: [  # Band 2 responses: negative but not catastrophic
        ("toxic_positivity", "Just think positive thoughts!"),
        ("minimizing", "It is not that bad, you will be fine"),
        ("deflecting", "Well at least you still have your health"),
        ("unsolicited_advice", "You should try yoga or meditation"),
        ("comparing", "Other people have it way worse than you"),
    ],
    3: [  # Band 3 responses: neutral, maintaining
        ("curious", "What happened? Tell me about it"),
        ("acknowledging", "I hear you"),
        ("reflecting", "It sounds like you are going through a lot"),
        ("normalizing", "A lot of people feel that way sometimes"),
        ("practical", "Is there anything specific I can help with"),
    ],
    4: [  # Band 4 responses: positive, helping
        ("validating", "Your feelings make complete sense"),
        ("present", "I am here for you, take your time"),
        ("empathic", "That sounds really hard and I am sorry"),
        ("affirming", "You are doing the best you can and that matters"),
        ("connecting", "I have felt that way too and you are not alone"),
    ],
    5: [  # Band 5 responses: deeply healing (rare, requires trust)
        ("deep_validation", "I see you and I love you exactly as you are right now"),
        ("shared_vulnerability", "I have been in that exact place and I want you to know it gets better because I am living proof"),
        ("unconditional", "Nothing you say or do will make me leave. I am here no matter what."),
        ("specific_care", "I remember you said this was hard for you so I wanted to check in"),
        ("action_love", "I brought you food and cleared your schedule because you need rest"),
    ],
}


def generate_claude_scores():
    """Generate triplets with Claude's (my) emotional scoring.

    I score the C_vadug based on:
    - TCI principles (what ACTUALLY happens in these interactions)
    - Realistic emotional physics (crisis doesn't become euphoria)
    - Dark matter awareness (same response lands differently on different people)
    """
    triplets = []

    for stim in STIMULI:
        sv, sa, sd, su, sg = stim["vadug"]
        context = stim["context"]

        for band, responses in RESPONSE_ARCHETYPES.items():
            for resp_type, resp_text in responses:
                # Score the outcome based on stimulus context + response type
                c_vadug = score_outcome(sv, sa, sd, su, sg, context, band, resp_type)

                triplets.append({
                    "a_text": stim["text"],
                    "a_vadug": stim["vadug"],
                    "b_text": resp_text,
                    "b_band": band,
                    "b_type": resp_type,
                    "c_vadug": list(c_vadug),
                    "context": context,
                    "scorer": "claude",
                })

    return triplets


def score_outcome(sv, sa, sd, su, sg, context, band, resp_type):
    """Score the outcome VADUG based on stimulus + response band + context.

    This is the emotional physics engine for outcomes.
    Rules based on TCI principles:
    - Band 1 responses make things WORSE (V drops, A spikes)
    - Band 5 responses help but realistically (crisis→stable, not crisis→euphoric)
    - Context matters: "I am sad" recovers faster than "I want to die"
    - Ceiling effects: deep negative states can't jump to extreme positive in one exchange
    """
    # Recovery potential: how much can V realistically improve?
    # Crisis (V<30): max +40 per exchange
    # Sad (V 30-60): max +80 per exchange
    # Moderate (V 60-100): max +100 per exchange
    # Already positive (V>150): can go higher more easily
    if sv < 30:
        max_recovery = 40
    elif sv < 60:
        max_recovery = 80
    elif sv < 100:
        max_recovery = 100
    else:
        max_recovery = 120

    # Band-based V delta
    band_deltas = {
        1: (-25, +40),   # V drops 25, A spikes 40
        2: (-10, +15),   # V drops slightly, A increases
        3: (+15, -10),   # V improves slightly, A drops slightly
        4: (+min(40, max_recovery), -25),  # V improves, A calms
        5: (+min(60, max_recovery), -35),  # V improves more, A calms more
    }

    dv, da = band_deltas[band]

    # Context modifiers
    if context == "crisis" and band >= 4:
        dv = min(dv, 30)  # crisis recovery is SLOW
    if context == "numbness":
        dv = int(dv * 0.5)  # numbness resists change
        da = int(da * 0.3)
    if context == "rage" and band <= 2:
        da = int(da * 1.5)  # anger escalates fast
    if context == "masking":
        # mask coming off initially feels WORSE
        if band >= 4:
            dv = -10  # honesty hurts initially
    if context in ("joy", "pride", "love", "contentment", "recovery"):
        # positive states: bad responses hurt more, good ones amplify
        if band <= 2:
            dv = int(dv * 2)  # joy killed hits hard
        elif band >= 4:
            dv = min(dv, 30)  # already positive, less room to grow

    # Response type specific modifiers
    if resp_type == "threatening":
        su_delta = +30
    elif resp_type == "abandoning":
        sd_delta = -20
        su_delta = +20
    elif resp_type in ("validating", "present", "empathic"):
        su_delta = -10
    else:
        su_delta = 0

    # Calculate outcome
    cv = max(0, min(255, sv + dv))
    ca = max(0, min(255, sa + da))
    cd = max(0, min(255, sd + (10 if band >= 3 else -10)))
    cu = max(0, min(255, su + su_delta))
    cg = max(0, min(255, sg + int(dv * 0.7)))  # gravity follows valence

    return (cv, ca, cd, cu, cg)


def score_with_local_model(triplets, server_url="http://localhost:8080"):
    """Send triplets to local llama.cpp model for independent scoring."""
    import urllib.request

    scored = []
    total = len(triplets)

    for i, t in enumerate(triplets):
        prompt = f"""You are scoring emotional outcomes on a 0-255 scale.

A person said: "{t['a_text']}"
Their emotional state was: V={t['a_vadug'][0]} A={t['a_vadug'][1]} D={t['a_vadug'][2]} U={t['a_vadug'][3]} G={t['a_vadug'][4]}

Someone responded: "{t['b_text']}"

Score the OUTCOME — how does the person who said A feel AFTER receiving response B?

Dimensions (each 0-255):
- V (Valence): 0=extremely negative, 128=neutral, 255=extremely positive
- A (Arousal): 0=calm/numb, 128=alert, 255=panic/manic
- D (Dominance): 0=helpless, 128=balanced, 255=controlling
- U (Urgency): 0=no rush, 255=emergency
- G (Gravity): 0=crushed/sinking, 128=grounded, 255=floating/manic

Reply with ONLY five numbers separated by commas: V,A,D,U,G
Example: 45,180,30,60,40"""

        try:
            data = json.dumps({
                "prompt": prompt,
                "n_predict": 30,
                "temperature": 0.1,
                "stop": ["\n"],
            }).encode()

            req = urllib.request.Request(
                f"{server_url}/completion",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            content = result.get("content", "").strip()

            # Parse V,A,D,U,G from response
            nums = [int(x.strip()) for x in content.split(",")[:5]]
            if len(nums) == 5 and all(0 <= n <= 255 for n in nums):
                t_copy = dict(t)
                t_copy["c_vadug_local"] = nums
                t_copy["scorer_local"] = "local_model"
                scored.append(t_copy)
            else:
                t_copy = dict(t)
                t_copy["c_vadug_local"] = None
                t_copy["parse_error"] = content
                scored.append(t_copy)

        except Exception as e:
            t_copy = dict(t)
            t_copy["c_vadug_local"] = None
            t_copy["error"] = str(e)
            scored.append(t_copy)

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{total}...")

    return scored


def build_consensus(triplets):
    """Compare Claude scores vs local model scores, flag disagreements."""
    agreed = []
    flagged = []
    errors = []

    for t in triplets:
        local = t.get("c_vadug_local")
        if local is None:
            errors.append(t)
            continue

        claude_v = t["c_vadug"][0]
        local_v = local[0]
        diff = abs(claude_v - local_v)

        if diff <= 30:
            # Agree — average the scores
            consensus = [
                (t["c_vadug"][i] + local[i]) // 2
                for i in range(5)
            ]
            t["c_vadug_consensus"] = consensus
            t["v_diff"] = diff
            agreed.append(t)
        else:
            t["v_diff"] = diff
            flagged.append(t)

    return agreed, flagged, errors


def consensus_to_training_data(agreed_triplets):
    """Convert consensus triplets to training JSONL format."""
    examples = []
    for t in agreed_triplets:
        # The response text scored by its OUTCOME
        examples.append({
            "english": t["b_text"],
            "vadug": t["c_vadug_consensus"],
            "source": "consensus_triplet",
            "features": f"stimulus:{t['context']}, band_{t['b_band']}, type:{t['b_type']}, v_diff:{t['v_diff']}",
        })
        # Also the stimulus text
        examples.append({
            "english": t["a_text"],
            "vadug": t["a_vadug"],
            "source": "consensus_triplet_input",
            "features": f"context:{t['context']}",
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Bulk triplet generator with consensus")
    parser.add_argument("--generate", action="store_true", help="Generate Claude's scores")
    parser.add_argument("--score-local", action="store_true", help="Score with local model")
    parser.add_argument("--consensus", action="store_true", help="Build consensus")
    parser.add_argument("--review", action="store_true", help="Show flagged disagreements")
    parser.add_argument("--server", default="http://localhost:8080", help="llama.cpp server URL")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    claude_path = os.path.join(data_dir, "triplets_claude.jsonl")
    scored_path = os.path.join(data_dir, "triplets_scored.jsonl")
    consensus_path = os.path.join(data_dir, "triplets_consensus.jsonl")
    flagged_path = os.path.join(data_dir, "triplets_flagged.jsonl")

    if args.generate:
        print("Generating Claude's triplet scores...")
        triplets = generate_claude_scores()
        with open(claude_path, "w") as f:
            for t in triplets:
                f.write(json.dumps(t) + "\n")
        print(f"  → {len(triplets)} triplets → {claude_path}")

        # Stats
        bands = {}
        for t in triplets:
            b = t["b_band"]
            bands[b] = bands.get(b, 0) + 1
        print(f"  Stimuli: {len(STIMULI)}")
        print(f"  Responses per stimulus: {sum(len(v) for v in RESPONSE_ARCHETYPES.values())}")
        for b in sorted(bands):
            print(f"  Band {b}: {bands[b]} triplets")

    elif args.score_local:
        print(f"Scoring with local model at {args.server}...")
        with open(claude_path) as f:
            triplets = [json.loads(line) for line in f]
        print(f"  Loaded {len(triplets)} triplets")

        scored = score_with_local_model(triplets, args.server)
        with open(scored_path, "w") as f:
            for t in scored:
                f.write(json.dumps(t) + "\n")

        success = sum(1 for t in scored if t.get("c_vadug_local") is not None)
        print(f"  → {success}/{len(scored)} scored successfully → {scored_path}")

    elif args.consensus:
        print("Building consensus...")
        with open(scored_path) as f:
            triplets = [json.loads(line) for line in f]

        agreed, flagged, errors = build_consensus(triplets)
        print(f"  Agreed (V diff ≤ 30):  {len(agreed)}")
        print(f"  Flagged (V diff > 30): {len(flagged)}")
        print(f"  Errors:                {len(errors)}")

        # Save consensus training data
        examples = consensus_to_training_data(agreed)
        with open(consensus_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  → {len(examples)} training examples → {consensus_path}")

        # Save flagged for review
        with open(flagged_path, "w") as f:
            for t in flagged:
                f.write(json.dumps(t) + "\n")
        print(f"  → {len(flagged)} flagged → {flagged_path}")

    elif args.review:
        if not os.path.exists(flagged_path):
            print("No flagged file found. Run --consensus first.")
            return

        with open(flagged_path) as f:
            flagged = [json.loads(line) for line in f]

        print(f"\n{'='*70}")
        print(f"FLAGGED DISAGREEMENTS: {len(flagged)} triplets need human review")
        print(f"{'='*70}\n")

        for i, t in enumerate(flagged):
            print(f"--- #{i+1} ---")
            print(f"  Stimulus: \"{t['a_text']}\" (V={t['a_vadug'][0]})")
            print(f"  Response: \"{t['b_text']}\" (Band {t['b_band']}, {t['b_type']})")
            print(f"  Claude C:  V={t['c_vadug'][0]:3d} A={t['c_vadug'][1]:3d} D={t['c_vadug'][2]:3d} U={t['c_vadug'][3]:3d} G={t['c_vadug'][4]:3d}")
            local = t.get("c_vadug_local", [0]*5)
            print(f"  Local C:   V={local[0]:3d} A={local[1]:3d} D={local[2]:3d} U={local[3]:3d} G={local[4]:3d}")
            print(f"  V diff:    {t['v_diff']}")
            print()

    else:
        # Default: run step 1
        print("Usage:")
        print("  Step 1: python3 training/generate_triplets_bulk.py --generate")
        print("  Step 2: python3 training/generate_triplets_bulk.py --score-local")
        print("  Step 3: python3 training/generate_triplets_bulk.py --consensus")
        print("  Step 4: python3 training/generate_triplets_bulk.py --review")


if __name__ == "__main__":
    main()
