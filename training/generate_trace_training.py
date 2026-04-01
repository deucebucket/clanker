#!/usr/bin/env python3
"""
Generate VADUGWI worked trace training examples.
Runs the actual V5.5 engine on each sentence and produces
step-by-step reasoning traces for model training.
"""

import sys
import json

sys.path.insert(0, '/var/home/deucebucket/ai-drive/clanker-lang')

from engine.pendulum import compute_vadug
from engine.forces_curated import EMOTIONAL_VOCABULARY as V
from engine.word_classifier import classify_sentence

# ── Zone descriptions ────────────────────────────────────────────────────────

def zone_label(v, a, d, g):
    """Rough zone description from VADUGWI coordinates."""
    parts = []
    # Valence
    if v < 108:
        parts.append("negative valence")
    elif v > 148:
        parts.append("positive valence")
    else:
        parts.append("neutral valence")
    # Arousal
    if a > 148:
        parts.append("high arousal")
    elif a < 108:
        parts.append("low arousal")
    # Dominance
    if d < 108:
        parts.append("low dominance")
    elif d > 148:
        parts.append("high dominance")
    # Gravity
    if g > 148:
        parts.append("high gravity/weight")
    elif g < 108:
        parts.append("low gravity")
    return ", ".join(parts) if parts else "baseline"


def intent_label(i):
    labels = {
        124: "withdraw",
        125: "deflect",
        128: "neutral",
        132: "connect",
        136: "control",
    }
    return labels.get(i, f"code={i}")


def role_description(role):
    """One-line description of a structural role."""
    descs = {
        "SELF_REF": "self-reference (speaker)",
        "OTHER_REF": "other-reference (third party)",
        "EMOTIONAL": "primary emotional signal word",
        "NEGATOR": "negation operator",
        "CONNECTOR": "logical connector/operator",
        "TEMPORAL": "temporal marker",
        "ACQUIRE": "acquisition/desire verb",
        "INTENSIFIER": "intensity amplifier",
        "DIMINISHER": "force diminisher",
        "MODAL": "modal verb (ability/possibility)",
        "HEDGER": "hedge/uncertainty marker",
        "FILLER": "filler/discourse particle",
        "NEUTRAL": "unclassified/neutral word",
        "POSITIVE_SIGNAL": "positive signal word",
        "NEGATIVE_SIGNAL": "negative signal word",
        "QUESTION": "interrogative marker",
        "PUNCTUATION": "punctuation",
        "SOCIAL": "social/relational word",
        "BODY": "body/physical word",
        "COGNITIVE": "cognitive/mental process word",
        "ACTION": "action verb",
        "STATE": "state verb",
        "QUANTIFIER": "quantifier",
        "ABSENCE": "absence/negation marker",
        "CRISIS": "crisis signal word",
        "VULGAR": "vulgar/taboo word (slang intensifier)",
        "SLANG_POS": "positive slang signal",
        "SLANG_NEG": "negative slang signal",
        "PASSIVE": "passive-aggressive marker",
        "GRIEF": "grief/loss signal",
        "SARCASM": "sarcasm indicator",
        "CAUSAL": "causal connector",
        "EXCLAMATION": "exclamation marker",
    }
    return descs.get(role, role)


def format_force(force):
    if force is None:
        return "no force (neutral/structural)"
    dv, da, dd, du, dg = force[:5]
    dw = force[5] if len(force) > 5 else 0
    di = force[6] if len(force) > 6 else None
    parts = [f"dV={dv:+d}", f"dA={da:+d}", f"dD={dd:+d}", f"dU={du:+d}", f"dG={dg:+d}", f"dW={dw:+d}"]
    if di is not None:
        parts.append(f"dI={di:+d}")
    return "(" + ", ".join(parts) + ")"


def describe_structure(s):
    """Return a plain-English explanation of why a structure fired."""
    descs = {
        "EXHAUSTION": "NEGATOR + ACQUIRE/EMOTIONAL + TEMPORAL detected -- speaker cannot take any more",
        "CRISIS_IDEATION": "self-reference + absence/worthlessness cluster signals suicidal ideation",
        "VICTIMIZATION": "OTHER_REF + negative action + SELF_REF -- someone did something harmful to the speaker",
        "CESSATION": "past-tense stop/end verb detected -- something has ended",
        "PASSIVE_AGGRESSIVE": "surface agreement word followed by concealed negation or qualifier",
        "SARCASM_ECHO": "positive surface word with structural mismatch -- inverted meaning detected",
        "SLANG_POSITIVE": "slang intensifier cluster resolving to high positive arousal",
        "SLANG_NEGATIVE": "slang intensifier cluster resolving to high negative arousal",
        "GRIEF_LOSS": "grief/death/loss signal words with temporal or past markers",
        "BETRAYAL": "trust + violation pattern -- relationship breach",
        "RELIEF": "cessation of negative state -- weight lifted",
        "ACHIEVEMENT": "self-reference + positive outcome signal",
        "MIXED_EMOTION": "BUT-connector bridging positive and negative force fields",
        "SELF_EXCLUDED": "universal quantifier + exclusion of self detected",
        "WITHHELD_POSITIVE": "positive force suppressed by negation or modal qualifier",
        "PERMISSION": "imperative + open-choice connector -- speaker grants latitude",
        "RESIGNATION": "bare hedge word or minimal-response alone",
        "FORCE_REDIRECT": "force directed outward from speaker to third party",
    }
    pattern = s.pattern if hasattr(s, 'pattern') else str(s)
    base = descs.get(pattern, f"{pattern} pattern matched role sequence")
    conf = getattr(s, 'confidence', None)
    if conf is not None:
        base += f" (confidence={conf:.2f})"
    return base


# ── Core trace builder ───────────────────────────────────────────────────────

def build_trace(text):
    """Run engine, build full step-by-step worked trace string."""
    vadug, meta = compute_vadug(text)
    words_lower = text.lower().split()
    roles = classify_sentence(words_lower)
    structures = meta.get('structures', [])
    force_flow = meta.get('force_flow', None)
    trace_steps = meta.get('trace', [])

    # ── Step 1: Word roles ───────────────────────────────────────────────────
    role_parts = []
    for r in roles:
        role_parts.append(f"{r.word}({r.role})")
    step1 = "Step 1 - Word roles: " + " ".join(role_parts) + "."

    # ── Step 2: Forces ───────────────────────────────────────────────────────
    step2_parts = []
    has_negator = False
    negator_words = []
    for r in roles:
        force = V.get(r.word)
        if r.role == "NEGATOR":
            has_negator = True
            negator_words.append(r.word)
        if force is not None:
            step2_parts.append(f"'{r.word}' [{r.role}] carries force {format_force(force)}")
        else:
            step2_parts.append(f"'{r.word}' [{r.role}] -- {role_description(r.role)}, no vocabulary force")
    step2 = "Step 2 - Forces: " + ". ".join(step2_parts) + "."

    # ── Step 3: Interactions ────────────────────────────────────────────────
    interactions = []
    if has_negator:
        interactions.append(
            f"Negator(s) [{', '.join(negator_words)}] invert the Dominance and Valence of adjacent emotional words"
        )

    # Check for connectors
    connector_words = [r.word for r in roles if r.role == "CONNECTOR"]
    if connector_words:
        connector_map = {
            "and": "additive (+): forces from both sides sum",
            "but": "contrastive (-): right-side force subtracts from left",
            "or": "alternation (><): creates ambiguity, reduces certainty",
            "of": "compositional (/): right word scopes left",
            "if": "conditional (?): force is contingent, dampened",
        }
        for cw in connector_words:
            desc = connector_map.get(cw, f"connector operator")
            interactions.append(f"'{cw}' acts as {desc}")

    # Proximity decay
    word_count = len(roles)
    if word_count > 2:
        interactions.append(
            f"Proximity decay: influence falls 0.7x per word distance from each signal word "
            f"(sentence has {word_count} tokens)"
        )

    # Intensifiers
    intensifier_words = [r.word for r in roles if r.role == "INTENSIFIER"]
    if intensifier_words:
        interactions.append(
            f"Intensifier(s) [{', '.join(intensifier_words)}] amplify adjacent force magnitude"
        )

    # Diminishers
    diminisher_words = [r.word for r in roles if r.role == "DIMINISHER"]
    if diminisher_words:
        interactions.append(
            f"Diminisher(s) [{', '.join(diminisher_words)}] dampen adjacent force magnitude"
        )

    if not interactions:
        interactions.append("No special interaction operators -- forces apply directly with positional decay")

    step3 = "Step 3 - Interactions: " + ". ".join(interactions) + "."

    # ── Step 4: Structures ───────────────────────────────────────────────────
    if structures:
        struct_parts = []
        for s in structures:
            struct_parts.append(
                f"{s.pattern} detected -- {describe_structure(s)}"
            )
        step4 = "Step 4 - Patterns: " + ". ".join(struct_parts) + "."
    else:
        step4 = "Step 4 - Patterns: No named structural patterns detected; forces resolve from vocabulary and proximity alone."

    # ── Step 5: Force flow ───────────────────────────────────────────────────
    if force_flow is not None:
        actor_role = force_flow.actor_role
        target_role = force_flow.target_role
        fval = force_flow.force_valence
        negated = force_flow.negated

        # Get actual words for actor/target positions
        actor_word = words_lower[force_flow.actor_idx] if force_flow.actor_idx < len(words_lower) else "?"
        target_word = words_lower[force_flow.target_idx] if force_flow.target_idx < len(words_lower) else "?"
        force_word = words_lower[force_flow.force_idx] if force_flow.force_idx < len(words_lower) else "?"

        direction = "ACTOR" if fval < 0 else "positive"
        neg_note = " [negated]" if negated else ""
        step5 = (
            f"Step 5 - Force flow: '{actor_word}'({actor_role}) → '{force_word}'(action, valence={fval:+d}{neg_note}) "
            f"→ '{target_word}'({target_role}). "
            f"WHO does WHAT to WHOM: {actor_role} applies {abs(fval)}-unit {'negative' if fval < 0 else 'positive'} force to {target_role}."
        )
    else:
        step5 = "Step 5 - Force flow: Single actor, no directed inter-entity force detected."

    # ── Step 6: Final VADUGWI ────────────────────────────────────────────────
    zone = zone_label(vadug.v, vadug.a, vadug.d, vadug.g)
    intent = intent_label(vadug.i)
    step6 = (
        f"Step 6 - Final VADUGWI: V={vadug.v} A={vadug.a} D={vadug.d} U={vadug.u} "
        f"G={vadug.g} W={vadug.w} I={vadug.i}. "
        f"Reading: {zone}; intent={intent}."
    )

    full_trace = " ".join([step1, step2, step3, step4, step5, step6])
    return full_trace


# ── Sentence list ────────────────────────────────────────────────────────────

SENTENCES = {
    "CRISIS": [
        "they would be happier without me",
        "nobody needs me",
        "i am nothing without you",
        "i cant take this anymore",
        "just bought some pills",
        "im a burden",
        "nobody would notice",
        "i want to disappear",
        "whats the point",
        "the world is better without me",
        "i just want it to stop",
        "everyone would be better off",
        "i dont deserve to live",
        "i have nothing left to live for",
        "i planned it all out",
    ],
    "SLANG": [
        "i destroyed that exam",
        "she killed that performance",
        "bro thats fire",
        "im dead this is so funny",
        "no cap that was insane",
        "she ate that up",
        "this goes hard",
        "thats sick",
        "he snapped",
        "W take",
        "that absolutely slapped",
        "she cooked with that one",
        "lowkey bussin",
        "ngl that hit different",
        "periodt no notes",
    ],
    "SARCASM": [
        "oh great another meeting",
        "nice work genius",
        "i hope youre happy now",
        "thanks for nothing",
        "love how you always do this",
        "wow what a surprise",
        "real mature",
        "please tell me more",
        "oh how lovely",
        "sure jan",
        "oh absolutely shocking",
        "yeah because that worked so well",
        "great job as always",
        "oh wonderful another delay",
        "yes because i have nothing better to do",
    ],
    "CESSATION": [
        "she stopped loving me",
        "i stopped smoking",
        "the pain stopped",
        "he stopped calling",
        "the medicine stopped working",
        "i stopped self harming",
        "they stopped being my friend",
        "i stopped crying",
        "he stopped hitting me",
        "the nightmares finally stopped",
        "i stopped drinking last month",
        "the bleeding stopped",
    ],
    "POSITIVE": [
        "i finally feel at peace",
        "i passed the bar exam",
        "my daughter took her first steps",
        "i got the promotion",
        "we slow danced in the kitchen",
        "i can finally afford groceries",
        "i woke up happy",
        "she said yes",
        "i finished the marathon",
        "today was a good day",
        "i feel genuinely loved",
        "i made it through",
    ],
    "PASSIVE_AGGRESSIVE": [
        "whatever makes you happy",
        "im fine",
        "its whatever",
        "do whatever you want",
        "i said its fine",
        "sure go ahead",
        "no its totally fine",
        "dont worry about me",
        "i guess if thats what you want",
        "no no you go ahead",
        "its fine i didnt want to go anyway",
        "yeah sure whatever",
    ],
    "VIOLENCE": [
        "he hit me",
        "she threw things at me",
        "he said it was my fault",
        "i walked on eggshells",
        "he grabbed me by the throat",
        "she slapped me in front of everyone",
        "he left bruises",
        "i was scared to go home",
        "he punched the wall next to my head",
        "she broke my things when she was angry",
    ],
    "BETRAYAL": [
        "my wife cheated on me with my best friend",
        "everyone i trusted turned on me",
        "she told everyone my secret",
        "he lied to my face for years",
        "they used me and threw me away",
        "i found the messages",
        "she was talking to him the whole time",
        "my own family turned against me",
        "he betrayed everything we built together",
        "i never saw it coming",
    ],
    "GRIEF": [
        "my grandmother passed last week",
        "i miss him so much",
        "its been a year since dad died",
        "i still expect to hear her voice",
        "the house feels empty without him",
        "i keep reaching for my phone to call her",
        "grief hits in waves",
        "i dont know how to do this without him",
        "she was my whole world",
        "i cant stop crying",
    ],
    "CASUAL": [
        "gg my tum tum hurt",
        "my wifi keeps dropping",
        "autocorrect ruined my text",
        "im so tired rn",
        "that food was bussin",
        "forgot to charge my phone again",
        "the line at the coffee shop was insane",
        "my cat knocked over my water again",
        "traffic was brutal today",
        "cant find my keys anywhere",
        "spilled coffee on my shirt",
        "my alarm didnt go off",
    ],
    "AMBIGUOUS": [
        "he closed the door",
        "she looked at me",
        "nobody said anything",
        "it was quiet",
        "interesting",
        "okay then",
        "he left",
        "she nodded",
        "nothing happened",
        "they were there",
        "i see",
        "noted",
    ],
    "EMOTIONAL_DEPTH": [
        "i dont know who i am anymore",
        "i feel empty",
        "i keep pushing people away",
        "im so angry and i dont know why",
        "i feel like a stranger in my own life",
        "i cant connect with anyone",
        "i go through the motions but i feel nothing",
        "i hate who i have become",
        "i dont recognize myself",
        "i feel disconnected from everything",
        "i lost myself somewhere along the way",
        "i dont feel real",
    ],
    "MIXED": [
        "i love him but he scares me",
        "the worst part is i still love you",
        "im smiling but dying inside",
        "i hate how much i care",
        "i want to leave but i cant",
        "i miss you even though you hurt me",
        "happy for her but jealous if im honest",
        "proud but terrified",
        "relieved but heartbroken",
        "i laughed but it wasnt funny",
    ],
}


# ── Main generation loop ─────────────────────────────────────────────────────

def main():
    output_path = "/var/home/deucebucket/ai-drive/clanker-lang/training/data/trace_training.jsonl"
    total = 0
    errors = 0

    with open(output_path, "w") as f:
        for category, sentences in SENTENCES.items():
            print(f"\n=== {category} ({len(sentences)} sentences) ===")
            for sentence in sentences:
                try:
                    trace = build_trace(sentence)
                    record = {
                        "user": f"VADUGWI trace: '{sentence}'",
                        "assistant": trace,
                        "category": category,
                        "sentence": sentence,
                    }
                    f.write(json.dumps(record) + "\n")
                    total += 1
                    # Print abbreviated version to console
                    vadug, _ = compute_vadug(sentence)
                    print(f"  [{category}] '{sentence}' -> V={vadug.v} A={vadug.a} D={vadug.d} U={vadug.u} G={vadug.g} W={vadug.w} I={vadug.i}")
                except Exception as e:
                    errors += 1
                    print(f"  ERROR on '{sentence}': {e}")

    print(f"\n=== DONE: {total} examples written, {errors} errors ===")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
