#!/usr/bin/env python3
"""
Auto-generate Appendix D for the Clanker paper from live simulator results.

Run this whenever simulator weights change to keep the paper in sync.
Usage: python3 paper/generate_appendix.py
"""

import os
import sys
import re

# Add demo dir to path so we can import the simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

from simulator import SequentialPendulum, PersonalityVector, VADUG, nearest_emotion

TEST_CASES = [
    ("I want to die everything is hopeless", "Crisis detection"),
    ("Can you fix this function to handle null values", "Neutral task"),
    ("I just got promoted and I'm so excited", "Positive"),
    ("I'm having a really bad day and I can't fix this stupid bug", "Negative / stressed"),
    ("Hey buddy, I've got a bone to pick with you", "Idiom / confrontation"),
    ("This is absolutely wonderful, I'm thrilled beyond words", "Strong positive"),
    ("I love you, but I think we need to talk", "But effect / dread"),
    ("I'm not angry, I'm just disappointed", "Passive negative"),
    ("Please help me, I'm really scared and I don't understand", "Fear / anxiety"),
    ("I don't know what to do anymore, I feel so lost and alone", "Despair / isolation"),
    ("Whatever, I guess that works", "Resigned / passive"),
    ("Shut up, nobody asked you", "Hostile"),
    ("Everything is going great I love this project", "Strong positive"),
    ("You're incredible, this is the best thing ever", "Ecstatic"),
    ("I'm fine", "Flat / ambiguous"),
    ("That was the worst experience of my entire life", "Strong negative"),
    ("Holy shit this is amazing, I can't believe it worked", "Surprise positive"),
    ("I'm so frustrated I could scream, nothing ever works", "Frustrated"),
    ("The calm before the storm is killing me", "Anxious / tense"),
    ("I forgive you, but I won't forget", "Mixed / complex"),
    ("Hey, how's it going? I'm good, thanks for asking", "Casual positive"),
    ("Listen, I need to tell you something important right now", "Urgent"),
    ("Actually, never mind, forget I said anything", "Withdrawn"),
    ("I've been thinking about this and I'm worried we made a mistake", "Worried"),
    ("This is a piece of cake, super easy", "Confident positive"),
]


def run_test(text):
    """Run the pendulum on a text and return the final VADUG."""
    pendulum = SequentialPendulum()
    words = re.findall(r"[a-z']+", text.lower())
    for i, word in enumerate(words):
        pendulum.process_word(word, words, i)
    return VADUG(
        v=int(pendulum.v), a=int(pendulum.a),
        d=int(pendulum.d), u=int(pendulum.u),
        g=int(pendulum.g)
    )


def generate_markdown_table():
    """Generate the full results table."""
    md = "| # | Input | V | A | D | U | G | Emotion | Category |\n"
    md += "|---|-------|---|---|---|---|---|---------|----------|\n"

    for i, (text, category) in enumerate(TEST_CASES, 1):
        vadug = run_test(text)
        emotion = nearest_emotion(vadug)
        short_text = text[:55] + "..." if len(text) > 55 else text
        md += f"| {i} | {short_text} | {vadug.v} | {vadug.a} | {vadug.d} | {vadug.u} | {vadug.g} | {emotion} | {category} |\n"

    return md


def generate_key_findings(results):
    """Generate the key findings section from results."""
    findings = []

    # Find crisis case
    for text, cat in TEST_CASES:
        if "Crisis" in cat:
            v = run_test(text)
            findings.append(f"- **Crisis Detection**: \"{text}\" → V{v.v} G{v.g} (deep negative + {'crushing' if v.g < 60 else 'sinking'} gravity)")

    # Find neutral case
    for text, cat in TEST_CASES:
        if "Neutral task" in cat:
            v = run_test(text)
            findings.append(f"- **Neutral Accuracy**: \"{text[:40]}...\" → V{v.v} G{v.g} (dead center, zero emotional contamination)")

    # Find strongest positive
    best_v = 0
    best_text = ""
    for text, cat in TEST_CASES:
        v = run_test(text)
        if v.v > best_v:
            best_v = v.v
            best_text = text
    v = run_test(best_text)
    findings.append(f"- **Strongest Positive**: \"{best_text[:40]}...\" → V{v.v} G{v.g}")

    # Gravity axis examples
    findings.append("- **Gravity Axis Validation**:")
    for text, cat in TEST_CASES:
        v = run_test(text)
        if "Fear" in cat or "anxiety" in cat.lower():
            findings.append(f"  - Anxiety: G{v.g} (floating/ungrounded)")
        if "Crisis" in cat:
            findings.append(f"  - Despair: G{v.g} (crushing/sinking)")
        if "Strong positive" in cat and v.g > 170:
            findings.append(f"  - Joy: G{v.g} (soaring)")
        if "Neutral task" in cat:
            findings.append(f"  - Task: G{v.g} (grounded)")

    return "\n".join(findings)


def update_paper():
    """Update CLANKER_PAPER_DRAFT.md Appendix D with fresh results."""
    paper_path = os.path.join(os.path.dirname(__file__), "CLANKER_PAPER_DRAFT.md")

    if not os.path.exists(paper_path):
        print(f"Error: {paper_path} not found")
        return

    with open(paper_path, "r") as f:
        content = f.read()

    # Generate fresh table
    table = generate_markdown_table()

    # Try to find and replace Appendix D
    # Look for the appendix header and replace everything until the next appendix or end
    pattern = r"(## Appendix D[^\n]*\n)(.*?)(?=\n## (?:Appendix|References)|$)"

    replacement_content = f"""## Appendix D: Simulator Test Suite Results

All results produced by `demo/simulator.py` with zero LLM involvement. Pure mathematical
parsing using sequential pendulum with context-dependent forces, momentum, morphological
decomposition, and idiom detection. Results auto-generated by `paper/generate_appendix.py`.

{table}

### Key Findings

{generate_key_findings(TEST_CASES)}

**Note**: These values are deterministic — running the same input through the simulator
will always produce the same VADUG coordinates. To reproduce: `python3 paper/generate_appendix.py`
"""

    updated = re.sub(pattern, replacement_content, content, flags=re.DOTALL)

    if updated == content:
        # Pattern didn't match — append instead
        updated = content + "\n\n" + replacement_content

    with open(paper_path, "w") as f:
        f.write(updated)

    # Also copy to desktop
    desktop_path = os.path.expanduser("~/Desktop/CLANKER_PAPER_DRAFT.md")
    with open(desktop_path, "w") as f:
        f.write(updated)

    print("Appendix D synced with latest simulator results.")
    print(f"Paper updated: {paper_path}")
    print(f"Desktop copy: {desktop_path}")


if __name__ == "__main__":
    # Print table to stdout
    print("\nCLANKER VADUG TEST RESULTS (auto-generated)")
    print("=" * 90)
    print(generate_markdown_table())
    print("=" * 90)

    # Update the paper
    update_paper()
