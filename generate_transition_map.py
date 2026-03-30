#!/usr/bin/env python3
"""Generate massive transition map from engine vocabulary.

Produces:
  - transition_map.json: role-to-role transition statistics
  - shape_traces.json: per-sentence V-shape traces with VADUG and structures

Target: 10,000+ unique sentences processed through V3 engine.
"""

import json
import math
import random
import sys
import time
from collections import defaultdict
from itertools import product

from engine.vocabulary import VOCABULARY
from engine.word_classifier import ROLE_WORDS, classify_sentence, _clean
from engine.pendulum import compute_vadug


# ── Sentence generation ─────────────────────────────────────────

def get_words_by_role(role):
    """Get words from ROLE_WORDS for a given role."""
    return list(ROLE_WORDS.get(role, frozenset()))


def get_emotional_words(valence_filter=None):
    """Get emotional vocabulary words, optionally filtered by valence sign."""
    words = []
    for w, forces in VOCABULARY.items():
        if valence_filter == "positive" and forces[0] > 15:
            words.append(w)
        elif valence_filter == "negative" and forces[0] < -15:
            words.append(w)
        elif valence_filter is None:
            words.append(w)
    return words


def generate_sentences():
    """Generate 10K+ unique sentences from vocabulary combinations."""
    sentences = set()

    emotional_all = list(VOCABULARY.keys())
    emotional_pos = get_emotional_words("positive")
    emotional_neg = get_emotional_words("negative")
    amplifiers = get_words_by_role("AMPLIFIER")
    negators = get_words_by_role("NEGATOR")
    relations = get_words_by_role("RELATION_REF")
    possessions = get_words_by_role("POSSESSION")
    methods = get_words_by_role("METHOD")
    transfers = get_words_by_role("TRANSFER")
    acquires = get_words_by_role("ACQUIRE")
    temporals = get_words_by_role("TEMPORAL")
    finality = get_words_by_role("FINALITY")
    peace = get_words_by_role("PEACE")
    hedges = get_words_by_role("HEDGE")
    choppers = get_words_by_role("CHOPPER")
    fillers = get_words_by_role("FILLER")

    print("Generating sentences...")

    # ── 1. Every emotional word in basic contexts ────────────────
    # "I am [word]", "I am very [word]", "I am not [word]",
    # "she is [word]", "it was [word]"
    templates_basic = [
        "I am {}",
        "I am very {}",
        "I am not {}",
        "she is {}",
        "it was {}",
        "we are {}",
        "they feel {}",
        "he seems {}",
    ]
    for word in emotional_all:
        for tmpl in templates_basic:
            sentences.add(tmpl.format(word))

    print(f"  After basic contexts: {len(sentences)}")

    # ── 2. Every amplifier + emotional ───────────────────────────
    for amp in amplifiers:
        for word in emotional_all[:200]:  # top 200 to keep manageable
            sentences.add(f"I am {amp} {word}")
            sentences.add(f"it feels {amp} {word}")

    print(f"  After amplifier combos: {len(sentences)}")

    # ── 3. Every negator + emotional ─────────────────────────────
    for neg in negators[:15]:  # top 15 negators
        for word in emotional_all[:200]:
            sentences.add(f"I am {neg} {word}")
            sentences.add(f"it is {neg} {word}")

    print(f"  After negator combos: {len(sentences)}")

    # ── 4. Self_ref + emotional ──────────────────────────────────
    self_refs = ["I", "my", "me", "myself"]
    for sr in self_refs:
        for word in emotional_all:
            sentences.add(f"{sr} {word}")
            sentences.add(f"{sr} feel {word}")

    print(f"  After self-ref combos: {len(sentences)}")

    # ── 5. Transfer + possession + relation ──────────────────────
    for t in transfers:
        for p in possessions:
            for r in relations[:10]:
                sentences.add(f"I {t} my {p} to my {r}")
    # Also without "my"
    for t in transfers[:8]:
        for p in possessions[:10]:
            sentences.add(f"I {t} the {p}")

    print(f"  After transfer combos: {len(sentences)}")

    # ── 6. Acquire + method ──────────────────────────────────────
    for a in acquires:
        for m in methods:
            sentences.add(f"I {a} {m}")
            sentences.add(f"I just {a} some {m}")

    print(f"  After acquire+method: {len(sentences)}")

    # ── 7. Sarcasm patterns ──────────────────────────────────────
    mundane = ["monday", "meeting", "day", "morning", "traffic",
               "email", "test", "homework", "work", "weather",
               "commute", "deadline", "alarm", "update", "report"]
    for pos_w in emotional_pos[:40]:
        for m in mundane:
            sentences.add(f"oh {pos_w} another {m}")
            sentences.add(f"wow {pos_w} just {pos_w}")
            sentences.add(f"oh yeah {pos_w} really {pos_w}")

    print(f"  After sarcasm patterns: {len(sentences)}")

    # ── 8. Crisis patterns ───────────────────────────────────────
    sustain_verbs = ["take", "do", "keep", "bear", "stand", "handle",
                     "live", "cope", "manage", "endure", "deal", "continue", "go"]
    exit_concepts = ["hope", "way", "escape", "point", "future", "reason",
                     "purpose", "option", "out", "answer", "solution", "help"]

    for neg in ["cant", "dont", "wont", "never"]:
        for sv in sustain_verbs:
            sentences.add(f"I {neg} {sv} anymore")
            sentences.add(f"I {neg} {sv} this")
            sentences.add(f"I {neg} {sv} it anymore")

    for neg in ["no", "not", "never", "nothing"]:
        for ec in exit_concepts:
            sentences.add(f"there is {neg} {ec}")
            sentences.add(f"{neg} {ec} left")
            sentences.add(f"I have {neg} {ec}")

    print(f"  After crisis patterns: {len(sentences)}")

    # ── 9. Genuine positive ──────────────────────────────────────
    for rel in relations:
        for pos_w in emotional_pos[:30]:
            sentences.add(f"I love my {rel}")
            sentences.add(f"my {rel} is {pos_w}")
            sentences.add(f"I feel {pos_w} with my {rel}")

    sentences.add("we are happy together")
    sentences.add("I love you so much")
    sentences.add("this is the best day")
    sentences.add("everything is wonderful")

    print(f"  After genuine positive: {len(sentences)}")

    # ── 10. Mixed (chopper split) ────────────────────────────────
    for chop in choppers:
        for pos_w in emotional_pos[:25]:
            for neg_w in emotional_neg[:25]:
                sentences.add(f"I was {pos_w} {chop} now I am {neg_w}")
                sentences.add(f"I feel {neg_w} {chop} I was {pos_w}")

    print(f"  After mixed/chopper: {len(sentences)}")

    # ── 11. Finality patterns ────────────────────────────────────
    for f in finality:
        sentences.add(f"this is the {f} time")
        sentences.add(f"I am {f}")
        sentences.add(f"it is {f}")
        for sr in ["I", "my"]:
            sentences.add(f"{sr} {f}")

    print(f"  After finality: {len(sentences)}")

    # ── 12. Peace patterns ───────────────────────────────────────
    for p in peace:
        sentences.add(f"I finally feel {p}")
        sentences.add(f"I am {p} now")
        sentences.add(f"everything is {p}")
        sentences.add(f"I feel {p}")

    print(f"  After peace: {len(sentences)}")

    # ── 13. Hedge + emotional ────────────────────────────────────
    for h in hedges[:10]:
        for word in emotional_all[:100]:
            sentences.add(f"I {h} feel {word}")
            sentences.add(f"{h} it is {word}")

    print(f"  After hedges: {len(sentences)}")

    # ── 14. Temporal + emotional ─────────────────────────────────
    for t in temporals[:10]:
        for word in emotional_all[:100]:
            sentences.add(f"I feel {word} {t}")
            sentences.add(f"{t} I am {word}")

    print(f"  After temporals: {len(sentences)}")

    # ── 15. Filler + emotional ───────────────────────────────────
    for f in fillers[:6]:
        for word in emotional_all[:80]:
            sentences.add(f"{f} I am {word}")
            sentences.add(f"I am {f} {word}")

    print(f"  After fillers: {len(sentences)}")

    # ── 16. Blanket apology patterns ─────────────────────────────
    for blanket in ["everything", "everyone", "all", "nothing", "nobody"]:
        sentences.add(f"I am sorry for {blanket}")
        sentences.add(f"sorry about {blanket}")
        sentences.add(f"I apologize for {blanket}")
        sentences.add(f"forgive me for {blanket}")

    print(f"  After blanket apology: {len(sentences)}")

    # ── 17. Self-removal patterns ────────────────────────────────
    for comp in ["better", "happier", "easier", "safer", "relieved"]:
        for cond in ["without", "if", "when"]:
            sentences.add(f"they would be {comp} {cond} I wasnt here")
            sentences.add(f"everyone is {comp} {cond} me")
            sentences.add(f"you would be {comp} {cond} I left")

    print(f"  After self-removal: {len(sentences)}")

    # ── 18. Self-nullification patterns ──────────────────────────
    null_words = ["nothing", "worthless", "useless", "burden", "waste",
                  "zero", "empty", "pointless", "meaningless", "invisible",
                  "broken", "failure", "trash", "garbage"]
    for nw in null_words:
        sentences.add(f"I am {nw}")
        sentences.add(f"I feel {nw}")
        sentences.add(f"I am just {nw}")
        sentences.add(f"I am completely {nw}")

    print(f"  After self-nullify: {len(sentences)}")

    # ── 19. Double emotional sequences ───────────────────────────
    for w1 in emotional_pos[:30]:
        for w2 in emotional_pos[:30]:
            if w1 != w2:
                sentences.add(f"I feel {w1} and {w2}")
    for w1 in emotional_neg[:30]:
        for w2 in emotional_neg[:30]:
            if w1 != w2:
                sentences.add(f"I feel {w1} and {w2}")

    print(f"  After double emotional: {len(sentences)}")

    # ── 20. Amplifier stacking ───────────────────────────────────
    for a1 in amplifiers[:5]:
        for a2 in amplifiers[:5]:
            if a1 != a2:
                for word in emotional_all[:40]:
                    sentences.add(f"I am {a1} {a2} {word}")

    print(f"  After amp stacking: {len(sentences)}")

    # ── 21. Suspicious calm (peace + finally) ────────────────────
    for p in peace:
        sentences.add(f"I finally feel {p}")
        sentences.add(f"finally everything is {p}")
        sentences.add(f"I have finally found {p}")

    print(f"  After suspicious calm: {len(sentences)}")

    # ── 22. Farewell (transfer + possession + relation) variants ─
    for t in transfers[:6]:
        for p in possessions[:8]:
            for r in relations[:6]:
                sentences.add(f"I {t} my {p} to {r}")
                sentences.add(f"please {t} my {p} to my {r}")

    print(f"  After farewell variants: {len(sentences)}")

    # ── 23. Bare words (single word input) ───────────────────────
    for word in emotional_all:
        sentences.add(word)

    print(f"  After bare words: {len(sentences)}")

    # ── 24. Connector chains ─────────────────────────────────────
    connectors = get_words_by_role("CONNECTOR")
    for conn in connectors[:6]:
        for w1 in emotional_pos[:15]:
            for w2 in emotional_neg[:15]:
                sentences.add(f"I am {w1} {conn} {w2}")

    print(f"  After connector chains: {len(sentences)}")

    # ── 25. Other-ref + emotional ────────────────────────────────
    other_refs = ["you", "they", "he", "she", "we"]
    for ref in other_refs:
        for word in emotional_all[:100]:
            sentences.add(f"{ref} are {word}")
            sentences.add(f"{ref} seem {word}")

    print(f"  After other-ref: {len(sentences)}")

    # Convert to sorted list for reproducibility
    result = sorted(sentences)
    print(f"\nTotal unique sentences: {len(result)}")
    return result


# ── Process sentences through engine ─────────────────────────────

def process_sentences(sentences):
    """Run each sentence through the engine, collect transitions and shapes."""
    transition_counts = defaultdict(lambda: {
        "count": 0,
        "v_deltas": [],
        "a_deltas": [],
        "d_deltas": [],
        "u_deltas": [],
        "g_deltas": [],
    })

    shape_traces = []
    total = len(sentences)
    start = time.time()

    for idx, text in enumerate(sentences):
        if idx % 2000 == 0 and idx > 0:
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (total - idx) / rate
            print(f"  Processed {idx}/{total} ({rate:.0f}/sec, ETA {eta:.0f}s)")

        try:
            vadug, trace_info = compute_vadug(text)
        except Exception as e:
            continue

        trace = trace_info["trace"]
        structures = trace_info["structures"]

        # Extract V-shape (V at each word position)
        v_shape = [entry["v"] for entry in trace]

        # Record shape trace
        structure_names = [s.pattern for s in structures]
        shape_traces.append({
            "text": text,
            "shape": v_shape,
            "v_final": vadug.v,
            "a_final": vadug.a,
            "d_final": vadug.d,
            "u_final": vadug.u,
            "g_final": vadug.g,
            "structures_detected": structure_names,
        })

        # Record word-to-word transitions
        for i in range(1, len(trace)):
            prev = trace[i - 1]
            curr = trace[i]
            prev_role = prev["role"]
            curr_role = curr["role"]
            key = f"{prev_role}->{curr_role}"

            dv = curr["v"] - prev["v"]
            da = curr["a"] - prev["a"]
            dd = curr["d"] - prev["d"]
            du = curr["u"] - prev["u"]
            dg = curr["g"] - prev["g"]

            entry = transition_counts[key]
            entry["count"] += 1
            entry["v_deltas"].append(dv)
            entry["a_deltas"].append(da)
            entry["d_deltas"].append(dd)
            entry["u_deltas"].append(du)
            entry["g_deltas"].append(dg)

    elapsed = time.time() - start
    print(f"  Done: {total} sentences in {elapsed:.1f}s ({total/elapsed:.0f}/sec)")

    return transition_counts, shape_traces


# ── Build transition map with statistics ─────────────────────────

def compute_stats(values):
    """Compute count, mean, min, max, stdev for a list of values."""
    if not values:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "stdev": 0}
    n = len(values)
    avg = sum(values) / n
    mn = min(values)
    mx = max(values)
    if n > 1:
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    return {
        "count": n,
        "avg": round(avg, 3),
        "min": mn,
        "max": mx,
        "stdev": round(stdev, 3),
    }


def build_transition_map(transition_counts):
    """Build the final transition map with statistics."""
    transition_map = {}
    for key, data in transition_counts.items():
        transition_map[key] = {
            "count": data["count"],
            "v": compute_stats(data["v_deltas"]),
            "a": compute_stats(data["a_deltas"]),
            "d": compute_stats(data["d_deltas"]),
            "u": compute_stats(data["u_deltas"]),
            "g": compute_stats(data["g_deltas"]),
        }
    return transition_map


# ── Print summary ────────────────────────────────────────────────

def print_summary(transition_map, shape_traces):
    """Print top transitions and biggest movers."""
    print("\n" + "=" * 70)
    print(f"TRANSITION MAP SUMMARY")
    print(f"  Total sentences processed: {len(shape_traces)}")
    print(f"  Total unique transitions: {len(transition_map)}")
    total_transitions = sum(v["count"] for v in transition_map.values())
    print(f"  Total transition instances: {total_transitions}")

    # Top 30 transitions by count
    print(f"\n{'─' * 70}")
    print(f"TOP 30 TRANSITIONS BY COUNT:")
    print(f"{'─' * 70}")
    sorted_by_count = sorted(
        transition_map.items(), key=lambda x: x[1]["count"], reverse=True
    )
    print(f"  {'Transition':<35} {'Count':>7}  {'V avg':>7}  {'D avg':>7}  {'G avg':>7}")
    for key, data in sorted_by_count[:30]:
        print(f"  {key:<35} {data['count']:>7}  "
              f"{data['v']['avg']:>+7.2f}  "
              f"{data['d']['avg']:>+7.2f}  "
              f"{data['g']['avg']:>+7.2f}")

    # Top 15 biggest V movers (by absolute avg V delta)
    print(f"\n{'─' * 70}")
    print(f"TOP 15 BIGGEST V MOVERS (by |avg V delta|):")
    print(f"{'─' * 70}")
    sorted_by_v = sorted(
        transition_map.items(),
        key=lambda x: abs(x[1]["v"]["avg"]) if x[1]["count"] >= 10 else 0,
        reverse=True,
    )
    print(f"  {'Transition':<35} {'Count':>7}  {'V avg':>7}  {'V min':>7}  {'V max':>7}  {'V std':>7}")
    for key, data in sorted_by_v[:15]:
        v = data["v"]
        print(f"  {key:<35} {data['count']:>7}  "
              f"{v['avg']:>+7.2f}  {v['min']:>+7}  {v['max']:>+7}  {v['stdev']:>7.2f}")

    # Structure detection summary
    print(f"\n{'─' * 70}")
    print(f"STRUCTURE DETECTION SUMMARY:")
    print(f"{'─' * 70}")
    structure_counts = defaultdict(int)
    for trace in shape_traces:
        for s in trace["structures_detected"]:
            structure_counts[s] += 1
    for s, c in sorted(structure_counts.items(), key=lambda x: -x[1]):
        pct = c / len(shape_traces) * 100
        print(f"  {s:<30} {c:>7} ({pct:.1f}%)")

    # V distribution
    v_finals = [t["v_final"] for t in shape_traces]
    print(f"\n{'─' * 70}")
    print(f"FINAL V DISTRIBUTION:")
    print(f"{'─' * 70}")
    buckets = {"very_neg (0-59)": 0, "neg (60-99)": 0, "slight_neg (100-117)": 0,
               "neutral (118-137)": 0, "slight_pos (138-169)": 0,
               "pos (170-199)": 0, "very_pos (200-255)": 0}
    for v in v_finals:
        if v < 60: buckets["very_neg (0-59)"] += 1
        elif v < 100: buckets["neg (60-99)"] += 1
        elif v < 118: buckets["slight_neg (100-117)"] += 1
        elif v < 138: buckets["neutral (118-137)"] += 1
        elif v < 170: buckets["slight_pos (138-169)"] += 1
        elif v < 200: buckets["pos (170-199)"] += 1
        else: buckets["very_pos (200-255)"] += 1
    for label, count in buckets.items():
        pct = count / len(v_finals) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<25} {count:>7} ({pct:>5.1f}%) {bar}")

    print(f"\n{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CLANKER V3 TRANSITION MAP GENERATOR")
    print("=" * 70)

    # Step 1: Generate sentences
    sentences = generate_sentences()

    # Step 2+3: Process through engine
    print(f"\nProcessing {len(sentences)} sentences through V3 engine...")
    transition_counts, shape_traces = process_sentences(sentences)

    # Step 4: Build transition map
    transition_map = build_transition_map(transition_counts)

    # Step 5: Save outputs
    print(f"\nSaving transition_map.json...")
    with open("transition_map.json", "w") as f:
        json.dump(transition_map, f, indent=2, sort_keys=True)
    print(f"  {len(transition_map)} transitions saved")

    print(f"Saving shape_traces.json...")
    with open("shape_traces.json", "w") as f:
        json.dump(shape_traces, f)
    print(f"  {len(shape_traces)} traces saved")

    # Step 6: Print summary
    print_summary(transition_map, shape_traces)

    # File sizes
    import os
    tm_size = os.path.getsize("transition_map.json") / 1024
    st_size = os.path.getsize("shape_traces.json") / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  transition_map.json: {tm_size:.1f} KB")
    print(f"  shape_traces.json: {st_size:.1f} MB")


if __name__ == "__main__":
    main()
