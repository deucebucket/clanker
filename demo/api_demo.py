#!/usr/bin/env python3
"""Clanker-Lang API Demo — state resolution as a service.

Text in → emotional state out. Zone, confidence, depth, alternatives.

Usage:
    python3 -m demo.api_demo "I miss you so much but I am furious you left"
    python3 -m demo.api_demo --interactive
    python3 -m demo.api_demo --conversation
"""

import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.shared import VADUG
from demo.pendulum_v2 import PendulumV2
from demo.zones import ZoneClassifier
from demo.confidence import ConfidenceScorer
from demo.waveform import WaveformAnalyzer
from demo.conversation import ConversationEngine


engine = PendulumV2()
zones = ZoneClassifier()
confidence = ConfidenceScorer()
waveform = WaveformAnalyzer()


def analyze(text):
    """Full analysis: VADUG + zone + confidence + waveform + depth."""
    vadug, trace = engine.process_text(text)
    zone = zones.classify_cascading(vadug)
    conf = confidence.score(vadug, trace)
    wave = waveform.analyze(trace)

    # Emotional depth: how far from neutral center
    depth = {
        "valence": vadug.v,       # 0=very negative, 128=neutral, 255=very positive
        "arousal": vadug.a,       # 0=numb, 128=moderate, 255=on fire
        "dominance": vadug.d,     # 0=helpless, 128=neutral, 255=in control
        "urgency": vadug.u,       # 0=no rush, 255=crisis routing
        "gravity": vadug.g,       # 0=crushing/sinking, 128=neutral, 255=floating/light
    }

    # Mixed state detection: are we between zones?
    mixed = None
    if zone.alternatives and zone.alternatives[0][1] < 0.8:
        alt_name, alt_dist = zone.alternatives[0]
        if alt_dist < zone.distance * 1.5:
            mixed = {
                "primary": zone.zone.split("/")[0],
                "secondary": alt_name,
                "blend": round(1 - (alt_dist / (zone.distance + alt_dist + 0.01)), 2),
            }

    return {
        "text": text,
        "zone": zone.zone,
        "confidence": zone.confidence,
        "depth": depth,
        "description": zones.describe(zone.zone.split("/")[0]),
        "mixed_state": mixed,
        "waveform": {
            "shape": wave.shape,
            "signature": wave.signature,
            "amplitude": round(wave.amplitude),
            "density": round(wave.density, 2),
        },
        "engine_confidence": conf.level,
        "alternatives": [{"zone": n, "distance": d} for n, d in zone.alternatives[:3]],
        "trace_summary": [
            {"word": t["word"], "role": t["role"], "v": round(t["v"])}
            for t in trace if t["role"] not in ("NEUTRAL", "OPERATOR")
        ],
    }


def print_analysis(result):
    """Pretty print an analysis result."""
    r = result
    mixed = r["mixed_state"]
    d = r["depth"]

    print(f'  "{r["text"]}"')
    print()

    if mixed:
        print(f'  Zone: {mixed["primary"]} + {mixed["secondary"]} '
              f'(blend: {mixed["blend"]:.0%} / {1-mixed["blend"]:.0%})')
    else:
        print(f'  Zone: {r["zone"]}')

    print(f'  Confidence: {r["confidence"]}  |  Engine: {r["engine_confidence"]}')
    print(f'  Description: {r["description"]}')
    print()
    print(f'  Emotional Depth:')
    for dim, val in d.items():
        bar = "█" * (val // 13) + "░" * (20 - val // 13)
        label = ""
        if dim == "valence":
            label = "negative" if val < 100 else "positive" if val > 155 else "neutral"
        elif dim == "dominance":
            label = "helpless" if val < 80 else "in control" if val > 170 else "moderate"
        elif dim == "gravity":
            label = "crushing" if val < 80 else "light" if val > 170 else "grounded"
        elif dim == "arousal":
            label = "numb" if val < 80 else "intense" if val > 170 else "moderate"
        elif dim == "urgency":
            label = "critical" if val > 100 else "low" if val < 30 else "moderate"
        print(f'    {dim:10s}: {val:3d} {bar} {label}')

    print()
    print(f'  Waveform: {r["waveform"]["shape"]} ({r["waveform"]["signature"]}) '
          f'amplitude={r["waveform"]["amplitude"]}')

    if r["alternatives"]:
        alts = ", ".join(f'{a["zone"]}({a["distance"]})' for a in r["alternatives"])
        print(f'  Nearby zones: {alts}')

    # Show emotional payloads
    payloads = [t for t in r["trace_summary"] if t["role"] in ("PAYLOAD", "IDIOM", "EVOKER+PAY")]
    if payloads:
        words = " → ".join(f'{t["word"]}(V={t["v"]})' for t in payloads)
        print(f'  Key words: {words}')

    print()


def interactive():
    """Interactive mode — type sentences, get zones."""
    print("=" * 60)
    print("  Clanker-Lang — Conversation State Resolver")
    print("  Type any sentence. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            text = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        result = analyze(text)
        print()
        print_analysis(result)


def conversation_mode():
    """Conversation mode — tracks trajectory across messages."""
    print("=" * 60)
    print("  Conversation Mode — trajectory tracking active")
    print("  Type messages in sequence. Type 'quit' to exit.")
    print("=" * 60)

    conv = ConversationEngine(engine)

    while True:
        try:
            text = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break

        msg = conv.process_message(text)
        result = analyze(text)

        print()
        print_analysis(result)

        if msg.alerts:
            for a in msg.alerts:
                print(f'  *** [{a.level}] {a.signal}: {a.details[:60]}')

        print(f'  Turn {msg.turn_number} | Arc: {msg.session_arc}')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive()
        elif sys.argv[1] == "--conversation":
            conversation_mode()
        else:
            text = " ".join(sys.argv[1:])
            result = analyze(text)
            print()
            print_analysis(result)
    else:
        # Demo
        demos = [
            "I am so happy right now",
            "I hate everything about this",
            "I miss you so much but I am furious you left",
            "Whatever makes you happy",
            "I want to die everything is hopeless",
            "I am fine",
            "I think maybe I might be a little upset",
            "I AM SO DONE WITH THIS",
        ]

        print("=" * 60)
        print("  Clanker-Lang — State Resolution Demo")
        print("=" * 60)

        for text in demos:
            print()
            result = analyze(text)
            print_analysis(result)
