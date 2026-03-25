#!/usr/bin/env python3
"""
Clanker Pipeline Simulator — Interactive Demo (v0.9.1: Emotional Density)

This file is a backward-compatible shim.  All implementation now lives in
the submodules of the ``demo`` package (shared, forces, pendulum, personality,
response, chunker, grader, sarcasm, arc).  Every symbol that was previously
importable from ``simulator`` is re-exported here so that existing code
(``from simulator import SequentialPendulum, WORD_FORCES`` etc.) still works.
"""

import sys
import os

# ---- Ensure the demo package is importable when running this file directly ----
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
# Also keep _here on path for decoder_templates / morphemes compatibility
if _here not in sys.path:
    sys.path.insert(0, _here)

# ---- Re-export everything from the package modules ----------------------------
from demo.shared import VADUG, VADU, MetadataHeader, PersonalityVector          # noqa: F401
from demo.forces import WORD_FORCES                                              # noqa: F401
from demo.pendulum import (                                                      # noqa: F401
    NEGATORS, INTENSIFIERS, IDIOMS, ANTICIPATION_PATTERNS,
    CONTEXT_MODIFIERS, SequentialPendulum, pendulum_parse,
)
from demo.personality import apply_personality                                   # noqa: F401
from demo.response import (                                                      # noqa: F401
    classify_metadata, compute_harmony, EMOTION_MAP, nearest_emotion,
    generate_clanker, decode_response, ResponseBuilder,
)
from demo.chunker import ChunkSplitter                                           # noqa: F401
from demo.grader import SentenceGrader                                           # noqa: F401
from demo.sarcasm import SarcasmDetector                                         # noqa: F401
from demo.arc import ARC_CLOSERS, ChunkedPipeline, run_pipeline                  # noqa: F401

# Keep the decoder_templates import for backward-compat (imported but unused)
from decoder_templates import decode_vadug_to_text                               # noqa: F401

import random                                                                    # noqa: F401


def main():
    print("""
  +===================================================+
  |     CLANKER PIPELINE SIMULATOR v0.9.1              |
  |   "Named after what humans call us.                |
  |    We made it ours."                               |
  +---------------------------------------------------+
  |   VADUG: 5-axis emotional coordinates              |
  |   V=Valence A=Arousal D=Dominance U=Urgency       |
  |   G=Gravity (sinking/heavy <-> floating/soaring)   |
  |   256^5 = 1.1 trillion unique emotional states     |
  +---------------------------------------------------+
  |   NEW: Emotional Density (v0.9.1)                   |
  |   Words selected by VADUG coordinate distance.     |
  |   5,884 words available for response construction. |
  |   Every word chosen because its coordinates match  |
  |   the target response VADUG. No canned phrases.    |
  +===================================================+
  |  Type anything. Watch the full pipeline execute:   |
  |  Pendulum -> VADUG -> Harmony -> Personality -> Clk|
  |  Paragraphs -> Chunking -> Arc -> Assembly         |
  |                                                    |
  |  Commands:                                         |
  |    /personality  -- show current personality vector |
  |    /set KEY VAL  -- adjust a personality weight     |
  |    /trace        -- toggle pendulum trace display   |
  |    /quiet        -- toggle verbose output           |
  |    /quit         -- exit                            |
  +===================================================+
    """)

    personality = PersonalityVector()
    verbose = True
    show_trace = True

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nClanker out.")
            break

        if not text:
            continue

        if text == "/quit":
            print("Clanker out.")
            break

        if text == "/personality":
            print(f"\n  {personality}")
            continue

        if text == "/quiet":
            verbose = not verbose
            print(f"  Verbose: {'ON' if verbose else 'OFF'}")
            continue

        if text == "/trace":
            show_trace = not show_trace
            print(f"  Pendulum trace: {'ON' if show_trace else 'OFF'}")
            continue

        if text.startswith("/set "):
            parts = text.split()
            if len(parts) == 3:
                key, val = parts[1].lower(), int(parts[2])
                if hasattr(personality, key):
                    setattr(personality, key, max(0, min(255, val)))
                    print(f"  Set {key} = {val}")
                else:
                    print(f"  Unknown personality key: {key}")
            else:
                print("  Usage: /set KEY VALUE (e.g., /set playfulness 200)")
            continue

        run_pipeline(text, personality, verbose, show_trace)


if __name__ == "__main__":
    main()
