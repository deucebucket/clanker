"""Clanker-Lang Three-Layer API — The production architecture.

Layer 1: Sentence Physics (0.1ms) — PendulumV2 with 25 forces
Layer 2: Conversation Trajectory (minutes) — ConversationEngine with escalation detection
Layer 3: Relationship / Dark Matter (days/weeks) — entity profiles, baseline drift

Usage:
    from demo.clanker_api import ClankerAPI

    # Create an API instance for a specific entity
    api = ClankerAPI(entity_id="resident_042")

    # Process messages over time
    r1 = api.process("I had an okay day")
    r2 = api.process("Something happened that bothered me")
    r3 = api.process("I am getting really frustrated")
    r4 = api.process("I just want it all to stop")
    # r4.crisis = True, r4.alerts = [ESCALATION, CRISIS_LOCK]

    # Get the entity's emotional profile
    profile = api.get_profile()
    # profile.traits = ["descending_arc", "crisis_history(1)"]

    # A + B = C
    # A = everything before (Dark Matter + conversation history)
    # B = current input
    # C = what it actually means for THIS person at THIS moment
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from demo.shared import VADUG
from demo.pendulum_v2 import PendulumV2
from demo.conversation import ConversationEngine, Alert, MessageResult
from demo.memory import ConversationMemory
from demo.preflight import PreflightAnalyzer


# Champion config (EXP-0059, 27 params, 56M evaluations)
CHAMPION_CONFIG = dict(
    momentum=0.903, force_scale=2.894, direct_push_cap=0.085,
    direct_push_trigger=158.7, crisis_v=68.9, crisis_u=95.5,
    question_dampener=0.529, scale_v=1.920, scale_a=0.789,
    scale_d=2.901, scale_u=0.30, scale_g=0.707,
    threshold_low=116.7, threshold_high=155.2,
    negation_decay_operator=0.850, negation_decay_neutral=0.60,
    negation_decay_payload=0.523,
    evoker_decay=0.704, conditional_dampener=0.339,
    clinical_dampener=0.319, passive_d_offset=-21.6,
    comparative_amplifier=1.152, superlative_amplifier=2.313,
    filler_d_offset=-6.0, litotes_dampener=0.825,
    weaponized_d_boost=9.6, tag_d_offset=-7.8,
)


@dataclass
class ProcessResult:
    """Full three-layer result from processing a message."""
    # Layer 1: Sentence Physics
    vadug: VADUG
    trace: list
    classification: str

    # Layer 2: Conversation
    turn_number: int = 0
    session_arc: str = "insufficient_data"
    alerts: List[Alert] = field(default_factory=list)
    crisis: bool = False

    # Layer 3: Dark Matter context
    entity_id: str = ""
    background_modifier: float = 1.0
    gravity_well: Optional[VADUG] = None
    relationship_traits: list = field(default_factory=list)

    # Timing
    processing_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "vadug": {"v": self.vadug.v, "a": self.vadug.a, "d": self.vadug.d,
                      "u": self.vadug.u, "g": self.vadug.g},
            "classification": self.classification,
            "description": self.vadug.describe(),
            "crisis": self.crisis,
            "alerts": [{"level": a.level, "signal": a.signal, "details": a.details}
                       for a in self.alerts],
            "turn_number": self.turn_number,
            "session_arc": self.session_arc,
            "entity_id": self.entity_id,
            "background_modifier": self.background_modifier,
            "traits": self.relationship_traits,
            "processing_ms": round(self.processing_ms, 3),
            "trace": [{"word": t["word"], "role": t["role"],
                        "v": round(t["v"], 1), "note": t.get("note", "")[:50]}
                       for t in self.trace],
        }


class ClankerAPI:
    """Three-layer emotional physics API.

    One instance per entity (person being monitored).
    Tracks their emotional trajectory across messages and sessions.
    """

    def __init__(self, entity_id: str = "anonymous", config: dict = None):
        self.entity_id = entity_id
        engine_config = config or CHAMPION_CONFIG
        self.engine = PendulumV2(**engine_config)
        self.conversation = ConversationEngine(self.engine)
        self.preflight = PreflightAnalyzer()
        self._message_count = 0

    def process(self, text: str) -> ProcessResult:
        """Process a message through all three layers.

        Layer 1: Sentence physics (25 forces, VADUG output)
        Layer 2: Conversation trajectory (escalation detection, arc tracking)
        Layer 3: Dark Matter context (background modifier, relationship traits)
        """
        t0 = time.perf_counter()
        self._message_count += 1

        # Layer 2 wraps Layer 1 (ConversationEngine calls PendulumV2 internally)
        msg_result = self.conversation.process_message(text)

        # Layer 3: Dark Matter context
        bg_mod = self.conversation.memory.get_background_modifier()
        profile = self.conversation.memory.emotional_profile
        gravity_well = self.conversation.memory.current_session.gravity_well

        # Check for crisis in alerts
        is_crisis = any(a.level == "CRISIS" for a in msg_result.alerts)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ProcessResult(
            # Layer 1
            vadug=msg_result.vadug,
            trace=msg_result.trace,
            classification=msg_result.classification,
            # Layer 2
            turn_number=msg_result.turn_number,
            session_arc=msg_result.session_arc,
            alerts=msg_result.alerts,
            crisis=is_crisis,
            # Layer 3
            entity_id=self.entity_id,
            background_modifier=bg_mod,
            gravity_well=gravity_well,
            relationship_traits=profile.get("traits", []),
            # Timing
            processing_ms=elapsed_ms,
        )

    def new_session(self):
        """Start a new conversation session (preserves relationship history)."""
        self.conversation.new_session()

    def get_profile(self) -> dict:
        """Get the entity's full emotional profile."""
        return {
            "entity_id": self.entity_id,
            "messages_processed": self._message_count,
            **self.conversation.memory.emotional_profile,
        }

    def save_state(self, path: str):
        """Save entity state to disk."""
        self.conversation.memory.save(path)

    def export_session(self) -> list:
        """Export current session as a list of VADUG snapshots."""
        return [
            {"turn": t.turn_number, "vadug": str(t.input_vadug),
             "intent": t.intent_mode, "grade": t.grade}
            for t in self.conversation.memory.current_session.turns
        ]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def demo():
    """Run a demo escalation scenario."""
    api = ClankerAPI(entity_id="resident_042")

    messages = [
        "I had an okay day today",
        "Something happened at work that bothered me",
        "I am getting really frustrated with everything",
        "Nobody listens to me and nothing ever changes",
        "I just want to give up on everything",
        "What is even the point anymore",
        "I just want it all to stop",
    ]

    print("=" * 70)
    print("  CLANKER API — Three-Layer Emotional Physics")
    print("  Entity: resident_042")
    print("=" * 70)

    for msg in messages:
        r = api.process(msg)
        alert_str = ""
        for a in r.alerts:
            alert_str += f" [{a.level}:{a.signal}]"

        crisis_marker = " *** CRISIS ***" if r.crisis else ""
        print(f"\n  Turn {r.turn_number}: \"{msg}\"")
        print(f"    VADUG: V={r.vadug.v:3d} A={r.vadug.a:3d} D={r.vadug.d:3d} "
              f"U={r.vadug.u:3d} G={r.vadug.g:3d}")
        print(f"    Arc: {r.session_arc}  |  {r.classification}  "
              f"|  {r.processing_ms:.1f}ms{alert_str}{crisis_marker}")

    print(f"\n{'=' * 70}")
    print(f"  PROFILE: {json.dumps(api.get_profile(), indent=2)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo()
