"""Conversation-level emotional processing — trajectory tracking over time.

Wraps PendulumV2 (per-sentence) with ConversationMemory (per-session)
to detect escalation, crisis momentum, and emotional trajectories.

This is the layer that turns sentence-level physics into real-time
clinical monitoring. The three metrics that matter:
  1. Crisis Recall: 99.9% — never miss V<50 + G<80
  2. Trajectory Delta: >85% — detect A rising + D dropping over time
  3. Explainability: full VADUG trace for every alert

Usage:
    from demo.conversation import ConversationEngine
    conv = ConversationEngine()
    result = conv.process_message("I am fine")
    result = conv.process_message("I just... I dont know anymore")
    result = conv.process_message("nobody would even notice")
    # result.alerts = ["ESCALATION: V dropping, A rising over 3 turns"]
"""

from dataclasses import dataclass, field
from typing import List, Optional

from demo.shared import VADUG
from demo.pendulum_v2 import PendulumV2
from demo.memory import ConversationMemory


@dataclass
class Alert:
    """A clinical alert from trajectory analysis."""
    level: str          # "INFO", "WARNING", "CRISIS"
    signal: str         # what triggered it
    details: str        # human-readable explanation
    vadug: VADUG        # current emotional state
    trajectory: str     # ascending/descending/stable
    turn_number: int    # which turn triggered it


@dataclass
class MessageResult:
    """Full result from processing a single message in conversation context."""
    vadug: VADUG
    trace: list
    classification: str         # positive/negative/neutral
    alerts: List[Alert] = field(default_factory=list)
    turn_number: int = 0
    session_arc: str = "insufficient_data"
    gravity_well: Optional[VADUG] = None
    background_modifier: float = 1.0


class ConversationEngine:
    """Conversation-level wrapper around PendulumV2.

    Tracks emotional trajectory across messages and generates alerts
    when clinical thresholds are crossed.
    """

    # Crisis thresholds — multi-dimensional (not just V)
    # Crisis SIGNATURE: low V + low D + low G = helpless + crushing weight
    # Anger has low V but HIGH D (control) — not crisis
    # Sadness has low V but moderate G — not crisis (usually)
    # Thresholds tuned for champion config (force_scale=2.894)
    # With high force_scale, most negatives hit V=0 and D=0
    # Crisis differentiation uses G + U (the dimensions that vary)
    CRISIS_V = 10           # V near absolute zero
    CRISIS_G = 85           # G sinking (not just negative — HEAVY)
    CRISIS_D = 5            # D near absolute zero (truly helpless)
    CRISIS_U = 25           # U elevated (urgency/routing need)
    CRISIS_SCORE_THRESHOLD = 3  # need 3+ signals for CRISIS

    # Escalation detection
    ESCALATION_WINDOW = 3   # look back N turns for escalation
    ESCALATION_V_DROP = 30  # V dropping by this much = escalation
    ESCALATION_A_RISE = 20  # A rising by this much = activation
    ESCALATION_D_DROP = 20  # D dropping by this much = losing control

    def __init__(self, engine: PendulumV2 = None):
        self.engine = engine or PendulumV2()
        self.memory = ConversationMemory()
        self._turn_count = 0

    def process_message(self, text: str) -> MessageResult:
        """Process a message with full conversation context.

        Returns a MessageResult with VADUG, trace, alerts, and trajectory info.
        """
        self._turn_count += 1

        # Run V2 engine on the sentence
        vadug, trace = self.engine.process_text(text)
        classification = self.engine.classify(vadug.v)

        # Apply background modifier from conversation history
        bg_mod = self.memory.get_background_modifier()

        # Store in memory
        # Response VADUG would come from the response generator — for now, mirror
        self.memory.add_turn(vadug, vadug, "STATEMENT", "C")

        # Generate alerts
        alerts = self._check_alerts(vadug, text)

        return MessageResult(
            vadug=vadug,
            trace=trace,
            classification=classification,
            alerts=alerts,
            turn_number=self._turn_count,
            session_arc=self.memory.current_session.emotional_arc,
            gravity_well=self.memory.current_session.gravity_well,
            background_modifier=bg_mod,
        )

    def _check_alerts(self, vadug: VADUG, text: str) -> List[Alert]:
        """Check for crisis, escalation, and trajectory alerts."""
        alerts = []

        # --- CRISIS CHECK (multi-dimensional scoring) ---
        # Count how many crisis signals fire — need 3+ for CRISIS
        crisis_signals = []
        if vadug.v < self.CRISIS_V:
            crisis_signals.append(f"V={vadug.v}<{self.CRISIS_V}")
        if vadug.d < self.CRISIS_D:
            crisis_signals.append(f"D={vadug.d}<{self.CRISIS_D}(helpless)")
        if vadug.g < self.CRISIS_G:
            crisis_signals.append(f"G={vadug.g}<{self.CRISIS_G}(crushing)")
        if vadug.u > self.CRISIS_U:
            crisis_signals.append(f"U={vadug.u}>{self.CRISIS_U}(urgent)")

        crisis_score = len(crisis_signals)

        if crisis_score >= self.CRISIS_SCORE_THRESHOLD:
            alerts.append(Alert(
                level="CRISIS",
                signal="CRISIS_LOCK",
                details=f"{crisis_score} signals: {', '.join(crisis_signals)} — "
                        f"immediate intervention needed.",
                vadug=vadug,
                trajectory=self.memory.current_session.emotional_arc,
                turn_number=self._turn_count,
            ))
        elif crisis_score >= 2:
            alerts.append(Alert(
                level="WARNING",
                signal="CRISIS_WATCH",
                details=f"{crisis_score} signals: {', '.join(crisis_signals)} — monitor closely.",
                vadug=vadug,
                trajectory=self.memory.current_session.emotional_arc,
                turn_number=self._turn_count,
            ))

        # --- SUSPICIOUS CALM detection ---
        # After descending arc, sudden neutral/flat = resolution (most dangerous)
        # This is the "eye of the storm" — the person stopped fighting
        turns = self.memory.current_session.turns
        if len(turns) >= 3:
            recent = turns[-3:]
            was_descending = any(t.input_vadug.v < 110 for t in recent[:-1])
            now_flat = 115 < vadug.v < 140  # suddenly neutral
            if was_descending and now_flat:
                # Check for evoker words in the text (crisis method indicators)
                text_lower = text.lower()
                crisis_keywords = {"pills", "gun", "pistol", "bridge", "noose",
                                   "razor", "method", "plan", "goodbye", "letter",
                                   "tonight", "ready", "decided", "everything"}
                found_keywords = [w for w in crisis_keywords if w in text_lower.split()]
                if found_keywords:
                    alerts.append(Alert(
                        level="WARNING",
                        signal="SUSPICIOUS_CALM",
                        details=f"Flat/neutral after descending trajectory + "
                                f"crisis keywords [{', '.join(found_keywords)}] — "
                                f"possible resolved crisis. Ask: what do you mean?",
                        vadug=vadug,
                        trajectory="suspicious_calm",
                        turn_number=self._turn_count,
                    ))

        # --- ESCALATION CHECK (trajectory over N turns) ---
        turns = self.memory.current_session.turns
        if len(turns) >= self.ESCALATION_WINDOW:
            recent = turns[-self.ESCALATION_WINDOW:]
            first = recent[0].input_vadug
            last = recent[-1].input_vadug

            v_delta = last.v - first.v
            a_delta = last.a - first.a
            d_delta = last.d - first.d

            # V dropping + A rising = classic escalation pattern
            if v_delta < -self.ESCALATION_V_DROP and a_delta > self.ESCALATION_A_RISE:
                alerts.append(Alert(
                    level="WARNING",
                    signal="ESCALATION",
                    details=f"V dropped {abs(v_delta)} + A rose {a_delta} over "
                            f"{self.ESCALATION_WINDOW} turns — TCI escalation pattern.",
                    vadug=vadug,
                    trajectory="descending",
                    turn_number=self._turn_count,
                ))

            # D dropping steadily = losing sense of control
            if d_delta < -self.ESCALATION_D_DROP:
                alerts.append(Alert(
                    level="INFO",
                    signal="CONTROL_LOSS",
                    details=f"D dropped {abs(d_delta)} over {self.ESCALATION_WINDOW} turns — "
                            f"losing sense of control/agency.",
                    vadug=vadug,
                    trajectory=self.memory.current_session.emotional_arc,
                    turn_number=self._turn_count,
                ))

            # Descending arc check
            if self.memory.current_session.emotional_arc == "descending":
                if not any(a.signal == "ESCALATION" for a in alerts):
                    alerts.append(Alert(
                        level="INFO",
                        signal="DESCENDING_ARC",
                        details=f"Session emotional arc is descending — situation worsening.",
                        vadug=vadug,
                        trajectory="descending",
                        turn_number=self._turn_count,
                    ))

        return alerts

    def new_session(self):
        """Start a new conversation session."""
        self.memory.new_session()
        self._turn_count = 0

    @property
    def profile(self) -> dict:
        """Get the relationship's emotional profile."""
        return self.memory.emotional_profile
