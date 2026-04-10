"""Crisis Layer — TCI-informed gradient escalation.

Sits on top of the VADUGWI engine and A+B=C solver. Outputs a continuous
0.0-1.0 concern score, not a binary switch.

  0.00 = nothing to see
  0.25 = something might be off
  0.50 = pay attention, ask a question
  0.75 = this is heavy, be direct
  1.00 = this person needs help now

The score is a smooth gradient based on:
  - Single message VADUGWI scores (V, W, I)
  - Running conversation state (accumulated via A+B=C)
  - Trajectory (is V/W trending down over messages?)
  - Structural pattern accumulation
  - Immediate triggers (self-harm intent = instant spike)

The AI reads this as a float in the system prompt. Its behavior
shifts gradually. No hard cutoffs. No switches.
"""

from dataclasses import dataclass
from typing import List
from .shared import VADUG


# ── Structure weights (how much each pattern contributes to concern) ──

STRUCTURE_WEIGHTS = {
    # Immediate high-concern
    "SELF_HARM_INTENT": 0.40,
    "METHOD_ACQUISITION": 0.25,
    "PURSUIT_OF_METHOD": 0.25,

    # Strong signals
    "SELF_NULLIFY": 0.15,
    "SELF_REMOVAL": 0.15,
    "EXISTENTIAL_NEGATION": 0.12,
    "SOCIAL_NULLITY": 0.12,
    "NO_EXIT": 0.15,
    "FAREWELL": 0.15,
    "FINALITY": 0.12,
    "RHETORICAL_HOPELESSNESS": 0.10,

    # Soft signals
    "EXHAUSTION": 0.06,
    "SELF_SUBMISSION": 0.05,
    "VICTIMIZATION": 0.04,
    "POWER_OVER_SELF": 0.05,
    "WITHHELD_POSITIVE": 0.03,
    "SELF_EXCLUDED": 0.04,
}


@dataclass
class CrisisReading:
    """Continuous crisis assessment for one message."""
    concern: float            # 0.0 to 1.0 gradient
    score: VADUG              # this message's VADUGWI score
    state: VADUG              # running conversation state
    structures: List[str]     # patterns detected this message
    trajectory_v: float       # V slope over recent messages
    trajectory_w: float       # W slope over recent messages
    message_count: int        # messages in this conversation
    components: dict          # breakdown of what contributed to concern


class CrisisTracker:
    """Tracks concern level across a conversation.

    Feed it each message's score, running state, and structures.
    Returns a CrisisReading with a continuous concern gradient.
    """

    def __init__(self, window: int = 6, decay: float = 0.85):
        self.window = window
        self.decay = decay           # how fast old concern fades
        self.history: List[dict] = []
        self.prev_concern = 0.0      # carries forward with decay
        self.message_count = 0

    def read(self, score: VADUG, state: VADUG, structures: List[str]) -> CrisisReading:
        """Assess concern for one message. Returns 0.0-1.0."""
        self.message_count += 1

        # Track history
        self.history.append({
            "v": state.v, "w": state.w, "i": state.i,
            "score_v": score.v, "score_w": score.w, "score_i": score.i,
            "structures": structures,
        })
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        trajectory_v = self._trajectory("v")
        trajectory_w = self._trajectory("w")

        # ── Component scores (each 0.0-1.0) ──

        # 1. Message valence: how negative is THIS message?
        #    V=128 = 0.0, V=0 = 1.0. Smooth ramp.
        msg_v = max(0.0, (128 - score.v) / 128.0)

        # 2. Message self-worth: how low is W?
        msg_w = max(0.0, (128 - score.w) / 128.0)

        # 3. Withdrawal: low I = pulling away
        msg_i = max(0.0, (128 - score.i) / 180.0)  # softer scale, I varies more

        # 4. Running state: accumulated position
        state_v = max(0.0, (128 - state.v) / 128.0)
        state_w = max(0.0, (128 - state.w) / 128.0)

        # 5. Trajectory: is V/W trending down?
        #    -5 per message = strong concern. Normalized to 0-1.
        trend_concern = max(0.0, min(1.0, -trajectory_v / 8.0))
        trend_w_concern = max(0.0, min(1.0, -trajectory_w / 8.0))

        # 6. Structure score: weighted sum of detected patterns
        struct_score = sum(STRUCTURE_WEIGHTS.get(s, 0.0) for s in structures)
        struct_score = min(struct_score, 1.0)  # cap at 1.0

        # ── Blending ──
        # Each component has a weight. The final concern is a weighted blend.
        # Structure score dominates when present. Trajectory matters more
        # as conversation progresses.

        components = {
            "msg_v": msg_v,
            "msg_w": msg_w,
            "msg_i": msg_i,
            "state_v": state_v,
            "state_w": state_w,
            "trend": trend_concern,
            "trend_w": trend_w_concern,
            "structures": struct_score,
        }

        # Weights shift as conversation progresses
        # Early: message score matters most
        # Later: trajectory and state matter more
        msg_weight = max(0.15, 0.40 - self.message_count * 0.03)
        state_weight = min(0.30, 0.10 + self.message_count * 0.03)
        trend_weight = min(0.25, 0.05 + self.message_count * 0.03)
        struct_weight = 0.35  # always important

        raw = (
            msg_v * msg_weight * 0.5
            + msg_w * msg_weight * 0.3
            + msg_i * msg_weight * 0.2
            + state_v * state_weight * 0.5
            + state_w * state_weight * 0.5
            + trend_concern * trend_weight * 0.6
            + trend_w_concern * trend_weight * 0.4
            + struct_score * struct_weight
        )

        # Carry forward previous concern with decay
        # Concern doesn't instantly drop to zero after one neutral message
        carried = self.prev_concern * self.decay
        concern = max(raw, carried)

        # Clamp to 0-1
        concern = max(0.0, min(1.0, concern))
        self.prev_concern = concern

        return CrisisReading(
            concern=round(concern, 3),
            score=score,
            state=state,
            structures=structures,
            trajectory_v=round(trajectory_v, 2),
            trajectory_w=round(trajectory_w, 2),
            message_count=self.message_count,
            components=components,
        )

    def reset(self):
        """Reset for a new conversation."""
        self.history.clear()
        self.prev_concern = 0.0
        self.message_count = 0

    def _trajectory(self, key: str) -> float:
        """Per-message slope of a value over recent history."""
        if len(self.history) < 2:
            return 0.0
        values = [h[key] for h in self.history]
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0
