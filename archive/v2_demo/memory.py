"""VADUG conversation memory — gravitational signatures that replace context windows.

Stores emotional trajectory across conversation turns as VADUG snapshots.
Each turn = 5 bytes. Computes session gravity wells and relationship-level
emotional profiles.
"""

from dataclasses import dataclass, field

from .shared import VADUG


@dataclass
class TurnMemory:
    """One conversation turn's emotional state."""
    turn_number: int
    input_vadug: VADUG
    response_vadug: VADUG
    intent_mode: str
    grade: str


@dataclass
class SessionMemory:
    """Emotional trajectory of one conversation session."""
    turns: list[TurnMemory] = field(default_factory=list)

    def add_turn(self, input_vadug: VADUG, response_vadug: VADUG,
                 intent_mode: str, grade: str):
        turn = TurnMemory(
            turn_number=len(self.turns) + 1,
            input_vadug=input_vadug,
            response_vadug=response_vadug,
            intent_mode=intent_mode,
            grade=grade,
        )
        self.turns.append(turn)

    @property
    def gravity_well(self) -> VADUG:
        """Average VADUG across all turns — the session's emotional center."""
        if not self.turns:
            return VADUG()
        n = len(self.turns)
        return VADUG(
            v=sum(t.input_vadug.v for t in self.turns) // n,
            a=sum(t.input_vadug.a for t in self.turns) // n,
            d=sum(t.input_vadug.d for t in self.turns) // n,
            u=sum(t.input_vadug.u for t in self.turns) // n,
            g=sum(t.input_vadug.g for t in self.turns) // n,
        )

    @property
    def emotional_arc(self) -> str:
        """Detect the session's overall emotional trajectory."""
        if len(self.turns) < 2:
            return "insufficient_data"
        first_half = self.turns[:len(self.turns) // 2]
        second_half = self.turns[len(self.turns) // 2:]
        first_v = sum(t.input_vadug.v for t in first_half) / len(first_half)
        second_v = sum(t.input_vadug.v for t in second_half) / len(second_half)
        diff = second_v - first_v
        if diff > 15:
            return "ascending"   # getting better
        elif diff < -15:
            return "descending"  # getting worse
        return "stable"

    @property
    def crisis_count(self) -> int:
        """How many turns triggered crisis."""
        return sum(1 for t in self.turns if t.grade in ("F", "F-", "F+"))

    def to_bytes(self) -> bytes:
        """Serialize session to minimal bytes: 5 bytes per turn."""
        data = []
        for t in self.turns:
            data.extend(t.input_vadug.to_bytes())
        return bytes(data)

    @property
    def byte_size(self) -> int:
        """How many bytes this session's emotional memory takes."""
        return len(self.turns) * 5


class ConversationMemory:
    """Multi-session emotional memory with gravitational profiles."""

    def __init__(self):
        self.sessions: list[SessionMemory] = []
        self.current_session: SessionMemory = SessionMemory()

    def new_session(self):
        """Start a new session, archive the current one."""
        if self.current_session.turns:
            self.sessions.append(self.current_session)
        self.current_session = SessionMemory()

    def add_turn(self, input_vadug, response_vadug,
                 intent_mode="STATEMENT", grade="C"):
        self.current_session.add_turn(
            input_vadug, response_vadug, intent_mode, grade)

    @property
    def relationship_gravity(self) -> VADUG:
        """Weighted average across all sessions — the relationship's emotional
        baseline.  Recent sessions weighted more heavily."""
        all_sessions = self.sessions + (
            [self.current_session] if self.current_session.turns else [])
        if not all_sessions:
            return VADUG()

        total_weight = 0
        weighted_v, weighted_a, weighted_d, weighted_u, weighted_g = (
            0, 0, 0, 0, 0)

        for i, session in enumerate(all_sessions):
            if not session.turns:
                continue
            weight = i + 1  # more recent = higher weight
            well = session.gravity_well
            weighted_v += well.v * weight
            weighted_a += well.a * weight
            weighted_d += well.d * weight
            weighted_u += well.u * weight
            weighted_g += well.g * weight
            total_weight += weight

        if total_weight == 0:
            return VADUG()

        return VADUG(
            v=int(weighted_v / total_weight),
            a=int(weighted_a / total_weight),
            d=int(weighted_d / total_weight),
            u=int(weighted_u / total_weight),
            g=int(weighted_g / total_weight),
        )

    def get_background_modifier(self) -> float:
        """Compute how background gravity modifies current processing.

        Heavy history (G < 80)  -> forces land harder (multiplier > 1.0)
        Light history (G > 180) -> forces dampened  (multiplier < 1.0)
        Neutral history (G ~128) -> no modification (multiplier = 1.0)
        """
        rg = self.relationship_gravity
        return 1.0 + (128 - rg.g) / 256

    @property
    def emotional_profile(self) -> dict:
        """Human-readable emotional profile of this relationship."""
        rg = self.relationship_gravity
        profile = {
            "sessions": len(self.sessions) + (
                1 if self.current_session.turns else 0),
            "total_turns": (sum(len(s.turns) for s in self.sessions)
                           + len(self.current_session.turns)),
            "relationship_vadug": str(rg),
            "background_modifier": round(self.get_background_modifier(), 3),
            "total_bytes": (sum(s.byte_size for s in self.sessions)
                           + self.current_session.byte_size),
        }

        # Characterize
        traits = []
        if rg.v < 90:
            traits.append("carries_pain")
        elif rg.v > 170:
            traits.append("generally_positive")
        if rg.g < 80:
            traits.append("heavy_history")
        elif rg.g > 180:
            traits.append("emotionally_buoyant")
        if rg.a > 170:
            traits.append("high_intensity")
        elif rg.a < 90:
            traits.append("calm_presence")
        if rg.d < 80:
            traits.append("feels_helpless")
        elif rg.d > 180:
            traits.append("in_control")

        total_crises = (sum(s.crisis_count for s in self.sessions)
                        + self.current_session.crisis_count)
        if total_crises > 0:
            traits.append(f"crisis_history({total_crises})")

        profile["traits"] = traits
        return profile

    def save(self, path: str):
        """Save memory to JSON."""
        import json
        data = {
            "sessions": [
                {
                    "gravity_well": str(s.gravity_well),
                    "arc": s.emotional_arc,
                    "turns": len(s.turns),
                    "crisis_count": s.crisis_count,
                    "bytes": list(s.to_bytes()),
                }
                for s in self.sessions
            ],
            "current_turns": len(self.current_session.turns),
            "relationship_gravity": str(self.relationship_gravity),
            "profile": self.emotional_profile,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
