"""Entity Interaction — two entities affecting each other in VADUG space.

Two engines with different personalities process each other's output.
The emotional physics determine what happens — not a script.

Usage:
    from demo.interaction import Interaction

    scene = Interaction(
        entity_a=("hothead", PersonalityVector(assertiveness=250, safety=40, ...)),
        entity_b=("anxious_kid", PersonalityVector(assertiveness=20, safety=240, ...)),
    )

    scene.exchange("hothead", "Move out of my way")
    scene.exchange("anxious_kid", "Sorry I didnt mean to")
    scene.exchange("hothead", "You are always in the way")

    print(scene.report())
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from demo.shared import PersonalityVector, VADUG
from demo.clanker_api import ClankerAPI, CHAMPION_CONFIG
from demo.zones import ZoneClassifier


@dataclass
class Turn:
    """One turn in an interaction."""
    speaker: str
    text: str
    speaker_zone: str
    speaker_vadug: VADUG
    receiver_zone: str
    receiver_vadug: VADUG
    alerts: list
    escalation: bool = False


class Interaction:
    """Two entities in emotional conversation."""

    def __init__(self, entity_a: tuple, entity_b: tuple,
                 config: dict = None):
        """Create an interaction between two entities.

        Args:
            entity_a: (name, PersonalityVector) or (name, PersonalityVector, dark_matter_profile)
            entity_b: same format
        """
        name_a, pv_a = entity_a[0], entity_a[1]
        name_b, pv_b = entity_b[0], entity_b[1]
        dm_a = entity_a[2] if len(entity_a) > 2 else "default"
        dm_b = entity_b[2] if len(entity_b) > 2 else "default"

        self.name_a = name_a
        self.name_b = name_b

        self.api_a = ClankerAPI(entity_id=name_a, config=config,
                                personality=pv_a, dark_matter_profile=dm_a)
        self.api_b = ClankerAPI(entity_id=name_b, config=config,
                                personality=pv_b, dark_matter_profile=dm_b)

        self.zone_classifier = ZoneClassifier()
        self.turns: List[Turn] = []

    def exchange(self, speaker: str, text: str) -> Turn:
        """One entity speaks, both are affected.

        The speaker processes their own words (feeling their own emotion).
        The receiver processes the same words through their personality filter.
        """
        if speaker == self.name_a:
            speaker_api = self.api_a
            receiver_api = self.api_b
        else:
            speaker_api = self.api_b
            receiver_api = self.api_a

        r_speaker = speaker_api.process(text)
        r_receiver = receiver_api.process(text)

        speaker_zone = self.zone_classifier.classify(r_speaker.vadug)
        receiver_zone = self.zone_classifier.classify(r_receiver.vadug)

        all_alerts = r_speaker.alerts + r_receiver.alerts
        escalation = any(a.level in ("WARNING", "CRISIS") for a in all_alerts)

        turn = Turn(
            speaker=speaker,
            text=text,
            speaker_zone=speaker_zone.zone,
            speaker_vadug=r_speaker.vadug,
            receiver_zone=receiver_zone.zone,
            receiver_vadug=r_receiver.vadug,
            alerts=[f"[{a.level}:{a.signal}]" for a in all_alerts
                    if a.level in ("WARNING", "CRISIS")],
            escalation=escalation,
        )

        self.turns.append(turn)
        return turn

    def report(self) -> str:
        """Generate a human-readable interaction report."""
        lines = [
            f"=== INTERACTION: {self.name_a} vs {self.name_b} ===",
            f"  {self.name_a}: sensitivity={self.api_a.engine.personality.emotional_sensitivity:.2f}x"
            if self.api_a.engine.personality else "",
            f"  {self.name_b}: sensitivity={self.api_b.engine.personality.emotional_sensitivity:.2f}x"
            if self.api_b.engine.personality else "",
            "",
        ]

        for i, t in enumerate(self.turns):
            esc = " *** ESCALATION ***" if t.escalation else ""
            alerts = " ".join(t.alerts) if t.alerts else ""

            lines.append(f"  T{i+1} [{t.speaker}]: \"{t.text}\"")
            lines.append(f"    Speaker: {t.speaker_zone:12s} V={t.speaker_vadug.v:3d} D={t.speaker_vadug.d:3d}")
            lines.append(f"    Receiver: {t.receiver_zone:12s} V={t.receiver_vadug.v:3d} D={t.receiver_vadug.d:3d}")
            if alerts:
                lines.append(f"    {alerts}{esc}")
            lines.append("")

        # Final state
        dm_a = self.api_a.dark_matter.state
        dm_b = self.api_b.dark_matter.state
        lines.append(f"  FINAL STATE:")
        lines.append(f"    {self.name_a}: drift={dm_a['drift']:+.2f} ({dm_a['disposition']})")
        lines.append(f"    {self.name_b}: drift={dm_b['drift']:+.2f} ({dm_b['disposition']})")

        escalation_count = sum(1 for t in self.turns if t.escalation)
        lines.append(f"    Escalation events: {escalation_count}/{len(self.turns)}")

        return "\n".join(lines)
