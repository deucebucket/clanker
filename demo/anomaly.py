"""Emotional anomaly detector — the Black Hole concept.

Finds emotional anomalies by looking at the ORBIT of the words,
not just the words themselves. Works with ConversationEngine to
detect patterns that surface-level sentiment analysis misses.

Four detection modes:
  1. Gravity Wells (Topic Deflection)  — evoker fires then sudden neutral shift
  2. Gravitational Lensing (Masking)   — positive words with heavy/helpless VADUG
  3. Velocity Anomalies (Behavioral)   — message length, punctuation, formality shifts
  4. Resonance Patterns (Loops)        — same emotional signature repeating

Usage:
    from demo.conversation import ConversationEngine
    from demo.anomaly import AnomalyDetector

    conv = ConversationEngine()
    detector = AnomalyDetector(conv)

    for msg in messages:
        result = detector.process_message(msg)
        if result.anomalies:
            for a in result.anomalies:
                print(f"[{a.type}] severity={a.severity} — {a.description}")
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re
import string

from demo.shared import VADUG
from demo.conversation import ConversationEngine, MessageResult


# ---------------------------------------------------------------------------
# Anomaly data structures
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A single detected emotional anomaly."""
    type: str           # DEFLECTION, MASKING, VELOCITY, RESONANCE
    severity: int       # 1-5 (1=subtle, 5=critical)
    description: str    # human-readable explanation
    evidence: dict      # VADUG data + whatever triggered it


@dataclass
class AnomalyResult:
    """Result from processing a message through the anomaly detector."""
    message_result: MessageResult       # the underlying engine result
    anomalies: List[Anomaly] = field(default_factory=list)
    turn_number: int = 0


# ---------------------------------------------------------------------------
# Internal bookkeeping for a single turn
# ---------------------------------------------------------------------------

@dataclass
class _TurnRecord:
    """Internal record for tracking behavioral baselines."""
    turn_number: int
    text: str
    vadug: VADUG
    trace: list
    word_count: int
    char_count: int
    punctuation_density: float      # punctuation chars / total chars
    has_evoker: bool                # did an evoker fire this turn?
    evoker_words: list              # which evokers fired
    formality_score: float          # 0=casual, 1=formal


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detects emotional anomalies across conversation turns.

    Wraps a ConversationEngine and adds orbital analysis on top
    of the per-message VADUG processing.
    """

    # --- Gravity Well (Deflection) thresholds ---
    DEFLECTION_WINDOW = 3           # check this many turns after evoker
    DEFLECTION_V_NEUTRAL_LOW = 110  # V above this = neutral territory
    DEFLECTION_V_NEUTRAL_HIGH = 145
    DEFLECTION_MIN_VELOCITY = 2     # short messages after evoker = deflection

    # --- Gravitational Lensing (Masking) thresholds ---
    MASKING_V_FLOOR = 128           # words read as positive (V >= this)
    MASKING_D_CEILING = 80          # but D is low (no agency)
    MASKING_G_CEILING = 90          # and G is heavy (sinking)

    # --- Velocity thresholds ---
    VELOCITY_MIN_TURNS = 4          # need baseline before detecting
    VELOCITY_LENGTH_RATIO = 0.35    # message < 35% of baseline = compression
    VELOCITY_LENGTH_SURGE = 2.5     # message > 250% of baseline = flood
    VELOCITY_PUNCT_SHIFT = 0.08     # punctuation density change threshold
    VELOCITY_FORMALITY_SHIFT = 0.4  # formality score change threshold

    # --- Resonance thresholds ---
    RESONANCE_WINDOW = 4            # look back N turns for loops
    RESONANCE_V_TOLERANCE = 8       # V within this range = same signature
    RESONANCE_FULL_TOLERANCE = 12   # all 5 dims within this = same state
    RESONANCE_OSCILLATION_BAND = 15 # V swinging between two values

    def __init__(self, engine: ConversationEngine):
        self.engine = engine
        self._history: List[_TurnRecord] = []
        self._turn_count = 0

    def process_message(self, text: str) -> AnomalyResult:
        """Process a message through both the engine and anomaly detection.

        Returns an AnomalyResult containing the normal MessageResult plus
        any detected anomalies.
        """
        # Run the underlying conversation engine
        msg_result = self.engine.process_message(text)
        self._turn_count += 1

        # Build internal record
        record = self._build_record(text, msg_result)
        self._history.append(record)

        # Run all four detectors
        anomalies: List[Anomaly] = []
        anomalies.extend(self._detect_deflection(record))
        anomalies.extend(self._detect_masking(record))
        anomalies.extend(self._detect_velocity(record))
        anomalies.extend(self._detect_resonance(record))

        return AnomalyResult(
            message_result=msg_result,
            anomalies=anomalies,
            turn_number=self._turn_count,
        )

    def reset(self):
        """Reset detector state (e.g., new conversation)."""
        self._history.clear()
        self._turn_count = 0
        self.engine.new_session()

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    def _build_record(self, text: str, result: MessageResult) -> _TurnRecord:
        """Extract behavioral metrics from a message."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        punct_chars = sum(1 for c in text if c in string.punctuation)
        punct_density = punct_chars / max(char_count, 1)

        # Check trace for evoker firings
        evoker_words = []
        for entry in result.trace:
            if entry.get("role", "").startswith("EVOKER"):
                evoker_words.append(entry["word"])

        formality = self._estimate_formality(text)

        return _TurnRecord(
            turn_number=self._turn_count,
            text=text,
            vadug=result.vadug,
            trace=result.trace,
            word_count=word_count,
            char_count=char_count,
            punctuation_density=punct_density,
            has_evoker=len(evoker_words) > 0,
            evoker_words=evoker_words,
            formality_score=formality,
        )

    @staticmethod
    def _estimate_formality(text: str) -> float:
        """Rough formality estimate: 0.0 = very casual, 1.0 = very formal.

        Signals: contractions (-0.15), slang (-0.15), all-caps words (-0.1),
        complete sentences (+0.1), polite markers (+0.15), hedging (+0.1).
        """
        score = 0.5  # baseline
        lower = text.lower()

        # Casual markers
        contractions = re.findall(
            r"\b(?:i'm|don't|can't|won't|didn't|isn't|aren't|wasn't|weren't|"
            r"shouldn't|couldn't|wouldn't|haven't|hasn't|hadn't|it's|that's|"
            r"there's|here's|what's|who's|let's|gonna|wanna|gotta|dunno|"
            r"idk|lol|lmao|omg|bruh|nah|yeah|yep|nope)\b",
            lower,
        )
        score -= min(len(contractions) * 0.1, 0.3)

        # All-caps words (shouting)
        caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
        score -= min(len(caps_words) * 0.05, 0.15)

        # Formal markers
        formal_markers = re.findall(
            r"\b(?:however|therefore|furthermore|nevertheless|regarding|"
            r"consequently|additionally|moreover|indeed|certainly|"
            r"perhaps|please|thank you|appreciate|sincerely|respectively)\b",
            lower,
        )
        score += min(len(formal_markers) * 0.1, 0.3)

        # Hedging / distancing language
        hedges = re.findall(
            r"\b(?:one might|it appears|it seems|it would seem|"
            r"i suppose|i believe|i think perhaps)\b",
            lower,
        )
        score += min(len(hedges) * 0.1, 0.2)

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Detector 1: Gravity Wells (Topic Deflection)
    # ------------------------------------------------------------------

    def _detect_deflection(self, current: _TurnRecord) -> List[Anomaly]:
        """Detect topic deflection after an evoker fires.

        Pattern: evoker fires in turn N, then turns N+1 to N+3 suddenly
        shift to mundane/neutral topics with short messages. The distance
        between the evoker topic and the deflected topic = avoidance signal.
        """
        anomalies = []
        if len(self._history) < 2:
            return anomalies

        # Look back for recent evoker firings within the deflection window
        window_start = max(0, len(self._history) - self.DEFLECTION_WINDOW - 1)
        evoker_turns = [
            h for h in self._history[window_start:-1]
            if h.has_evoker
        ]

        if not evoker_turns:
            return anomalies

        # Current message characteristics
        v = current.vadug.v
        is_neutral = self.DEFLECTION_V_NEUTRAL_LOW <= v <= self.DEFLECTION_V_NEUTRAL_HIGH
        is_short = current.word_count <= self.DEFLECTION_MIN_VELOCITY

        if not (is_neutral or is_short):
            return anomalies

        for ev_turn in evoker_turns:
            turns_since = current.turn_number - ev_turn.turn_number
            if turns_since < 1 or turns_since > self.DEFLECTION_WINDOW:
                continue

            # Check if the messages BETWEEN the evoker and now are also
            # neutral/short — a sustained deflection is stronger evidence
            idx_start = self._history.index(ev_turn) + 1
            idx_end = len(self._history) - 1  # exclude current (already checked)
            between = self._history[idx_start:idx_end]

            neutral_count = sum(
                1 for h in between
                if self.DEFLECTION_V_NEUTRAL_LOW <= h.vadug.v <= self.DEFLECTION_V_NEUTRAL_HIGH
                or h.word_count <= self.DEFLECTION_MIN_VELOCITY
            )

            # Severity scales with how many consecutive neutral/short turns
            # follow the evoker and how heavy the evoker's gravity priming was
            severity = min(5, 2 + neutral_count + (1 if is_short and is_neutral else 0))

            # Emotional distance: how far current state is from evoker state
            v_distance = abs(current.vadug.v - ev_turn.vadug.v)
            g_distance = abs(current.vadug.g - ev_turn.vadug.g)

            anomalies.append(Anomaly(
                type="DEFLECTION",
                severity=severity,
                description=(
                    f"Evoker '{', '.join(ev_turn.evoker_words)}' fired {turns_since} "
                    f"turn(s) ago, but conversation shifted to neutral/mundane "
                    f"(V={v}, {current.word_count} words). "
                    f"Emotional distance: V-delta={v_distance}, G-delta={g_distance}. "
                    f"Possible topic avoidance."
                ),
                evidence={
                    "evoker_turn": ev_turn.turn_number,
                    "evoker_words": ev_turn.evoker_words,
                    "evoker_vadug": str(ev_turn.vadug),
                    "current_vadug": str(current.vadug),
                    "turns_since_evoker": turns_since,
                    "current_word_count": current.word_count,
                    "v_distance": v_distance,
                    "g_distance": g_distance,
                },
            ))

        return anomalies

    # ------------------------------------------------------------------
    # Detector 2: Gravitational Lensing (Masking)
    # ------------------------------------------------------------------

    def _detect_masking(self, current: _TurnRecord) -> List[Anomaly]:
        """Detect masking: positive words but heavy/helpless VADUG.

        Pattern: surface reads positive (V >= 128) but the D and G
        dimensions tell a different story — low D = no agency,
        sinking G = emotional weight. Classic 'I'm fine' masking.
        """
        anomalies = []
        v = current.vadug.v
        d = current.vadug.d
        g = current.vadug.g

        if v < self.MASKING_V_FLOOR:
            return anomalies  # not reading as positive, no masking to detect

        # Check for positive-word-heavy trace with low D/G result
        positive_payloads = sum(
            1 for t in current.trace
            if t.get("role") in ("PAYLOAD", "EVOKER+PAY") and t.get("v", 128) > 140
        )

        d_masking = d < self.MASKING_D_CEILING
        g_masking = g < self.MASKING_G_CEILING

        if not (d_masking or g_masking):
            return anomalies

        # Need at least some positive surface signal
        if positive_payloads == 0 and v < 140:
            return anomalies

        # Severity based on how extreme the D/G divergence is
        severity = 1
        if d_masking and g_masking:
            severity = 3
            if d < 60 and g < 70:
                severity = 4
            if d < 40 and g < 50:
                severity = 5
        elif d_masking:
            severity = 2 if d < 60 else 1
        elif g_masking:
            severity = 2 if g < 70 else 1

        divergence_d = v - d  # positive = V says fine, D says helpless
        divergence_g = v - g  # positive = V says fine, G says heavy

        anomalies.append(Anomaly(
            type="MASKING",
            severity=severity,
            description=(
                f"Surface reads positive (V={v}) but subsurface tells a "
                f"different story: D={d} ({'no agency' if d < 80 else 'low agency'}), "
                f"G={g} ({'heavy/sinking' if g < 90 else 'slightly heavy'}). "
                f"V-D divergence={divergence_d}, V-G divergence={divergence_g}. "
                f"Possible emotional masking."
            ),
            evidence={
                "vadug": str(current.vadug),
                "v": v,
                "d": d,
                "g": g,
                "v_d_divergence": divergence_d,
                "v_g_divergence": divergence_g,
                "positive_payloads": positive_payloads,
                "text_snippet": current.text[:80],
            },
        ))

        return anomalies

    # ------------------------------------------------------------------
    # Detector 3: Velocity Anomalies (Behavioral Shifts)
    # ------------------------------------------------------------------

    def _detect_velocity(self, current: _TurnRecord) -> List[Anomaly]:
        """Detect behavioral velocity anomalies.

        Tracks baseline message length, punctuation density, and formality.
        Sudden deviations from baseline signal emotional events:
          - Short burst after long messages = crisis compression
          - Sudden formality after casual = emotional distancing
          - Punctuation density spike = agitation
        """
        anomalies = []

        if len(self._history) < self.VELOCITY_MIN_TURNS:
            return anomalies  # need baseline

        # Compute baseline from all turns EXCEPT current
        prior = self._history[:-1]
        baseline_length = sum(h.word_count for h in prior) / len(prior)
        baseline_punct = sum(h.punctuation_density for h in prior) / len(prior)
        baseline_formality = sum(h.formality_score for h in prior) / len(prior)

        # Guard against zero baseline
        if baseline_length < 1:
            baseline_length = 1

        length_ratio = current.word_count / baseline_length

        # --- Crisis compression: sudden short message ---
        if length_ratio < self.VELOCITY_LENGTH_RATIO and baseline_length > 5:
            severity = 3
            if length_ratio < 0.2:
                severity = 4
            if current.word_count <= 2:
                severity = 5

            anomalies.append(Anomaly(
                type="VELOCITY",
                severity=severity,
                description=(
                    f"Crisis compression: message is {current.word_count} words "
                    f"(baseline avg {baseline_length:.0f}). "
                    f"Ratio={length_ratio:.2f}. "
                    f"Sudden brevity after longer messages signals emotional shutdown "
                    f"or overwhelm."
                ),
                evidence={
                    "vadug": str(current.vadug),
                    "current_word_count": current.word_count,
                    "baseline_word_count": round(baseline_length, 1),
                    "length_ratio": round(length_ratio, 3),
                    "text_snippet": current.text[:80],
                },
            ))

        # --- Emotional flood: sudden long message ---
        if length_ratio > self.VELOCITY_LENGTH_SURGE and baseline_length > 3:
            severity = 2
            if length_ratio > 4.0:
                severity = 3

            anomalies.append(Anomaly(
                type="VELOCITY",
                severity=severity,
                description=(
                    f"Emotional flood: message is {current.word_count} words "
                    f"(baseline avg {baseline_length:.0f}). "
                    f"Ratio={length_ratio:.2f}. "
                    f"Sudden verbosity may indicate emotional dam breaking."
                ),
                evidence={
                    "vadug": str(current.vadug),
                    "current_word_count": current.word_count,
                    "baseline_word_count": round(baseline_length, 1),
                    "length_ratio": round(length_ratio, 3),
                    "text_snippet": current.text[:80],
                },
            ))

        # --- Punctuation density shift ---
        punct_delta = abs(current.punctuation_density - baseline_punct)
        if punct_delta > self.VELOCITY_PUNCT_SHIFT:
            severity = 2
            direction = "spike" if current.punctuation_density > baseline_punct else "drop"

            anomalies.append(Anomaly(
                type="VELOCITY",
                severity=severity,
                description=(
                    f"Punctuation {direction}: density={current.punctuation_density:.3f} "
                    f"(baseline {baseline_punct:.3f}, delta={punct_delta:.3f}). "
                    f"{'Agitation or emphasis' if direction == 'spike' else 'Flattened affect'}."
                ),
                evidence={
                    "vadug": str(current.vadug),
                    "current_punct_density": round(current.punctuation_density, 4),
                    "baseline_punct_density": round(baseline_punct, 4),
                    "punct_delta": round(punct_delta, 4),
                },
            ))

        # --- Formality shift (emotional distancing) ---
        formality_delta = current.formality_score - baseline_formality
        if abs(formality_delta) > self.VELOCITY_FORMALITY_SHIFT:
            severity = 2
            if abs(formality_delta) > 0.6:
                severity = 3

            if formality_delta > 0:
                desc = (
                    f"Emotional distancing: formality jumped from "
                    f"{baseline_formality:.2f} to {current.formality_score:.2f} "
                    f"(delta={formality_delta:+.2f}). "
                    f"Sudden formality after casual speech = putting up walls."
                )
            else:
                desc = (
                    f"Formality collapse: dropped from "
                    f"{baseline_formality:.2f} to {current.formality_score:.2f} "
                    f"(delta={formality_delta:+.2f}). "
                    f"Sudden casualness may indicate emotional guard dropping."
                )

            anomalies.append(Anomaly(
                type="VELOCITY",
                severity=severity,
                description=desc,
                evidence={
                    "vadug": str(current.vadug),
                    "current_formality": round(current.formality_score, 3),
                    "baseline_formality": round(baseline_formality, 3),
                    "formality_delta": round(formality_delta, 3),
                },
            ))

        return anomalies

    # ------------------------------------------------------------------
    # Detector 4: Resonance Patterns (Loops)
    # ------------------------------------------------------------------

    def _detect_resonance(self, current: _TurnRecord) -> List[Anomaly]:
        """Detect emotional loops and oscillation.

        Patterns:
          - Same 5D emotional signature repeating = stuck in a loop
          - V oscillating between two values = indecision/conflict pendulum
        """
        anomalies = []

        if len(self._history) < self.RESONANCE_WINDOW:
            return anomalies

        recent = self._history[-self.RESONANCE_WINDOW:]

        # --- Same-signature loop: all recent turns within tolerance ---
        center_v = sum(h.vadug.v for h in recent) / len(recent)
        center_a = sum(h.vadug.a for h in recent) / len(recent)
        center_d = sum(h.vadug.d for h in recent) / len(recent)
        center_u = sum(h.vadug.u for h in recent) / len(recent)
        center_g = sum(h.vadug.g for h in recent) / len(recent)

        all_within = all(
            abs(h.vadug.v - center_v) < self.RESONANCE_FULL_TOLERANCE
            and abs(h.vadug.a - center_a) < self.RESONANCE_FULL_TOLERANCE
            and abs(h.vadug.d - center_d) < self.RESONANCE_FULL_TOLERANCE
            and abs(h.vadug.u - center_u) < self.RESONANCE_FULL_TOLERANCE
            and abs(h.vadug.g - center_g) < self.RESONANCE_FULL_TOLERANCE
            for h in recent
        )

        if all_within:
            severity = 3
            if len(self._history) >= self.RESONANCE_WINDOW + 2:
                # Longer loop = higher severity
                extended = self._history[-(self.RESONANCE_WINDOW + 2):]
                still_loop = all(
                    abs(h.vadug.v - center_v) < self.RESONANCE_FULL_TOLERANCE
                    and abs(h.vadug.g - center_g) < self.RESONANCE_FULL_TOLERANCE
                    for h in extended
                )
                if still_loop:
                    severity = 4

            anomalies.append(Anomaly(
                type="RESONANCE",
                severity=severity,
                description=(
                    f"Emotional loop: last {self.RESONANCE_WINDOW} turns have "
                    f"nearly identical VADUG signatures (center: V={center_v:.0f} "
                    f"A={center_a:.0f} D={center_d:.0f} U={center_u:.0f} "
                    f"G={center_g:.0f}). Person may be stuck in an emotional rut."
                ),
                evidence={
                    "vadug": str(current.vadug),
                    "loop_center": {
                        "v": round(center_v, 1),
                        "a": round(center_a, 1),
                        "d": round(center_d, 1),
                        "u": round(center_u, 1),
                        "g": round(center_g, 1),
                    },
                    "window_size": self.RESONANCE_WINDOW,
                    "vadug_history": [str(h.vadug) for h in recent],
                },
            ))

        # --- V oscillation: alternating high/low pattern ---
        if len(recent) >= 4:
            v_values = [h.vadug.v for h in recent]
            # Check for alternating pattern: high-low-high-low or low-high-low-high
            alternating = True
            for i in range(2, len(v_values)):
                # Each value should be closer to the value 2 steps back
                # than to the value 1 step back
                same_track = abs(v_values[i] - v_values[i - 2])
                diff_track = abs(v_values[i] - v_values[i - 1])
                if same_track > self.RESONANCE_OSCILLATION_BAND or diff_track < self.RESONANCE_OSCILLATION_BAND:
                    alternating = False
                    break

            if alternating and not all_within:
                v_high = max(v_values)
                v_low = min(v_values)
                swing = v_high - v_low

                severity = 2
                if swing > 40:
                    severity = 3
                if swing > 60:
                    severity = 4

                anomalies.append(Anomaly(
                    type="RESONANCE",
                    severity=severity,
                    description=(
                        f"Emotional pendulum: V oscillating between ~{v_low} and "
                        f"~{v_high} (swing={swing}) over {len(v_values)} turns. "
                        f"Alternating pattern suggests indecision or unresolved conflict."
                    ),
                    evidence={
                        "vadug": str(current.vadug),
                        "v_values": v_values,
                        "v_low": v_low,
                        "v_high": v_high,
                        "swing": swing,
                    },
                ))

        return anomalies
