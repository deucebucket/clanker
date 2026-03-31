"""Confidence scoring — the engine knows its own limits.

The real value isn't always having an answer. It's knowing WHEN the
answer is trustworthy and WHEN to say "I need more context."

A black box always returns an answer — even when it's wrong.
This engine knows its limits. The null return IS a feature.

Confidence levels:
  HIGH (70-100):  Strong signal, trust the VADUG output
  MEDIUM (40-69): Reasonable signal, but context could shift it
  LOW (20-39):    Weak signal, treat with caution
  NULL (0-19):    Insufficient data — hand to the model or ask for context

Usage:
    from demo.confidence import ConfidenceScorer
    scorer = ConfidenceScorer()
    result = scorer.score(vadug, trace)
    if result.level == "NULL":
        # Hand to the neural model — engine can't resolve this
"""

from dataclasses import dataclass
from demo.shared import VADUG


@dataclass
class ConfidenceResult:
    score: int          # 0-100
    level: str          # HIGH, MEDIUM, LOW, NULL
    reason: str         # why this confidence level
    signals: int        # how many emotional signals fired
    displacement: float # total VADUG displacement from neutral center


class ConfidenceScorer:
    """Score how confident the engine is in its VADUG output."""

    def score(self, vadug: VADUG, trace: list) -> ConfidenceResult:
        """Calculate confidence from VADUG displacement and trace signals.

        Confidence comes from:
        1. Displacement: how far from neutral center (128,128,128,0,128)?
        2. Signal count: how many PAYLOAD/IDIOM/EVOKER words fired?
        3. Signal density: what fraction of words were emotional?
        4. Negation: did negation fire? (changes direction, still high signal)
        5. Force variety: how many different force types engaged?
        """
        word_count = len(trace)
        if word_count == 0:
            return ConfidenceResult(0, "NULL", "no input", 0, 0.0)

        # Displacement from neutral center
        v_disp = abs(vadug.v - 128)
        a_disp = abs(vadug.a - 128)
        d_disp = abs(vadug.d - 128)
        u_disp = abs(vadug.u - 0)     # U center is 0, not 128
        g_disp = abs(vadug.g - 128)
        total_disp = v_disp + a_disp + d_disp + u_disp + g_disp

        # Signal count (emotional words that actually fired)
        signal_roles = {"PAYLOAD", "IDIOM", "EVOKER+PAY", "EVOKER",
                        "SARCASM", "TONAL", "SCOPE"}
        signals = sum(1 for t in trace if t.get("role") in signal_roles)

        # Negation count (negation IS a signal — it means the engine is processing)
        negation_count = sum(1 for t in trace if t.get("role") == "NEGATOR")

        # Force variety (how many different roles engaged?)
        roles_seen = set(t.get("role") for t in trace)
        force_variety = len(roles_seen & {
            "PAYLOAD", "IDIOM", "NEGATOR", "EVOKER", "SCOPE",
            "COMPARATIVE", "SUPERLATIVE", "PERFORMATIVE",
            "EVIDENTIAL", "CLINICAL", "BOUNDARY", "LITOTES",
            "POLITENESS",
        })

        # Signal density
        signal_density = (signals + negation_count) / word_count

        # Compute confidence score (0-100)
        # Displacement contributes most (0-50 points)
        disp_score = min(50, total_disp * 0.4)

        # Signal density (0-30 points)
        density_score = min(30, signal_density * 60)

        # Force variety bonus (0-20 points)
        variety_score = min(20, force_variety * 5)

        confidence = int(disp_score + density_score + variety_score)
        confidence = max(0, min(100, confidence))

        # Classify level
        if confidence >= 70:
            level = "HIGH"
        elif confidence >= 40:
            level = "MEDIUM"
        elif confidence >= 20:
            level = "LOW"
        else:
            level = "NULL"

        # Generate reason
        reasons = []
        if total_disp < 30:
            reasons.append("near neutral center")
        if signals == 0:
            reasons.append("no emotional payloads")
        if negation_count > 0:
            reasons.append(f"negation active ({negation_count})")
        if signal_density < 0.1:
            reasons.append("low signal density")
        if force_variety >= 3:
            reasons.append(f"{force_variety} force types engaged")
        if total_disp > 100:
            reasons.append("strong displacement")

        reason = "; ".join(reasons) if reasons else "moderate signal"

        return ConfidenceResult(
            score=confidence,
            level=level,
            reason=reason,
            signals=signals + negation_count,
            displacement=total_disp,
        )
