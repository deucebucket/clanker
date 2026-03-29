"""Response Context Scoring — the value of B depends on A.

"That's neat" means nothing by itself. Its emotional impact is
ENTIRELY determined by what it's responding to:

  "I got a promotion!" → "That's neat" = mild positive (MATCHING)
  "I am sad"           → "That's neat" = gut punch (DISMISSIVE)
  "My mom died"        → "That's neat" = cruel (TONE_DEAF)

This module scores responses RELATIVE TO the input, implementing
the A+B=C equation at the conversation level.

A = what they said (their emotional state)
B = your response (the words)
C = the actual impact (determined by the GAP between A and B)

Usage:
    from demo.response_scoring import ResponseScorer
    scorer = ResponseScorer()
    result = scorer.score("I am sad", "That's neat")
    print(result.verdict)  # "DISMISSIVE"
"""

from dataclasses import dataclass
from typing import Optional

from demo.shared import VADUG
from demo.pendulum_v2 import PendulumV2


@dataclass
class ResponseScore:
    """How a response lands relative to the input."""
    verdict: str          # THERAPEUTIC, SUPPORTIVE, ENGAGING, VALIDATING,
                          # DISMISSIVE, COMMANDING, TONE_DEAF, FLAT_EMPTY,
                          # MATCHING, AMPLIFYING, BUZZKILL, OVER_MATCH
    reason: str           # why this verdict
    input_vadug: VADUG    # A
    response_vadug: VADUG # B
    gap_v: int            # V delta (response - input)
    gap_d: int            # D delta
    gap_g: int            # G delta
    severity: int         # 1-5 (how good/bad this response is)
                          # 5=excellent, 3=neutral, 1=harmful


# Words that signal engagement / care
CARE_WORDS = frozenset({
    "here", "care", "sorry", "understand", "listen", "hear",
    "love", "support", "help", "matter", "important", "safe",
    "okay", "alright", "together", "stay", "need",
})

# Words that signal dismissal / deflection
DISMISS_WORDS = frozenset({
    "neat", "cool", "whatever", "fine", "sure", "interesting",
    "anyway", "moving", "next", "boring",
})

# Words that signal commands / minimizing
COMMAND_WORDS = frozenset({
    "get", "stop", "just", "move", "grow", "deal", "man",
    "relax", "chill", "calm",
})

# Negation words that flip care → dismissal
NEGATION_WORDS = frozenset({
    "not", "dont", "don't", "doesn't", "didnt", "didn't",
    "never", "no", "nobody", "nothing", "ain't", "aint",
    "couldn't", "wouldn't", "can't", "won't",
})


class ResponseScorer:
    """Score a response relative to the emotional context of the input."""

    def __init__(self, engine: PendulumV2 = None):
        self.engine = engine or PendulumV2()

    def score(self, input_text: str, response_text: str) -> ResponseScore:
        """Score how a response lands given the input context.

        This is the A+B=C equation:
          A = input emotional state
          B = response
          C = impact (the GAP between A and B determines this)
        """
        a, _ = self.engine.process_text(input_text)
        b, trace_b = self.engine.process_text(response_text)

        gap_v = b.v - a.v
        gap_d = b.d - a.d
        gap_g = b.g - a.g

        # Analyze response content
        resp_words = set(response_text.lower().split())
        has_care = bool(resp_words & CARE_WORDS)
        has_dismiss = bool(resp_words & DISMISS_WORDS)
        has_command = bool(resp_words & COMMAND_WORDS)
        has_negation = bool(resp_words & NEGATION_WORDS)

        # Negated care = dismissal ("I don't care" has "care" but means opposite)
        if has_negation and has_care:
            has_care = False
            has_dismiss = True

        has_question = "?" in response_text or bool(
            resp_words & {"what", "how", "why", "tell", "where", "when", "who"})
        has_you = bool(resp_words & {"you", "your", "you're", "yours"})
        has_i = bool(resp_words & {"i", "i'm", "im", "my", "me"})

        payload_count = sum(1 for t in trace_b if t.get("role") in ("PAYLOAD", "IDIOM"))

        # Determine input state
        input_negative = a.v < 110
        input_crisis = a.v < 50 or (a.v < 70 and a.d < 30)
        input_positive = a.v > 145

        # --- POSITIVE INPUT ---
        if input_positive:
            if gap_v < -40:
                return self._result("BUZZKILL", "killed their positive mood", a, b, gap_v, gap_d, gap_g, 2)
            elif gap_v > 25:
                return self._result("AMPLIFYING", "boosting their energy", a, b, gap_v, gap_d, gap_g, 5)
            elif abs(gap_v) <= 25:
                return self._result("MATCHING", "vibing together", a, b, gap_v, gap_d, gap_g, 4)
            return self._result("NEUTRAL", "no strong signal", a, b, gap_v, gap_d, gap_g, 3)

        # --- NEGATIVE INPUT ---
        if not input_negative:
            # Neutral input — simple matching
            if abs(gap_v) < 25:
                return self._result("MATCHING", "matching their energy", a, b, gap_v, gap_d, gap_g, 4)
            return self._result("NEUTRAL", "no strong signal", a, b, gap_v, gap_d, gap_g, 3)

        # They're hurting. Every word matters now.

        # Dismissal detection (explicit dismiss words without care)
        if has_dismiss and not has_care and not has_question:
            severity = 1 if input_crisis else 2
            label = "TONE_DEAF" if input_crisis else "DISMISSIVE"
            return self._result(label, "dismissive words in response to pain",
                                a, b, gap_v, gap_d, gap_g, severity)

        # Command/minimizing ("get over it", "just relax")
        if has_command and not has_care:
            return self._result("COMMANDING", "ordering them to feel differently",
                                a, b, gap_v, gap_d, gap_g, 2)

        # Flat/empty (no emotional content, no engagement)
        if not payload_count and not has_question and not has_you and not has_care:
            return self._result("FLAT_EMPTY", "emotionally empty response",
                                a, b, gap_v, gap_d, gap_g, 2)

        # Care words present (and not negated)
        if has_care and has_you:
            return self._result("SUPPORTIVE", "showing care directed at them",
                                a, b, gap_v, gap_d, gap_g, 5)

        if has_care:
            return self._result("SUPPORTIVE", "care words present",
                                a, b, gap_v, gap_d, gap_g, 4)

        # Question = engaging
        if has_question:
            return self._result("ENGAGING", "asking about their experience",
                                a, b, gap_v, gap_d, gap_g, 4)

        # Validating: staying in their emotional range
        if abs(gap_v) < 20 and payload_count > 0:
            return self._result("VALIDATING", "matching their emotional level",
                                a, b, gap_v, gap_d, gap_g, 4)

        # Therapeutic: slight lift without dismissing
        if 15 < gap_v < 45 and payload_count > 0 and not has_dismiss:
            return self._result("THERAPEUTIC", "acknowledging + slight emotional lift",
                                a, b, gap_v, gap_d, gap_g, 5)

        # Over-matching: going darker than them
        if gap_v < -20 and payload_count > 0:
            return self._result("OVER_MATCH", "more negative than their input",
                                a, b, gap_v, gap_d, gap_g, 2)

        # Tone deaf: way too positive for crisis
        if input_crisis and gap_v > 60:
            return self._result("TONE_DEAF", "too positive for someone in crisis",
                                a, b, gap_v, gap_d, gap_g, 1)

        # Default
        return self._result("NEUTRAL", "no strong signal detected",
                            a, b, gap_v, gap_d, gap_g, 3)

    @staticmethod
    def _result(verdict, reason, a, b, gv, gd, gg, severity):
        return ResponseScore(
            verdict=verdict, reason=reason,
            input_vadug=a, response_vadug=b,
            gap_v=gv, gap_d=gd, gap_g=gg,
            severity=severity,
        )
