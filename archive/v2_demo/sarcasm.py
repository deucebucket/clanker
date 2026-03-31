"""Sarcasm detection for the Clanker pipeline."""

from .shared import VADUG

class SarcasmDetector:
    """Detects sarcasm from pendulum trajectory patterns.

    Three signals:
    1. Trajectory Reversal: positive spike -> immediate drop
    2. Intensity Mismatch: strong positive word in negative context
    3. Context Contradiction: chunk grade contradicts recent emotional history

    Pure math from the pendulum trajectory. No sentiment classifier. No training data.
    """

    # Confidence levels
    NONE = 0
    LOW = 1       # one signal detected
    MODERATE = 2  # two signals detected
    HIGH = 3      # all three signals or very strong single signal

    def analyze_trajectory(self, history):
        """Check pendulum history for trajectory reversal and intensity mismatch.

        Args:
            history: list of trace dicts with 'word', 'v', 'a', 'd', 'u', 'g'

        Returns:
            (detected: bool, confidence: int, signals: list[str])
        """
        signals = []

        # Signal 1: Trajectory Reversal
        # A positive word causes a V spike, but within 2-3 words the trajectory
        # drops significantly. The positive was fake — the context was negative.
        for i in range(1, len(history)):
            spike = history[i]['v'] - history[i-1]['v']
            if spike > 25:  # positive spike
                # Check next 3 words for drop
                for j in range(i+1, min(i+4, len(history))):
                    drop = history[i]['v'] - history[j]['v']
                    if drop > 20:
                        signals.append(
                            f"REVERSAL: '{history[i]['word']}' spiked V+{spike} "
                            f"then dropped V-{drop} by '{history[j]['word']}'"
                        )
                        break

        # Signal 2: Intensity Mismatch
        # A very strong positive word appears in a context where the overall
        # sentiment is negative or neutral. The word is TOO positive.
        for i in range(len(history)):
            if i > 0:
                spike = history[i]['v'] - history[i-1]['v']
                if spike > 35:  # very strong positive word
                    # Check surrounding context (3 before, 3 after)
                    start = max(0, i-3)
                    end = min(len(history), i+4)
                    surrounding = [h['v'] for h in history[start:end] if h != history[i]]
                    if surrounding:
                        avg_surrounding = sum(surrounding) / len(surrounding)
                        if avg_surrounding < 115:
                            signals.append(
                                f"MISMATCH: '{history[i]['word']}' too positive (V+{spike}) "
                                f"for context (avg V={avg_surrounding:.0f})"
                            )

        # Determine confidence
        if len(signals) >= 3:
            confidence = SarcasmDetector.HIGH
        elif len(signals) == 2:
            confidence = SarcasmDetector.MODERATE
        elif len(signals) == 1:
            confidence = SarcasmDetector.LOW
        else:
            confidence = SarcasmDetector.NONE

        return len(signals) > 0, confidence, signals

    def analyze_context(self, previous_chunks, current_chunk):
        """Check for context contradiction sarcasm (Signal 3).

        Previous negative context + current positive with LOW arousal = sarcasm.
        Low arousal is key: genuine positive after negative is HIGH arousal
        (relief/excitement). Flat delivery of positive words after negative
        context = sarcasm or passive aggression.

        Args:
            previous_chunks: list of previous chunk results with 'vadug' key
            current_chunk: current chunk result with 'vadug' key

        Returns:
            (detected: bool, details: str)
        """
        if not previous_chunks:
            return False, ""

        prev_avg_v = sum(c['vadug'].v for c in previous_chunks) / len(previous_chunks)
        curr_v = current_chunk['vadug'].v
        curr_a = current_chunk['vadug'].a

        # Previous negative, current positive, low arousal = sarcasm
        if prev_avg_v < 90 and curr_v > 135 and curr_a < 145:
            return True, (
                f"CONTRADICTION: previous context avg V={prev_avg_v:.0f} (negative), "
                f"current V={curr_v} with low A={curr_a} "
                f"(flat delivery of positive after negative = likely sarcastic)"
            )

        return False, ""

    def adjust_grade(self, grade, confidence, grader):
        """Adjust the sentence grade downward when sarcasm is detected.

        If words say B but sarcasm detected, the real grade is probably C- or D.
        The surface reads positive but the meaning is negative.

        Args:
            grade: original grade string (e.g. "B")
            confidence: sarcasm confidence level (LOW/MODERATE/HIGH)
            grader: SentenceGrader instance for bump operations

        Returns:
            (adjusted_grade: str, adjustment_note: str)
        """
        if confidence == SarcasmDetector.NONE:
            return grade, ""

        original = grade
        if confidence == SarcasmDetector.HIGH:
            # Drop 3-4 half-steps
            for _ in range(4):
                grade = grader._bump_down(grade)
        elif confidence == SarcasmDetector.MODERATE:
            # Drop 2-3 half-steps
            for _ in range(3):
                grade = grader._bump_down(grade)
        elif confidence == SarcasmDetector.LOW:
            # Drop 1 half-step
            grade = grader._bump_down(grade)

        if original != grade:
            note = f"Grade adjusted: {original} -> {grade} (surface positive, meaning negative)"
        else:
            note = ""
        return grade, note

    def get_label(self, confidence):
        """Get human-readable sarcasm label."""
        if confidence == self.HIGH:
            return "SARCASM DETECTED (high confidence)"
        elif confidence == self.MODERATE:
            return "Possible sarcasm (moderate confidence)"
        elif confidence == self.LOW:
            return "Hint of sarcasm (low confidence)"
        return "No sarcasm detected"

    def display(self, detected, confidence, signals, context_signal=None,
                grade_note="", verbose=True):
        """Print the sarcasm analysis report."""
        if not verbose:
            return
        if not detected and not context_signal:
            return

        print(f"\n--- SARCASM ANALYSIS ---")
        for i, signal in enumerate(signals):
            print(f"  Signal {i+1}: {signal}")
        if context_signal:
            print(f"  Signal {len(signals)+1}: {context_signal}")
        print(f"")
        print(f"  Verdict: {self.get_label(confidence)}")
        if grade_note:
            print(f"  {grade_note}")
        if confidence >= self.MODERATE:
            print(f"  Response mode: address underlying frustration, not surface positivity")

