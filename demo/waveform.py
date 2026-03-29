"""Emotional Waveform Analysis — the SHAPE of emotion in a sentence.

Every sentence has a trajectory through VADUG space. Two sentences can
end at the same final V but follow completely different paths:

  "I was happy but now I am sad"   → DESCENDING wave (163 → 116)
  "I was sad but now I am happy"   → ASCENDING wave  (93 → 140)

The SHAPE carries information the final number doesn't:
  - Jokes: UP → sharp DROP (the punchline is the inversion)
  - Grief: slow DESCENT (sinking, no recovery)
  - Sarcasm: UP → immediate DROP (fake positive → crash)
  - Rage: sharp SPIKE then sustain (explosion → burn)
  - Run-on: gradual BUILD with no resolution (overwhelm)

Usage:
    from demo.waveform import WaveformAnalyzer
    analyzer = WaveformAnalyzer()
    wave = analyzer.analyze(trace)
    print(wave.shape)      # "DESCENDING"
    print(wave.amplitude)  # 47
    print(wave.signature)  # "grief" or "sarcasm_kickflip" or "joke_punchline"
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Waveform:
    """The emotional shape of a sentence."""
    # Raw trajectory
    v_points: List[float]       # V values at each payload word
    a_points: List[float]       # A values
    d_points: List[float]       # D values
    g_points: List[float]       # G values

    # Shape metrics
    shape: str                  # ASCENDING, DESCENDING, SPIKE, CRASH, OSCILLATING, FLAT
    amplitude: float            # peak-to-trough range
    start_v: float              # first payload V
    end_v: float                # last payload V
    peak_v: float               # highest V
    trough_v: float             # lowest V
    delta_v: float              # end - start (direction)

    # Waveform signature
    signature: str              # grief, rage, joke, sarcasm, overwhelm, neutral
    confidence: float           # 0.0-1.0

    # Timing
    word_count: int             # total words in sentence
    payload_count: int          # number of payload words (the wave points)
    density: float              # payload_count / word_count (emotional density)


class WaveformAnalyzer:
    """Analyze the emotional trajectory shape of processed text."""

    def analyze(self, trace: list) -> Waveform:
        """Extract waveform from a pendulum trace.

        Args:
            trace: list of trace dicts from PendulumV2.process_text()

        Returns:
            Waveform with shape classification and signature
        """
        # Extract payload points (the emotional content — not operators/neutral)
        payload_roles = {'PAYLOAD', 'IDIOM', 'EVOKER+PAY', 'SARCASM', 'TONAL'}
        payloads = [t for t in trace if t.get('role') in payload_roles]

        v_points = [t['v'] for t in payloads]
        a_points = [t['a'] for t in payloads]
        d_points = [t['d'] for t in payloads]
        g_points = [t['g'] for t in payloads]

        word_count = len(trace)
        payload_count = len(payloads)

        if payload_count < 1:
            return Waveform(
                v_points=[], a_points=[], d_points=[], g_points=[],
                shape="FLAT", amplitude=0, start_v=128, end_v=128,
                peak_v=128, trough_v=128, delta_v=0,
                signature="neutral", confidence=0.0,
                word_count=word_count, payload_count=0, density=0,
            )

        start_v = v_points[0]
        end_v = v_points[-1]
        peak_v = max(v_points)
        trough_v = min(v_points)
        amplitude = peak_v - trough_v
        delta_v = end_v - start_v
        density = payload_count / word_count if word_count > 0 else 0

        # Classify the wave shape
        shape = self._classify_shape(v_points, amplitude, delta_v)

        # Identify the waveform signature
        signature, confidence = self._identify_signature(
            v_points, a_points, d_points, g_points,
            shape, amplitude, delta_v, density
        )

        return Waveform(
            v_points=v_points, a_points=a_points,
            d_points=d_points, g_points=g_points,
            shape=shape, amplitude=amplitude,
            start_v=start_v, end_v=end_v,
            peak_v=peak_v, trough_v=trough_v,
            delta_v=delta_v,
            signature=signature, confidence=confidence,
            word_count=word_count, payload_count=payload_count,
            density=density,
        )

    def _classify_shape(self, v_points, amplitude, delta_v):
        """Classify the raw shape of the V trajectory."""
        if len(v_points) < 2:
            return "FLAT"

        if amplitude < 15:
            return "FLAT"

        # Check for spike-crash (sarcasm/joke pattern)
        # Peak is near the beginning, crash at end
        peak_idx = v_points.index(max(v_points))
        trough_idx = v_points.index(min(v_points))

        if peak_idx < len(v_points) * 0.4 and trough_idx > len(v_points) * 0.6:
            if amplitude > 30:
                return "SPIKE_CRASH"  # up then down hard

        if trough_idx < len(v_points) * 0.4 and peak_idx > len(v_points) * 0.6:
            if amplitude > 30:
                return "RECOVERY"  # down then up

        # Check for oscillation
        if len(v_points) >= 3:
            direction_changes = 0
            for i in range(2, len(v_points)):
                prev_dir = v_points[i-1] - v_points[i-2]
                curr_dir = v_points[i] - v_points[i-1]
                if (prev_dir > 5 and curr_dir < -5) or (prev_dir < -5 and curr_dir > 5):
                    direction_changes += 1
            if direction_changes >= 2:
                return "OSCILLATING"

        # Simple ascending/descending
        if delta_v > 20:
            return "ASCENDING"
        elif delta_v < -20:
            return "DESCENDING"

        return "FLAT"

    def _identify_signature(self, v_pts, a_pts, d_pts, g_pts,
                            shape, amplitude, delta_v, density):
        """Identify the emotional signature (genre) of the waveform."""

        if not v_pts:
            return "neutral", 0.0

        avg_v = sum(v_pts) / len(v_pts)
        avg_a = sum(a_pts) / len(a_pts) if a_pts else 128
        avg_d = sum(d_pts) / len(d_pts) if d_pts else 128
        avg_g = sum(g_pts) / len(g_pts) if g_pts else 128

        # Sarcasm kickflip: positive spike then crash
        if shape == "SPIKE_CRASH" and max(v_pts) > 150 and min(v_pts) < 120:
            return "sarcasm_kickflip", 0.8

        # Joke punchline: buildup then sharp inversion (either direction)
        if shape == "SPIKE_CRASH" and amplitude > 50:
            return "joke_punchline", 0.6

        # Grief: slow descent, low A (calm suffering), heavy G
        if shape == "DESCENDING" and avg_a < 140 and avg_g < 110:
            return "grief", 0.7

        # Rage: high A, high D, negative V, high density
        if avg_a > 170 and avg_d > 150 and avg_v < 80:
            return "rage", 0.8

        # Overwhelm/run-on: low density, descending, low D
        if density < 0.3 and avg_d < 100 and shape in ("DESCENDING", "FLAT"):
            return "overwhelm", 0.6

        # Crisis: descending, low everything
        if shape == "DESCENDING" and avg_v < 60 and avg_d < 60 and avg_g < 90:
            return "crisis_descent", 0.9

        # Recovery: ascending from negative
        if shape == "RECOVERY" and v_pts[0] < 100 and v_pts[-1] > 130:
            return "recovery", 0.7

        # Joy: ascending or flat-high, high V
        if avg_v > 160 and avg_a > 130:
            return "joy", 0.7

        # Calm positive
        if avg_v > 140 and avg_a < 140:
            return "contentment", 0.5

        # Anxious: high A, moderate V, low D
        if avg_a > 160 and avg_d < 100 and 80 < avg_v < 140:
            return "anxiety", 0.6

        # Default
        if avg_v > 140:
            return "positive", 0.4
        elif avg_v < 110:
            return "negative", 0.4
        return "neutral", 0.3
