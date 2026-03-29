"""Entropy-based neutral detection for the Clanker pipeline.

Uses Shannon entropy of the V trajectory to detect neutral content.
High entropy = random/uniform pushes = no clear emotional signal = NEUTRAL.
Low entropy = concentrated directional spikes = clear emotion = EMOTIONAL.

Works WITH the AdaptiveClassifier, not replacing it.
When entropy is high (uncertain trajectory), override to neutral.
"""

import math


class EntropyClassifier:
    """Use trajectory entropy to detect neutral content.

    Works WITH the AdaptiveClassifier, not replacing it.
    When entropy is high (uncertain trajectory), override to neutral.
    """

    def compute_trajectory_entropy(self, pendulum_history):
        """Compute Shannon entropy of the V trajectory.

        Bins the V deltas into categories and computes entropy.
        High entropy = random/uniform distribution = no clear signal.
        Low entropy = concentrated in one direction = clear emotion.
        """
        if not pendulum_history or len(pendulum_history) < 3:
            return 0.5  # not enough data, moderate entropy

        # Extract V values from history
        v_values = []
        for step in pendulum_history:
            v = step.get('v', 128)
            v_values.append(v)

        # Compute deltas between consecutive V values
        deltas = [v_values[i+1] - v_values[i] for i in range(len(v_values)-1)]

        if not deltas:
            return 1.0  # no movement = maximum entropy = neutral

        # Bin deltas: strong_neg, mild_neg, near_zero, mild_pos, strong_pos
        bins = [0] * 5
        for d in deltas:
            if d < -10:
                bins[0] += 1      # strong negative push
            elif d < -3:
                bins[1] += 1      # mild negative
            elif d <= 3:
                bins[2] += 1      # near zero (no real push)
            elif d <= 10:
                bins[3] += 1      # mild positive
            else:
                bins[4] += 1      # strong positive push

        total = sum(bins)
        if total == 0:
            return 1.0

        # Shannon entropy
        entropy = 0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize to 0-1 (max entropy for 5 bins = log2(5) ~ 2.32)
        max_entropy = math.log2(5)
        return entropy / max_entropy

    def compute_trajectory_variance(self, pendulum_history):
        """Compute variance of V trajectory -- how much did V actually move?"""
        if not pendulum_history or len(pendulum_history) < 2:
            return 0

        v_values = [step.get('v', 128) for step in pendulum_history]
        mean_v = sum(v_values) / len(v_values)
        variance = sum((v - mean_v) ** 2 for v in v_values) / len(v_values)
        return variance

    def compute_spike_count(self, pendulum_history):
        """Count how many significant V spikes occurred."""
        if not pendulum_history:
            return 0
        v_values = [step.get('v', 128) for step in pendulum_history]
        spikes = 0
        for i in range(1, len(v_values)):
            if abs(v_values[i] - v_values[i-1]) > 10:
                spikes += 1
        return spikes

    def should_override_neutral(self, pendulum_history, current_v, intent_mode):
        """Determine if the trajectory suggests this is actually neutral content.

        Returns True if entropy analysis suggests neutral, False to trust V.
        """
        entropy = self.compute_trajectory_entropy(pendulum_history)
        variance = self.compute_trajectory_variance(pendulum_history)
        spikes = self.compute_spike_count(pendulum_history)

        # Near center + high entropy + low spikes = NEUTRAL
        # Tuned via sweep on 3,372 sentences: entropy>0.8, var<30, margin=10, spikes<4
        v_near_center = abs(current_v - 128) < 10  # was 20 — tighter
        high_entropy = entropy > 0.8  # was 0.7 — stricter
        low_spikes = spikes < 4  # was 2 — more permissive

        # For STATEMENT/QUESTION intent, be more willing to call neutral
        if intent_mode in ('STATEMENT', 'QUESTION', 'REQUEST', 'GREETING'):
            if v_near_center and high_entropy and low_spikes:
                return True
            if variance < 30 and v_near_center:
                return True

        # For any intent, if trajectory is COMPLETELY flat, it's neutral
        if variance < 10 and abs(current_v - 128) < 10:
            return True

        return False
