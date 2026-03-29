"""Emotional Dark Matter — the chaos variable that makes each entity unique.

Not random noise. A PERSISTENT bias shaped by accumulated experience.
Two entities with identical VADUG can respond differently because their
dark matter was shaped by different histories.

The 6th dimension: V, A, D, U, G... and X.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class DarkMatter:
    """Persistent emotional chaos variable.
    
    seed: unique per entity (like DNA — you're born with a baseline)
    range: emotional volatility (0=robot, 15=human, 50=unstable)
    drift: accumulated trauma/joy (shifts over time from experience)
    resilience: how fast drift recovers toward zero (0.99=slow, 0.999=fast)
    """
    seed: int = 42
    range: int = 15
    drift: float = 0.0
    resilience: float = 0.999
    _rng: random.Random = field(default=None, repr=False)
    
    # Trauma accumulation — events that shift the baseline
    _trauma_log: list = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        self._rng = random.Random(self.seed)
        # Base bias from seed — consistent per entity
        self.base_bias = (self._rng.random() - 0.5) * self.range
    
    def apply(self, v, a=128, d=128, u=0, g=128):
        """Apply dark matter to VADUG. Returns modified values.
        
        The modification is:
        - Persistent (same entity always biases the same direction)
        - Drifting (trauma/joy shifts the bias over time)
        - Slightly noisy (not perfectly predictable — human jitter)
        """
        # Core bias = base (from birth/seed) + drift (from experience)
        bias = self.base_bias + self.drift
        
        # Tiny jitter — not random, but chaotic (deterministic from state)
        jitter = self._rng.gauss(0, self.range * 0.05)
        
        # Apply to V primarily (emotional valence is most affected by dark matter)
        v_mod = v + bias + jitter
        
        # G (gravity) also affected — trauma makes everything heavier
        g_mod = g + (bias * 0.5)  # half the V effect
        
        # A (arousal) — traumatized entities have higher baseline arousal
        a_mod = a + abs(self.drift) * 0.3  # trauma increases vigilance
        
        # Clamp
        v_mod = max(0, min(255, int(v_mod)))
        a_mod = max(0, min(255, int(a_mod)))
        g_mod = max(0, min(255, int(g_mod)))
        
        return v_mod, a_mod, d, u, g_mod
    
    def experience(self, v, g):
        """Process an emotional experience — shifts the dark matter.
        
        Negative experiences (low V, low G) make dark matter heavier.
        Positive experiences lighten it. But trauma ACCUMULATES faster
        than joy heals — negativity bias in the dark matter itself.
        """
        # How far from neutral?
        v_impact = (v - 128) / 255.0  # -0.5 to +0.5
        g_impact = (g - 128) / 255.0
        
        # Combined emotional impact
        impact = v_impact * 0.7 + g_impact * 0.3
        
        # Negativity bias: negative experiences stick 2x harder
        if impact < 0:
            impact *= 2.0
        
        # Accumulate
        self.drift += impact * 0.5
        
        # Natural recovery (resilience) — drift toward zero over time
        self.drift *= self.resilience
        
        # Log significant events
        if abs(impact) > 0.2:
            self._trauma_log.append({
                "impact": round(impact, 3),
                "drift_after": round(self.drift, 3),
                "v": v, "g": g,
            })
    
    def traumatize(self, severity=1.0):
        """Direct trauma injection. Severity 0-1."""
        self.drift -= severity * self.range * 0.5
        self._trauma_log.append({
            "type": "trauma",
            "severity": severity,
            "drift_after": round(self.drift, 3),
        })
    
    def heal(self, amount=0.5):
        """Healing event. Doesn't erase trauma — lightens it."""
        self.drift *= (1.0 - amount * 0.3)  # max 30% reduction per heal
        self._trauma_log.append({
            "type": "heal",
            "amount": amount,
            "drift_after": round(self.drift, 3),
        })
    
    @property
    def state(self):
        """Current dark matter state summary."""
        total_bias = self.base_bias + self.drift
        if total_bias < -10:
            disposition = "darkened"
        elif total_bias < -3:
            disposition = "heavy"
        elif total_bias < 3:
            disposition = "balanced"
        elif total_bias < 10:
            disposition = "light"
        else:
            disposition = "radiant"
        
        return {
            "base_bias": round(self.base_bias, 2),
            "drift": round(self.drift, 2),
            "total_bias": round(total_bias, 2),
            "disposition": disposition,
            "range": self.range,
            "trauma_events": len(self._trauma_log),
            "stability": "stable" if self.range < 10 else "normal" if self.range < 20 else "volatile" if self.range < 35 else "chaotic",
        }
    
    def __str__(self):
        s = self.state
        return f"DarkMatter(X={s['total_bias']:+.1f}, {s['disposition']}, {s['stability']})"


# Preset dark matter profiles
def new_entity(name="default"):
    """Create a dark matter profile for common archetypes."""
    profiles = {
        "default":     DarkMatter(seed=42, range=15, resilience=0.999),
        "optimist":    DarkMatter(seed=100, range=10, drift=5.0, resilience=0.9995),
        "pessimist":   DarkMatter(seed=200, range=10, drift=-5.0, resilience=0.998),
        "traumatized": DarkMatter(seed=300, range=25, drift=-15.0, resilience=0.995),
        "resilient":   DarkMatter(seed=400, range=8, drift=-3.0, resilience=0.9999),
        "volatile":    DarkMatter(seed=500, range=35, resilience=0.99),
        "stoic":       DarkMatter(seed=600, range=3, resilience=0.9999),
        "empath":      DarkMatter(seed=700, range=20, resilience=0.997),
        "child":       DarkMatter(seed=800, range=30, drift=3.0, resilience=0.995),
        "elder":       DarkMatter(seed=900, range=8, drift=-2.0, resilience=0.9998),
    }
    return profiles.get(name, profiles["default"])
