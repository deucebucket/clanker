"""Output modes — same 5D engine, different output filters.

The engine ALWAYS computes full VADUG. These modes translate
the internal state to whatever format the consumer needs.

Like a camera shooting RAW but exporting JPEG/PNG/TIFF.
"""


class OutputMode:
    """Translate 5D VADUG into different output formats."""

    @staticmethod
    def three_color(v, a, d, u, g, intent_mode, entropy_classifier=None, history=None):
        """3-color sentiment: positive / negative / neutral.
        Uses V+G blend tuned for academic benchmarks."""
        # Entropy override
        if entropy_classifier and history:
            if entropy_classifier.should_override_neutral(history, v, intent_mode):
                return "neutral"

        # V+G blend (tuned: +0.5pp over V-only)
        blended = v * 0.9 + g * 0.1

        if blended > 128:
            return "positive"
        elif blended < 124:
            return "negative"
        return "neutral"

    @staticmethod
    def dimensional(v, a, d, u, g):
        """Raw 5D coordinates — for EmoBank, research, detailed analysis."""
        return {"v": v, "a": a, "d": d, "u": u, "g": g}

    @staticmethod
    def emotion_label(v, a, d, u, g):
        """Map VADUG to nearest named emotion."""
        # Distance-based matching against emotion prototypes
        EMOTIONS = [
            ("enraged",     20, 230, 180, 70, 170),
            ("furious",     30, 210, 170, 60, 165),
            ("angry",       50, 190, 160, 40, 155),
            ("frustrated",  65, 170, 130, 30, 140),
            ("annoyed",     85, 150, 135, 20, 130),
            ("anxious",     70, 200, 60,  60, 100),
            ("terrified",   30, 230, 30,  80, 40),
            ("scared",      50, 200, 50,  60, 60),
            ("nervous",     80, 170, 80,  40, 110),
            ("sad",         60, 120, 80,  10, 70),
            ("depressed",   30, 80,  40,  5,  30),
            ("grief",       20, 140, 30,  30, 20),
            ("lonely",      50, 100, 50,  15, 50),
            ("nostalgic",   90, 110, 100, 5,  90),
            ("bored",       100, 70, 120, 0,  110),
            ("neutral",     128, 128, 128, 0,  128),
            ("content",     150, 90,  145, 0,  145),
            ("calm",        145, 80,  150, 0,  140),
            ("relieved",    155, 90,  145, 0,  150),
            ("hopeful",     160, 130, 140, 10, 155),
            ("happy",       175, 150, 150, 5,  165),
            ("excited",     190, 190, 160, 15, 180),
            ("thrilled",    210, 200, 170, 10, 195),
            ("ecstatic",    230, 210, 180, 5,  220),
            ("loving",      200, 160, 155, 5,  190),
            ("grateful",    180, 130, 145, 5,  170),
            ("proud",       190, 150, 190, 5,  185),
            ("amused",      170, 160, 140, 5,  160),
            ("surprised",   140, 190, 120, 30, 150),
            ("shocked",     110, 220, 80,  50, 130),
            ("disgusted",   40, 170, 150, 20, 140),
        ]

        best_dist = float("inf")
        best_label = "neutral"
        for label, ev, ea, ed, eu, eg in EMOTIONS:
            dist = ((v-ev)**2 * 3 + (a-ea)**2 + (d-ed)**2 + (u-eu)**2 * 0.5 + (g-eg)**2 * 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    @staticmethod
    def crisis_check(v, a, d, u, g):
        """Binary safety check — is this person in crisis?"""
        if v < 50 and g < 50:
            return {"crisis": True, "severity": "high", "v": v, "g": g}
        if v < 70 and g < 70:
            return {"crisis": True, "severity": "moderate", "v": v, "g": g}
        if v < 60:
            return {"crisis": True, "severity": "moderate", "v": v, "g": g}
        return {"crisis": False, "severity": "none", "v": v, "g": g}

    @staticmethod
    def binary_sentiment(v, g):
        """Binary positive/negative — no neutral option. For SST-2 style tests."""
        blended = v * 0.9 + g * 0.1
        return "positive" if blended >= 126 else "negative"

    @staticmethod
    def confidence_score(v, a, d, u, g):
        """How confident are we in this reading? 0.0 to 1.0."""
        # Distance from center across all dimensions
        total_deviation = abs(v-128) + abs(a-128) + abs(d-128) + u + abs(g-128)
        # More deviation = more confident (something definitely happened)
        confidence = min(1.0, total_deviation / 200.0)
        return round(confidence, 3)
