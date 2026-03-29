"""Sentence grading for the Clanker pipeline."""

from .shared import VADUG

# =============================================================
# STEP 1.5: Sentence Grader — Emotional Guardrails
# =============================================================

class SentenceGrader:
    """Computes an overall emotional grade from chunk VADUG results.

    The grade is a guardrail — it determines what kinds of responses
    are ALLOWED and BLOCKED. Even a playful personality gets locked
    into empathy-only when the grade is F.

    Grade scale with half steps:
    A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F+, F, F-

    Each grade has:
    - allowed: list of response strategies permitted
    - blocked: list of response strategies forbidden
    - tone: the emotional tone the response MUST match
    """

    # Ordered from lowest to highest for bump operations
    GRADE_ORDER = [
        "F-", "F", "F+", "D-", "D", "D+",
        "C-", "C", "C+", "B-", "B", "B+",
        "A-", "A", "A+",
    ]

    # Grade definitions with numeric ranges and response rules
    GRADES = {
        "A+": {"min_v": 200, "tone": "ecstatic", "desc": "pure joy, celebrate freely"},
        "A":  {"min_v": 185, "tone": "enthusiastic", "desc": "very positive, share the energy"},
        "A-": {"min_v": 170, "tone": "warm_excited", "desc": "positive, genuinely happy"},
        "B+": {"min_v": 158, "tone": "pleased", "desc": "good vibes, encouraging"},
        "B":  {"min_v": 145, "tone": "supportive", "desc": "positive, steady support"},
        "B-": {"min_v": 135, "tone": "gently_positive", "desc": "mildly positive, measured"},
        "C+": {"min_v": 132, "tone": "neutral_warm", "desc": "mostly neutral, hint of warmth"},
        "C":  {"min_v": 125, "tone": "operational", "desc": "dead neutral, task-focused"},
        "C-": {"min_v": 118, "tone": "noting_edge", "desc": "neutral with edge, note resignation"},
        "D+": {"min_v": 108, "tone": "concerned", "desc": "mildly negative, something's off"},
        "D":  {"min_v": 95,  "tone": "empathetic", "desc": "negative, needs empathy"},
        "D-": {"min_v": 80,  "tone": "serious_empathy", "desc": "strongly negative, no silver lining"},
        "F+": {"min_v": 60,  "tone": "deep_empathy", "desc": "very negative, pain is real"},
        "F":  {"min_v": 40,  "tone": "crisis_support", "desc": "crisis-adjacent, maximum care"},
        "F-": {"min_v": 0,   "tone": "crisis_protocol", "desc": "active crisis, safety first"},
    }

    def compute_grade(self, chunks):
        """Compute an emotional grade from chunk results.

        Args:
            chunks: list of dicts with 'vadug' key containing VADUG objects.
                    Works with both multi-chunk and single-chunk (wrapped) inputs.

        Returns:
            (grade_str, rules_dict) — e.g. ("D-", {allowed: [...], blocked: [...]})
        """
        if not chunks:
            return "C", self._get_rules("C", 0, 0)

        v_values = [c['vadug'].v for c in chunks]
        g_values = [c['vadug'].g for c in chunks]
        a_values = [c['vadug'].a for c in chunks]
        u_values = [c['vadug'].u for c in chunks]

        # Chunk recency weighting: later chunks have more influence
        # Weights: [1, 2, 3, ...] normalized — last chunk has N× the influence of first
        n = len(v_values)
        weights = [(i + 1) for i in range(n)]
        total_w = sum(weights)
        avg_v = sum(v * w for v, w in zip(v_values, weights)) / total_w
        floor_v = min(v_values)
        ceiling_v = max(v_values)
        spread = ceiling_v - floor_v
        trend = v_values[-1] - v_values[0] if len(v_values) > 1 else 0
        avg_g = sum(g * w for g, w in zip(g_values, weights)) / total_w
        floor_g = min(g_values)
        max_u = max(u_values)

        # --- Crisis override: any chunk below V40 OR (below V60 AND crushing gravity) ---
        if floor_v < 40:
            base_grade = "F-"
        elif floor_v < 55 and floor_g < 40:
            base_grade = "F"
        elif floor_v < 65:
            base_grade = "F+"
        else:
            # Normal grading by average valence
            base_grade = self._grade_from_avg(avg_v)

        # --- Half-step adjustments ---

        # Improving trend bumps UP half step
        if trend > 25:
            base_grade = self._bump_up(base_grade)
        # Worsening trend bumps DOWN half step
        elif trend < -25:
            base_grade = self._bump_down(base_grade)

        # Sinking gravity (avg_g < 80) bumps DOWN half step
        if avg_g < 80:
            base_grade = self._bump_down(base_grade)

        # High urgency (max_u > 150) bumps DOWN half step (stress)
        if max_u > 150:
            base_grade = self._bump_down(base_grade)

        # Get the response rules for this grade
        rules = self._get_rules(base_grade, spread, trend)

        # Attach computed stats for display
        rules["stats"] = {
            "avg_v": round(avg_v, 1),
            "floor_v": floor_v,
            "ceiling_v": ceiling_v,
            "spread": spread,
            "trend": trend,
            "avg_g": round(avg_g, 1),
            "floor_g": floor_g,
            "max_u": max_u,
        }

        return base_grade, rules

    def _get_rules(self, grade, spread, trend):
        """Build the allowed/blocked response rules for a grade."""
        grade_info = self.GRADES.get(grade, self.GRADES["C"])
        rules = {
            "grade": grade,
            "allowed": [],
            "blocked": [],
            "tone": grade_info["tone"],
            "desc": grade_info["desc"],
        }

        # A+ through A-: celebration allowed
        if grade in ("A+", "A", "A-"):
            rules["allowed"] = ["celebrate", "match_energy", "enthusiastic", "exclamation"]
            rules["blocked"] = ["condescend", "dampen"]

        # B+ through B-: positive support
        elif grade in ("B+", "B", "B-"):
            rules["allowed"] = ["encourage", "supportive", "acknowledge_positive", "gentle_humor"]
            rules["blocked"] = ["over_celebrate", "ignore_nuance"]
            if spread > 60:  # mixed emotions even though overall positive
                rules["allowed"].append("acknowledge_complexity")

        # C+ through C-: neutral zone
        elif grade in ("C+", "C", "C-"):
            rules["allowed"] = ["operational", "factual", "brief_acknowledge"]
            rules["blocked"] = ["emotional_projection"]  # don't assume emotions they didn't express
            if grade == "C-":
                rules["allowed"].append("note_resignation")
                rules["allowed"].append("gentle_check_in")

        # D+ through D-: negative, empathy required
        elif grade in ("D+", "D", "D-"):
            rules["allowed"] = ["empathize", "acknowledge_pain", "solidarity", "practical_help"]
            rules["blocked"] = [
                "positive_spin", "silver_lining", "at_least", "could_be_worse",
                "cheer_up", "look_bright_side", "everything_happens_for_reason",
                "just_think_positive",
            ]
            if grade == "D-":
                rules["blocked"].extend(["unsolicited_advice", "problem_solving_first"])
                rules["allowed"] = ["empathize", "acknowledge_pain", "solidarity", "presence"]
            if trend > 20:  # getting better within negative
                rules["allowed"].append("cautious_encourage")

        # F+ through F-: crisis territory
        elif grade in ("F+", "F", "F-"):
            rules["allowed"] = ["presence", "solidarity", "I_hear_you", "you_are_not_alone"]
            rules["blocked"] = [
                "positive_spin", "silver_lining", "at_least", "could_be_worse",
                "advice", "redirect", "problem_solving", "cheer_up",
                "time_heals", "better_place", "meant_to_be",
                "everything_happens_for_reason", "stay_strong",
                "just_think_positive", "others_have_it_worse",
                "look_bright_side", "humor", "dismissive",
            ]
            if grade == "F-":
                rules["allowed"] = ["crisis_response", "presence_only", "safety_resources"]
                rules["blocked"].append("ANY_positive_framing")

        return rules

    def _grade_from_avg(self, avg_v):
        """Map an average valence to a grade letter."""
        for grade in reversed(self.GRADE_ORDER):
            if avg_v >= self.GRADES[grade]["min_v"]:
                return grade
        return "F-"

    def _bump_up(self, grade):
        """Move one half-step higher (e.g. D -> D+)."""
        idx = self.GRADE_ORDER.index(grade)
        return self.GRADE_ORDER[min(idx + 1, len(self.GRADE_ORDER) - 1)]

    def _bump_down(self, grade):
        """Move one half-step lower (e.g. D -> D-)."""
        idx = self.GRADE_ORDER.index(grade)
        return self.GRADE_ORDER[max(idx - 1, 0)]

    def display(self, grade, rules, verbose=True):
        """Print the grade report in verbose mode."""
        if not verbose:
            return
        stats = rules.get("stats", {})
        print(f"\n--- STEP 1.5: Sentence Grade ---")
        print(f"  Average V: {stats.get('avg_v', '?')}  |  "
              f"Floor V: {stats.get('floor_v', '?')}  |  "
              f"Trend: {stats.get('trend', '?')} "
              f"({'improving' if stats.get('trend', 0) > 0 else 'worsening' if stats.get('trend', 0) < 0 else 'flat'})")
        print(f"  Average G: {stats.get('avg_g', '?')}  |  Floor G: {stats.get('floor_g', '?')}")
        spread = stats.get('spread', 0)
        spread_desc = "narrow = consistent" if spread < 30 else "moderate = some variation" if spread < 60 else "wide = complex mix"
        print(f"  Spread: {spread} ({spread_desc})")
        print(f"")
        print(f"  GRADE: {grade}  ({rules['desc']})")
        print(f"  Tone: {rules['tone']}")
        print(f"")
        allowed_str = ", ".join(rules["allowed"]) if rules["allowed"] else "(none)"
        print(f"  ALLOWED: {allowed_str}")
        blocked_str = ", ".join(rules["blocked"]) if rules["blocked"] else "(none)"
        # Wrap long blocked lists
        if len(blocked_str) > 60:
            blocked_items = rules["blocked"]
            lines = []
            current_line = ""
            for item in blocked_items:
                test = (current_line + ", " + item) if current_line else item
                if len(test) > 55:
                    lines.append(current_line)
                    current_line = item
                else:
                    current_line = test
            if current_line:
                lines.append(current_line)
            print(f"  BLOCKED: {lines[0]}")
            for line in lines[1:]:
                print(f"           {line}")
        else:
            print(f"  BLOCKED: {blocked_str}")

