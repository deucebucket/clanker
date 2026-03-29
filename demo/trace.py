"""Glass-box trace — captures every transformation for debugging."""

import re


class PipelineTrace:
    """Captures every transformation for debugging."""

    def __init__(self):
        self.steps = []
        self.bookend = None
        self.intent = None
        self.misses = []  # sentences where prediction != truth

    def trace_sentence(self, text, truth_label=None):
        """Run full pipeline on one sentence, capture every step."""
        from .pendulum import SequentialPendulum
        from .intent import IntentDetector
        from .classifier import AdaptiveClassifier
        from .bookend import BookendParser
        from .tonal import TonalAnalyzer

        intent_det = IntentDetector()
        classifier = AdaptiveClassifier()
        bookend = BookendParser()
        tonal = TonalAnalyzer()

        trace = {"text": text, "steps": []}

        # Bookend
        bp = bookend.parse(text)
        trace["bookend"] = bp

        # Intent
        mode, conf = intent_det.detect(text)
        trace["intent"] = {"mode": mode, "confidence": conf}

        # Pendulum (captures word-by-word trace)
        p = SequentialPendulum()
        words = re.findall(r"[a-z']+", text.lower())
        for i, w in enumerate(words):
            v_before = int(p.v)
            p.process_word(w, words, i)
            v_after = int(p.v)
            step = {
                "word": w,
                "v_before": v_before,
                "v_after": v_after,
                "v_delta": v_after - v_before,
            }
            # Get source from history if available
            if p.history and len(p.history) > i:
                h = p.history[i] if i < len(p.history) else {}
                step["source"] = h.get("source", "unknown")
            trace["steps"].append(step)

        # Tonal
        tone = tonal.analyze(p.history, intent_mode=mode)
        trace["tonal"] = {"tone": tone["tone"], "confidence": tone["confidence"]}

        # Final
        final_v = int(p.v)
        pred = classifier.classify(final_v, mode)
        trace["final_v"] = final_v
        trace["final_g"] = int(p.g)
        trace["prediction"] = pred
        trace["truth"] = truth_label
        trace["correct"] = pred == truth_label if truth_label else None

        # If wrong, analyze why
        if truth_label and pred != truth_label:
            trace["diagnosis"] = self._diagnose(trace, truth_label, pred)
            self.misses.append(trace)

        return trace

    def _diagnose(self, trace, truth, pred):
        """Try to identify WHY the prediction was wrong."""
        reasons = []

        final_v = trace["final_v"]

        if pred == "neutral" and truth != "neutral":
            reasons.append(f"Landed neutral (V={final_v}) -- forces too weak or cancelled out")
            # Find which words pushed toward neutral
            for step in trace["steps"]:
                if abs(step["v_delta"]) < 3:
                    reasons.append(f"  '{step['word']}' had no impact (delta={step['v_delta']})")

        if pred == "positive" and truth == "negative":
            reasons.append(f"False positive (V={final_v}) -- positive forces overwhelmed negative")
            pos_words = [s for s in trace["steps"] if s["v_delta"] > 5]
            for s in pos_words[:3]:
                reasons.append(f"  '{s['word']}' pushed V up {s['v_delta']}")

        if pred == "negative" and truth == "positive":
            reasons.append(f"False negative (V={final_v}) -- negative forces overwhelmed positive")
            neg_words = [s for s in trace["steps"] if s["v_delta"] < -5]
            for s in neg_words[:3]:
                reasons.append(f"  '{s['word']}' pushed V down {s['v_delta']}")

        if trace["bookend"]["pattern"] == "isolation" and truth == "negative" and pred != "negative":
            reasons.append("Bookend detected isolation but final result wasn't negative")

        return reasons

    def save_misses(self, path):
        """Save all missed predictions for analysis."""
        import json
        with open(path, 'w') as f:
            json.dump(self.misses, f, indent=2)

    def print_trace(self, trace):
        """Pretty print a single trace."""
        print(f"\nINPUT: {trace['text']}")
        bp = trace['bookend']
        print(f"[BOOKEND] {bp['pattern']} ({bp['direction']}) tail={bp['tail_modifier']}")
        print(f"[INTENT] {trace['intent']['mode']} ({trace['intent']['confidence']:.2f})")
        print(f"[TONAL] {trace['tonal']['tone']} ({trace['tonal']['confidence']:.2f})")
        print(f"\n  {'Word':<15} {'V before':>8} {'V after':>8} {'Delta':>6}")
        print(f"  {'-'*40}")
        for step in trace['steps']:
            delta = step['v_delta']
            marker = '^' if delta > 3 else 'v' if delta < -3 else '.'
            print(f"  {step['word']:<15} {step['v_before']:>8} {step['v_after']:>8} {delta:>+5} {marker}")
        print(f"\n  Final: V={trace['final_v']} G={trace['final_g']} -> {trace['prediction']}" +
              (f" (truth: {trace['truth']})" if trace['truth'] else ""))
        if 'diagnosis' in trace:
            print(f"  DIAGNOSIS:")
            for r in trace['diagnosis']:
                print(f"    {r}")
