"""Two-pass tonal analysis for sarcasm and tone detection.

Pass 1 (handled by SequentialPendulum): process words, get raw VADUG.
    This is "what the words literally say."

Pass 2 (this module): analyze the PATTERN of forces, not just the final state.
    Look at the trajectory shape for tonal signatures.

The key insight: sarcasm has a distinctive tonal signature. Positive words in
a negative trajectory create a specific pattern that's different from genuine
positivity.
"""

import statistics


class TonalAnalyzer:
    """Pass 2: Analyze emotional trajectory patterns for tone detection.

    Reads from pendulum.history (already tracked). Called AFTER the pendulum
    processes all words but BEFORE the grader scores the result.
    """

    # Tone labels
    SINCERE = "sincere"
    SARCASTIC = "sarcastic"
    DEADPAN = "deadpan"
    HYPERBOLIC = "hyperbolic"
    UNDERSTATED = "understated"

    def analyze(self, word_trajectory: list[dict], intent_mode: str = None) -> dict:
        """Analyze the pendulum's history for tonal patterns.

        Args:
            word_trajectory: list of trace dicts from pendulum.history,
                each with keys 'word', 'v', 'a', 'd', 'u', 'g'.
            intent_mode: optional intent from Layer 0 (EMOTIONAL, VENTING, etc.)
                If EMOTIONAL/VENTING, sarcasm detection thresholds are raised
                to prevent false positives on genuine distress.

        Returns:
            {
                "tone": "sincere" | "sarcastic" | "deadpan" | "hyperbolic" | "understated",
                "confidence": 0.0-1.0,
                "signals": [...list of detected signal descriptions...],
                "adjustment": optional dict with recommended VADUG adjustments
            }
        """
        if not word_trajectory or len(word_trajectory) < 2:
            return {
                "tone": self.SINCERE,
                "confidence": 0.5,
                "signals": [],
                "adjustment": None,
            }

        # Suppress sarcasm on emotional/crisis inputs — nobody says "I lost my job" sarcastically
        suppress_sarcasm = intent_mode in ("EMOTIONAL", "VENTING", "CRISIS")

        signals = []

        # Run all signal detectors (skip sarcasm signals on emotional intent)
        if not suppress_sarcasm:
            signals.extend(self._detect_valence_whiplash(word_trajectory))
            signals.extend(self._detect_intensity_mismatch(word_trajectory))
            signals.extend(self._detect_peak_and_fade(word_trajectory))
        signals.extend(self._detect_deadpan(word_trajectory))
        signals.extend(self._detect_hyperbole(word_trajectory))
        signals.extend(self._detect_understatement(word_trajectory))
        signals.extend(self._detect_trailing_drop(word_trajectory))

        # Classify tone from signals
        tone, confidence, adjustment = self._classify_tone(signals, word_trajectory)

        return {
            "tone": tone,
            "confidence": round(confidence, 2),
            "signals": [s["description"] for s in signals],
            "adjustment": adjustment,
        }

    # ── Signal detectors ──────────────────────────────────────────────

    def _detect_valence_whiplash(self, traj):
        """Signal 1: V jumps then drops within 3 words = sarcasm candidate.

        Thresholds tuned for real pendulum output: a spike of +20 followed
        by a drop of -15 within 3 words is enough for a candidate signal.
        """
        signals = []
        for i in range(1, len(traj)):
            spike = traj[i]['v'] - traj[i - 1]['v']
            if spike >= 20:
                # Check next 3 words for a drop
                for j in range(i + 1, min(i + 4, len(traj))):
                    drop = traj[i]['v'] - traj[j]['v']
                    if drop >= 15:
                        conf = min(1.0, (spike + drop) / 60.0)
                        signals.append({
                            "type": "valence_whiplash",
                            "confidence": conf,
                            "description": (
                                f"WHIPLASH: '{traj[i]['word']}' spiked V+{spike:.0f} "
                                f"then dropped V-{drop:.0f} by '{traj[j]['word']}'"
                            ),
                        })
                        break
        return signals

    def _detect_intensity_mismatch(self, traj):
        """Signal 2: Arousal spike with positive push but low final valence."""
        signals = []
        final_v = traj[-1]['v']

        for i in range(len(traj)):
            if traj[i]['a'] > 155:
                if i > 0:
                    v_push = traj[i]['v'] - traj[i - 1]['v']
                else:
                    v_push = traj[i]['v'] - 128

                if v_push > 10 and final_v < 140:
                    conf = min(1.0, (traj[i]['a'] - 130) / 80.0 * (1.0 - final_v / 180.0))
                    conf = max(0.0, conf)
                    signals.append({
                        "type": "intensity_mismatch",
                        "confidence": conf,
                        "description": (
                            f"INTENSITY MISMATCH: '{traj[i]['word']}' has A={traj[i]['a']} "
                            f"with V+{v_push:.0f} push but final V={final_v:.0f}"
                        ),
                    })
        return signals

    def _detect_peak_and_fade(self, traj):
        """Signal: Positive peak in the trajectory that fades to lower final V.

        Catches "I love working overtime" pattern: a strong positive word
        pushes V high, but subsequent words erode it back. The peak-to-end
        drop indicates the positive was undercut.
        """
        signals = []
        if len(traj) < 3:
            return signals

        v_values = [t['v'] for t in traj]
        peak_v = max(v_values)
        peak_idx = v_values.index(peak_v)
        final_v = v_values[-1]

        # Peak must be clearly positive and not at the end
        if peak_v > 145 and peak_idx < len(traj) - 1:
            fade = peak_v - final_v
            if fade >= 15:
                # Confidence scales with how much the positive faded
                conf = min(1.0, fade / 40.0)
                # Boost if final V is back near or below neutral
                if final_v < 145:
                    conf = min(1.0, conf + 0.15)
                signals.append({
                    "type": "peak_and_fade",
                    "confidence": conf,
                    "description": (
                        f"PEAK & FADE: V peaked at {peak_v} "
                        f"('{traj[peak_idx]['word']}') then faded to {final_v} "
                        f"(drop={fade:.0f})"
                    ),
                })
        return signals

    def _detect_deadpan(self, traj):
        """Signal 3: V stays flat despite emotionally loaded words."""
        signals = []
        if len(traj) < 3:
            return signals

        v_values = [t['v'] for t in traj]
        v_variance = statistics.variance(v_values) if len(v_values) > 1 else 0

        # Check if there are emotionally loaded words (arousal varies)
        a_values = [t['a'] for t in traj]
        a_range = max(a_values) - min(a_values)

        if v_variance < 100 and a_range > 20:
            conf = min(1.0, a_range / 60.0 * (1.0 - v_variance / 100.0))
            conf = max(0.0, conf)
            signals.append({
                "type": "deadpan",
                "confidence": conf,
                "description": (
                    f"DEADPAN: V variance={v_variance:.1f} (stdev={v_variance**0.5:.1f}) "
                    f"despite A range={a_range:.0f}"
                ),
            })
        return signals

    def _detect_hyperbole(self, traj):
        """Signal 4: Sustained high arousal with strong positive valence = exaggeration.

        Tuned for real pendulum output where A typically tops out around
        140-200 for emphatic phrases. Also checks for consistently elevated
        V above neutral as a sign of "everything is amazing" hyperbole.
        """
        signals = []
        if len(traj) < 3:
            return signals

        a_values = [t['a'] for t in traj]
        v_values = [t['v'] for t in traj]

        avg_a = statistics.mean(a_values)
        max_a = max(a_values)
        avg_v = statistics.mean(v_values)
        min_v = min(v_values)

        # Hyperbole: arousal elevated, valence consistently positive,
        # and V never dips below neutral (it's ALL positive, no nuance).
        # We look at the "emotional" portion of the trajectory — words that
        # actually moved the pendulum — to avoid neutral filler diluting the ratio.
        high_a_count = sum(1 for a in a_values if a > 140)

        # Also count words where V is clearly above neutral
        positive_count = sum(1 for v in v_values if v > 140)

        # Peak valence — how high did it go?
        peak_v = max(v_values)

        if (positive_count >= 3 and peak_v > 150 and min_v >= 125
                and high_a_count >= 2 and len(traj) > 4):
            # Confidence based on how elevated the emotional words are
            v_elevation = (avg_v - 128) / 80.0
            a_elevation = (avg_a - 128) / 80.0
            sustained = positive_count / len(v_values)
            conf = min(1.0, (v_elevation + a_elevation + sustained) / 2.0)
            conf = max(0.0, conf)
            if conf > 0.25:
                signals.append({
                    "type": "hyperbole",
                    "confidence": conf,
                    "description": (
                        f"HYPERBOLE: {positive_count}/{len(v_values)} steps V>140, "
                        f"peak V={peak_v}, {high_a_count} steps A>140 "
                        f"(sustained positivity)"
                    ),
                })
        return signals

    def _detect_understatement(self, traj):
        """Signal 5: Loaded emotional context but very low arousal = 'it's fine' energy.

        For short phrases like "I'm fine", we look at the final state:
        V moved away from neutral but arousal stayed flat.
        """
        signals = []
        if len(traj) < 2:
            return signals

        a_values = [t['a'] for t in traj]
        avg_a = statistics.mean(a_values)
        max_a = max(a_values)

        v_values = [t['v'] for t in traj]
        final_v = v_values[-1]
        v_deviation = abs(final_v - 128)

        # Also check if V dipped below neutral (negative emotional content)
        # with arousal staying low — classic understatement
        v_moved = v_deviation > 5 or abs(statistics.mean(v_values) - 128) > 5

        if max_a < 145 and avg_a < 140 and v_moved:
            # Confidence based on how flat the arousal is relative to V movement
            flatness = (145 - avg_a) / 30.0
            movement = min(1.0, (v_deviation + 5) / 25.0)
            conf = min(1.0, flatness * movement)
            conf = max(0.0, conf)
            if conf > 0.2:
                signals.append({
                    "type": "understatement",
                    "confidence": conf,
                    "description": (
                        f"UNDERSTATED: avg A={avg_a:.0f}, max A={max_a:.0f}, "
                        f"final V={final_v:.0f} (V moved {v_deviation:.0f} from neutral)"
                    ),
                })
        return signals

    def _detect_trailing_drop(self, traj):
        """Signal 6: Sentence starts very positive, last 2 words pull negative."""
        signals = []
        if len(traj) < 4:
            return signals

        mid = len(traj) // 2
        early_v = statistics.mean([t['v'] for t in traj[:mid]])
        late_v = statistics.mean([t['v'] for t in traj[-2:]])

        drop = early_v - late_v
        if early_v > 140 and drop > 15:
            conf = min(1.0, drop / 50.0 * (early_v - 128) / 60.0)
            conf = max(0.0, conf)
            signals.append({
                "type": "trailing_drop",
                "confidence": conf,
                "description": (
                    f"TRAILING DROP: early avg V={early_v:.0f} -> "
                    f"last 2 words avg V={late_v:.0f} (drop={drop:.0f})"
                ),
            })
        return signals

    # ── Tone classification ───────────────────────────────────────────

    def _classify_tone(self, signals, traj):
        """Determine overall tone from detected signals.

        Returns (tone, confidence, adjustment).
        """
        if not signals:
            return self.SINCERE, 0.8, None

        # Group signals by type
        by_type = {}
        for s in signals:
            t = s["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)

        # Sarcasm signals
        sarcasm_types = {"valence_whiplash", "intensity_mismatch", "trailing_drop",
                         "peak_and_fade"}
        sarcasm_signals = [s for s in signals if s["type"] in sarcasm_types]
        sarcasm_conf = max((s["confidence"] for s in sarcasm_signals), default=0.0)
        # Boost confidence when multiple sarcasm signals agree
        if len(sarcasm_signals) >= 2:
            sarcasm_conf = min(1.0, sarcasm_conf + 0.15 * (len(sarcasm_signals) - 1))

        # Deadpan
        deadpan_signals = by_type.get("deadpan", [])
        deadpan_conf = max((s["confidence"] for s in deadpan_signals), default=0.0)

        # Hyperbole
        hyperbole_signals = by_type.get("hyperbole", [])
        hyperbole_conf = max((s["confidence"] for s in hyperbole_signals), default=0.0)

        # Understatement
        understatement_signals = by_type.get("understatement", [])
        understatement_conf = max((s["confidence"] for s in understatement_signals), default=0.0)

        # Pick the dominant tone
        candidates = [
            (self.SARCASTIC, sarcasm_conf),
            (self.DEADPAN, deadpan_conf),
            (self.HYPERBOLIC, hyperbole_conf),
            (self.UNDERSTATED, understatement_conf),
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        best_tone, best_conf = candidates[0]

        # If the best non-sincere signal is weak, call it sincere.
        # Use a lower threshold for understatement/deadpan since they are
        # inherently subtle signals.
        threshold = 0.2 if best_tone in (self.UNDERSTATED, self.DEADPAN) else 0.3
        if best_conf < threshold:
            return self.SINCERE, 0.8 - best_conf, None

        # Build adjustment recommendation for sarcasm
        adjustment = None
        if best_tone == self.SARCASTIC and best_conf >= 0.5:
            final_v = traj[-1]['v']
            if final_v > 128:
                # Literal reading was positive but tone says sarcastic:
                # recommend pulling V down
                pull = int((final_v - 128) * min(1.0, best_conf) * 0.7)
                adjustment = {
                    "v_delta": -pull,
                    "reason": (
                        f"Sarcastic tone detected (conf={best_conf:.2f}): "
                        f"pull V down by {pull} from {final_v:.0f}"
                    ),
                }

        return best_tone, best_conf, adjustment

    def display(self, result, verbose=True):
        """Print the tonal analysis report."""
        if not verbose:
            return
        if result["tone"] == self.SINCERE and not result["signals"]:
            return

        print(f"\n--- TONAL ANALYSIS (Pass 2) ---")
        print(f"  Tone: {result['tone']} (confidence: {result['confidence']:.0%})")
        for i, signal in enumerate(result["signals"]):
            print(f"  Signal {i + 1}: {signal}")
        if result["adjustment"]:
            print(f"  Adjustment: {result['adjustment']['reason']}")


def apply_tonal_adjustment(v, a, d, u, g, tone_result, intent_mode):
    """Adjust VADUG based on detected tone.

    Called AFTER pendulum processing to correct V when sarcasm is detected
    on non-emotional content. Sarcasm on STATEMENT/CASUAL intent means
    the surface-positive words are ironic — flip V toward negative.

    Never adjusts EMOTIONAL/VENTING/CRISIS — those are genuine.
    """
    if tone_result['tone'] != 'sarcastic':
        return v, a, d, u, g
    if tone_result['confidence'] < 0.7:
        return v, a, d, u, g
    if intent_mode in ('EMOTIONAL', 'VENTING', 'CRISIS'):
        return v, a, d, u, g  # never flip genuine emotion

    # Sarcasm detected on non-emotional content
    if v > 140:
        # Reads positive but is sarcastic -> flip toward negative
        v = int(128 - (v - 128) * 0.6)

    return v, a, d, u, g
