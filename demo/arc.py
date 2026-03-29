"""Emotional arc detection and pipeline orchestration for the Clanker pipeline."""

import re
import random

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .pendulum import SequentialPendulum
from .response import (
    classify_metadata, compute_harmony, nearest_emotion,
    generate_clanker, decode_response, ResponseBuilder,
)
from .personality import apply_personality
from .chunker import ChunkSplitter
from .grader import SentenceGrader
from .sarcasm import SarcasmDetector
from .tonal import TonalAnalyzer, apply_tonal_adjustment
from .intent import IntentDetector


# ── Arc Closers ──

ARC_CLOSERS = {
    "valley": [  # was bad, now good
        "How many people get to say that? Congrats.",
        "Sounds like it's all working out.",
        "That's a hell of a silver lining.",
    ],
    "peak": [  # was good, now bad
        "I'm here if you need to talk through it.",
        "That's rough. We'll figure it out.",
    ],
    "descending": [  # getting progressively worse
        "That's a lot. Let's take it one thing at a time.",
        "I hear you. We'll work through this together.",
    ],
    "ascending": [  # getting progressively better
        "Things are looking up!",
        "Love to see the momentum.",
    ],
    "flat_negative": [  # sustained bad
        "You're not alone in this.",
        "I'm here. What do you need right now?",
    ],
    "flat_positive": [  # sustained good
        "That's amazing all around!",
        "Everything's clicking!",
    ],
    "contrast": [  # brief positive emphasizes surrounding negativity
        "That one bright moment doesn't erase the rest.",
        "I hear the weight around that one good thing.",
        "The hard stuff is still there. I see that.",
    ],
    "mixed": [  # complex
        "That's a lot of feelings. All valid.",
        "Life's complicated like that. I'm here for all of it.",
    ],
}


# =============================================================
# STEP 1.5: Sentence Grader — Emotional Guardrails
# =============================================================

class ChunkedPipeline:
    """Runs the full Clanker pipeline per-chunk and assembles an arc-aware response.

    For paragraphs with multiple emotional beats:
    1. Split into chunks at natural boundaries
    2. Run a FRESH pendulum on each chunk
    3. Analyze the emotional arc across chunks
    4. Build a summary response (G axis controls length, not per-chunk play-by-play)
    5. Assemble into one coherent reply

    v0.9.1: Emotional density — uses build_summary_response instead of
    per-chunk build_full_response. Heavy inputs (low G) get brief responses.
    """

    def __init__(self):
        self.splitter = ChunkSplitter()
        self._used_phrases = set()  # tracks response openers across chunks in a single run

    def _choose_unique(self, options):
        """Pick a phrase from *options* that hasn't been used yet in this pipeline run.

        If every option has already been used, prefer the least-recently-added
        one (i.e. fall back to random from the full list so we don't crash).
        """
        unused = [o for o in options if o not in self._used_phrases]
        if unused:
            choice = random.choice(unused)
        else:
            # All exhausted — allow reuse but still randomise
            choice = random.choice(options)
        self._used_phrases.add(choice)
        return choice

    def process(self, text: str, personality: PersonalityVector,
                verbose: bool = True, show_trace: bool = False):
        """Run the chunked pipeline. Returns (assembled_response, chunk_results, arc)."""
        # Reset per-run phrase tracking so multi-chunk responses avoid repetition
        self._used_phrases.clear()

        # 1. Split into chunks
        chunks = self.splitter.split(text)

        if verbose:
            print(f"\n--- STEP 1: Emotional Chunking ---")
            print(f"  Input split into {len(chunks)} chunks:")

        # 2. Run pendulum on each chunk separately, then apply tonal adjustment
        chunk_results = []
        tonal = TonalAnalyzer()
        intent_det = IntentDetector()
        intent_mode, _intent_conf = intent_det.detect(text)
        for i, chunk in enumerate(chunks):
            pendulum = SequentialPendulum()  # FRESH pendulum per chunk
            vadug, history = pendulum.process_text(chunk)

            # Tonal sarcasm adjustment per chunk
            tone_result = tonal.analyze(history, intent_mode=intent_mode)
            old_v = vadug.v
            new_v, new_a, new_d, new_u, new_g = apply_tonal_adjustment(
                vadug.v, vadug.a, vadug.d, vadug.u, vadug.g,
                tone_result, intent_mode,
            )
            if new_v != old_v:
                vadug = VADUG(v=new_v, a=new_a, d=new_d, u=new_u, g=new_g)

            emotion = nearest_emotion(vadug)
            chunk_results.append({
                'text': chunk,
                'vadug': vadug,
                'history': history,
                'pendulum': pendulum,
                'emotion': emotion,
                'index': i,
                'tone': tone_result,
            })

            if verbose:
                # Gravity descriptor
                g_desc = ""
                if vadug.g < 60:
                    g_desc = "sinking"
                elif vadug.g < 100:
                    g_desc = "heavy"
                elif vadug.g < 148:
                    g_desc = "grounded"
                elif vadug.g < 200:
                    g_desc = "light"
                else:
                    g_desc = "soaring!"
                print(f"\n  Chunk {i+1}: \"{chunk}\"")
                print(f"    VADUG: V{vadug.v} A{vadug.a} D{vadug.d} U{vadug.u} G{vadug.g}")
                print(f"    Emotion: {emotion} ({g_desc})")
                if show_trace:
                    print(pendulum.render_trace())

        # 3. Analyze the emotional arc
        arc = self.analyze_arc(chunk_results)
        if verbose:
            print(f"\n  Arc: {arc.upper()} ({self._arc_description(arc, chunk_results)})")

        # 3.5. Sentence grader — emotional guardrail
        grader = SentenceGrader()
        grade, grade_rules = grader.compute_grade(chunk_results)
        grader.display(grade, grade_rules, verbose=verbose)

        # 3.6. Sarcasm detection — three-signal analysis from pendulum trajectory
        sarcasm = SarcasmDetector()
        sarcasm_flag = False
        all_sarcasm_signals = []
        context_signal = None

        # Check each chunk's trajectory for reversal and mismatch signals
        for cr in chunk_results:
            detected, conf, signals = sarcasm.analyze_trajectory(cr['history'])
            if detected:
                all_sarcasm_signals.extend(signals)

        # Check for context contradiction across chunks
        for i, cr in enumerate(chunk_results):
            if i > 0:
                prev = chunk_results[:i]
                is_contradicted, detail = sarcasm.analyze_context(prev, cr)
                if is_contradicted:
                    context_signal = detail
                    cr['sarcasm'] = True
                    cr['sarcasm_detail'] = detail

        # Combine trajectory signals + context contradiction for overall confidence
        total_signals = len(all_sarcasm_signals) + (1 if context_signal else 0)
        if total_signals >= 3:
            sarcasm_confidence = SarcasmDetector.HIGH
        elif total_signals == 2:
            sarcasm_confidence = SarcasmDetector.MODERATE
        elif total_signals == 1:
            sarcasm_confidence = SarcasmDetector.LOW
        else:
            sarcasm_confidence = SarcasmDetector.NONE

        # Adjust grade if sarcasm detected
        grade_note = ""
        if sarcasm_confidence >= SarcasmDetector.LOW:
            sarcasm_flag = True
            grade, grade_note = sarcasm.adjust_grade(grade, sarcasm_confidence, grader)
            # Recompute rules with adjusted grade
            stats = grade_rules.get("stats", {})
            grade_rules = grader._get_rules(grade, stats.get("spread", 0), stats.get("trend", 0))
            grade_rules["stats"] = stats

        sarcasm.display(
            sarcasm_flag, sarcasm_confidence, all_sarcasm_signals,
            context_signal=context_signal, grade_note=grade_note,
            verbose=verbose
        )

        # 4. Generate response via ResponseBuilder (math-based, summary mode)
        if verbose:
            print(f"\n--- STEP 2: Emotional Density (ResponseBuilder) ---")

        builder = ResponseBuilder()
        builder_response = builder.build_summary_response(
            chunk_results, arc, grade, grade_rules, personality,
            verbose=verbose
        )

        # If sarcasm detected at moderate+ confidence, override the assembled response
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            sarcasm_closer = self._choose_unique([
                "I can tell that's not really how you feel.",
                "I hear what you're saying, but I also hear what you're not saying.",
                "The words say fine, but the feeling doesn't.",
                "I'm picking up on the frustration underneath.",
                "You don't have to pretend it's okay.",
            ])
            builder_response = builder_response.rstrip('.!?') + ". " + sarcasm_closer

        # Fallback: if ResponseBuilder produced empty/trivial, use template system
        if not builder_response or builder_response.strip() in ("", "I hear you."):
            if verbose:
                print(f"  (ResponseBuilder produced minimal output, falling back to templates)")
            responses = []
            seen_negative = False
            for i, cr in enumerate(chunk_results):
                response_vadug = compute_harmony(cr['vadug'], personality)
                response_vadug, _ = apply_personality(response_vadug, cr['vadug'], personality)

                is_negative = cr['vadug'].v < 135
                is_reversal = self._is_reversal_chunk(cr['text'])
                is_last = (i == len(chunk_results) - 1)

                chunk_lower = cr['text'].lower()
                negative_content_words = {"sick", "broke", "broken", "died", "lost",
                                           "hurt", "failed", "crash", "fire", "rent",
                                           "raising", "can't take", "much more",
                                           "don't know", "struggle", "pain"}
                has_negative_content = any(w in chunk_lower for w in negative_content_words)
                if has_negative_content and cr['vadug'].v < 155:
                    is_negative = True

                response_text = self._decode_chunk_response(
                    cr, response_vadug, personality,
                    is_first_negative=(is_negative and not seen_negative),
                    is_subsequent_negative=(is_negative and seen_negative),
                    is_reversal=is_reversal,
                    is_last=is_last,
                    grade_rules=grade_rules,
                )

                if is_negative:
                    seen_negative = True

                if response_text:
                    responses.append(response_text)
                    if verbose:
                        print(f"  Chunk {i+1} response (template): \"{response_text}\"")

            if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
                closer = self._choose_unique([
                    "I can tell that's not really how you feel.",
                    "I hear what you're saying, but I also hear what you're not saying.",
                    "The words say fine, but the feeling doesn't.",
                    "I'm picking up on the frustration underneath.",
                    "You don't have to pretend it's okay.",
                ])
            else:
                closer = self.get_arc_closer(arc, chunk_results, grade_rules)

            assembled = self.assemble(responses, closer, arc, chunk_results)
        else:
            assembled = builder_response

        if verbose:
            print(f"\n--- STEP 3: Assembled Response ---")
            print(f"  \"{assembled}\"")

        return assembled, chunk_results, arc

    def analyze_arc(self, chunks: list) -> str:
        """Detect the emotional pattern across chunks.

        Returns one of: descending, ascending, valley, peak,
                        flat_negative, flat_positive, mixed
        """
        if len(chunks) < 2:
            v = chunks[0]['vadug'].v if chunks else 128
            if v < 110:
                return "flat_negative"
            elif v > 148:
                return "flat_positive"
            return "mixed"

        v_values = [c['vadug'].v for c in chunks]
        g_values = [c['vadug'].g for c in chunks]

        # Threshold for "negative" vs "positive"
        # Using 135 instead of 118 because pendulum averaging dilutes
        # negative signals over multi-word chunks
        neg_threshold = 135
        pos_threshold = 148

        # Check flat patterns
        all_negative = all(v < neg_threshold for v in v_values)
        all_positive = all(v > pos_threshold for v in v_values)

        if all_negative:
            # Check if descending
            if self._is_monotonic_decreasing(v_values):
                return "descending"
            return "flat_negative"

        if all_positive:
            if self._is_monotonic_increasing(v_values):
                return "ascending"
            return "flat_positive"

        # Check for CONTRAST: brief positive among negatives
        # Rule 1: positive chunk (V > 140) flanked by negatives (V < 110)
        # Rule 2: single positive chunk among 3+ negative chunks
        contrast = self._detect_contrast(chunks, v_values)
        if contrast:
            return "contrast"

        # Check for valley: dips then rises
        min_idx = v_values.index(min(v_values))
        max_idx = v_values.index(max(v_values))

        # Valley: minimum is in the first half, maximum in second half
        # and there's a significant swing
        v_range = max(v_values) - min(v_values)

        if v_range < 20:
            # Very small range — basically flat
            avg_v = sum(v_values) / len(v_values)
            if avg_v < neg_threshold:
                return "flat_negative"
            elif avg_v > pos_threshold:
                return "flat_positive"
            return "mixed"

        min_val = min(v_values)
        max_val = max(v_values)

        # Valley: starts negative/low, ends positive — check both averages
        # and the final value (which carries the most weight for how the
        # person is FEELING at the end)
        n = len(v_values)
        mid = n // 2
        early_avg = sum(v_values[:mid + 1]) / (mid + 1)
        late_vals = v_values[mid:]
        late_avg = sum(late_vals) / max(1, len(late_vals))
        final_v = v_values[-1]

        if early_avg < neg_threshold and (late_avg > pos_threshold or final_v > pos_threshold):
            return "valley"

        # Peak: starts positive, ends negative/low
        first_v = v_values[0]
        if (early_avg > pos_threshold or first_v > pos_threshold) and (late_avg < neg_threshold or final_v < neg_threshold):
            return "peak"

        # Check monotonic patterns
        if self._is_monotonic_decreasing(v_values):
            return "descending"
        if self._is_monotonic_increasing(v_values):
            return "ascending"

        # Valley with clear dip: minimum is in the first portion, maximum in second
        if (min_idx < max_idx and
            v_values[-1] > v_values[0] + 15 and
            min_val < v_values[-1] - 20):
            return "valley"

        # Peak with clear rise: maximum in first portion, minimum in second
        if (max_idx < min_idx and
            v_values[-1] < v_values[0] - 15 and
            max_val > v_values[-1] + 20):
            return "peak"

        return "mixed"

    def _is_monotonic_decreasing(self, values: list) -> bool:
        """Check if values are generally decreasing (allows small fluctuations)."""
        if len(values) < 2:
            return False
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        return decreases >= len(values) * 0.6

    def _is_monotonic_increasing(self, values: list) -> bool:
        """Check if values are generally increasing (allows small fluctuations)."""
        if len(values) < 2:
            return False
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        return increases >= len(values) * 0.6

    def _detect_contrast(self, chunks: list, v_values: list) -> bool:
        """Detect CONTRAST pattern: brief positive in a negative context.

        Returns True if:
        - A positive chunk (V > 140) is flanked by negative chunks (V < 110), OR
        - A single positive chunk exists among 3+ negative chunks

        When contrast is detected, marks the positive chunk(s) with
        'contrast': True so downstream weighting can invert their contribution.
        """
        n = len(v_values)
        if n < 3:
            return False

        # Pendulum averaging dilutes negative signals — use content-aware
        # thresholds: a chunk is "effectively negative" if V < 135, OR if
        # V < 155 and it contains clearly negative content words.
        neg_content_words = {"lost", "lose", "died", "death", "sick", "broke",
                             "broken", "hurt", "fail", "failed", "fired",
                             "can't sleep", "can't pay", "falling apart",
                             "struggling", "pain", "rent", "crash", "alone",
                             "crying", "miss", "gone", "worst", "terrible",
                             "horrible", "afraid", "scared", "worried"}
        pos_threshold = 140

        def is_effectively_negative(i):
            v = v_values[i]
            if v < 135:
                return True
            if v < 155:
                text_lower = chunks[i]['text'].lower()
                return any(w in text_lower for w in neg_content_words)
            return False

        neg_count = sum(1 for i in range(n) if is_effectively_negative(i))
        pos_indices = [i for i in range(n)
                       if v_values[i] > pos_threshold and not is_effectively_negative(i)]

        if not pos_indices:
            return False

        # Rule 1: positive chunk flanked by negatives on both sides
        for idx in pos_indices:
            if 0 < idx < n - 1:
                if is_effectively_negative(idx - 1) and is_effectively_negative(idx + 1):
                    chunks[idx]['contrast'] = True
                    return True

        # Rule 2: single positive chunk among 3+ negative chunks
        if len(pos_indices) == 1 and neg_count >= 3:
            chunks[pos_indices[0]]['contrast'] = True
            return True

        # Rule 2b: single positive among majority-negative (for shorter sequences)
        if len(pos_indices) == 1 and neg_count >= n - 1:
            chunks[pos_indices[0]]['contrast'] = True
            return True

        return False

    def _is_reversal_chunk(self, text: str) -> bool:
        """Check if a chunk starts with a reversal word (but, however, etc.)."""
        first_word = text.split()[0].lower().strip('.,!?;:') if text.split() else ""
        return first_word in ChunkSplitter.REVERSAL_WORDS

    def _arc_description(self, arc: str, chunks: list) -> str:
        """Human-readable description of the arc."""
        emotions = [c['emotion'] for c in chunks]
        if arc == "contrast":
            # Find the contrast chunk
            contrast_idx = next((i for i, c in enumerate(chunks) if c.get('contrast')), None)
            if contrast_idx is not None:
                bright = emotions[contrast_idx]
                surrounding = [e for i, e in enumerate(emotions) if i != contrast_idx]
                return f"{'|'.join(surrounding)} with brief {bright} (contrast)"
            return " -> ".join(emotions) + " (contrast)"
        elif arc == "valley":
            # Find the pivot point
            v_values = [c['vadug'].v for c in chunks]
            min_idx = v_values.index(min(v_values))
            low_emotions = emotions[:min_idx + 1]
            high_emotions = emotions[min_idx + 1:]
            low = low_emotions[-1] if low_emotions else emotions[0]
            high = high_emotions[-1] if high_emotions else emotions[-1]
            return f"{low} -> reversal -> {high}"
        elif arc == "peak":
            return f"{emotions[0]} -> reversal -> {emotions[-1]}"
        elif arc == "descending":
            return f"{emotions[0]} -> ... -> {emotions[-1]}"
        elif arc == "ascending":
            return f"{emotions[0]} -> ... -> {emotions[-1]}"
        elif arc == "flat_negative":
            return "sustained " + (emotions[0] if emotions else "negative")
        elif arc == "flat_positive":
            return "sustained " + (emotions[0] if emotions else "positive")
        else:
            return " -> ".join(emotions)

    def _decode_chunk_response(self, chunk_result, response_vadug, personality,
                                is_first_negative=False,
                                is_subsequent_negative=False,
                                is_reversal=False,
                                is_last=False,
                                grade_rules=None):
        """Generate a response for a single chunk.

        Uses simplified template logic, FILTERED by grade rules:
        - First negative chunk: full acknowledge + stabilize
        - Subsequent negative: just acknowledge (shorter)
        - Reversal chunk (after "but"): match the new energy
        - Positive chunk: brief celebration or skip

        Grade guardrails:
        - If grade is F-range, lock to presence-only responses
        - If grade blocks "positive_spin", suppress positive celebration
        - If grade blocks "at_least", suppress silver-lining framing
        - If grade blocks "unsolicited_advice", use presence-style stabilize
        """
        v = response_vadug.v
        a = response_vadug.a
        d = response_vadug.d
        u = response_vadug.u
        g = response_vadug.g
        input_v = chunk_result['vadug'].v
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # --- F-range crisis override: presence only ---
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                return self._choose_unique([
                    "I'm here.",
                    "I hear you.",
                    "You're not alone.",
                ])
            else:
                return self._choose_unique([
                    "I hear you. That's real pain.",
                    "I'm here with you.",
                    "You don't have to carry this alone.",
                    "I hear you.",
                ])

        if is_reversal:
            # Match the energy of the new direction
            chunk_lower = chunk_result['text'].lower()
            if input_v > 148:
                # Reversal to positive — but check if positive_spin is blocked
                if "positive_spin" in blocked:
                    return self._choose_unique([
                        "But I hear the shift there.",
                        "But that part sounds different.",
                    ])
                return self._choose_unique([
                    "But a dream job? That's incredible.",
                    "But that's amazing news!",
                    "Now that changes everything.",
                    "But wait — that's actually great.",
                    "Hold on though — that's exciting!",
                ])
            elif any(w in chunk_lower for w in ["honest", "study", "prepare", "admit", "fault"]):
                # Reversal to self-awareness/honesty
                # Check if "at_least" framing is blocked
                if "at_least" in blocked:
                    return self._choose_unique([
                        "But you see it clearly.",
                        "But you know what happened.",
                    ])
                return self._choose_unique([
                    "But hey, at least you're honest about it.",
                    "But you know exactly why.",
                    "But you're being real about it.",
                ])
            else:
                # Reversal to negative
                return self._choose_unique([
                    "But that part is tough.",
                    "Though that's a hard turn.",
                    "But I hear the hard part too.",
                ])

        if is_first_negative:
            # Full acknowledge + stabilize
            # If the content is clearly negative but pendulum diluted it,
            # use content-aware acknowledgment
            chunk_lower = chunk_result['text'].lower()
            if input_v > 130:
                # Pendulum says borderline — use content-aware response
                ack = self._get_content_aware_acknowledge(chunk_lower, input_v)
            else:
                ack = self._get_acknowledge(v, g, input_v, chunk_result['vadug'].g)
            # D- blocks unsolicited_advice — use presence-style stabilize
            if "unsolicited_advice" in blocked:
                stab = self._choose_unique([
                    "I'm right here.",
                    "I'm with you.",
                ])
            else:
                stab = self._get_stabilize(d, a)
            return f"{ack} {stab}".strip()

        if is_subsequent_negative:
            # Just acknowledge, shorter — pass blocked list for filtering
            return self._get_short_acknowledge(input_v, chunk_result['vadug'].g,
                                                chunk_result['text'],
                                                blocked=blocked)

        if input_v > 150:
            # Positive chunk — but if grade blocks positive framing, suppress
            if "positive_spin" in blocked or "ANY_positive_framing" in blocked:
                return ""
            if input_v > 190:
                return self._choose_unique([
                    "That's exciting!",
                    "Love that for you.",
                    "That's the good stuff.",
                ])
            # Mildly positive — might skip entirely to avoid being verbose
            return ""

        # Neutral — skip
        return ""

    def _get_content_aware_acknowledge(self, chunk_lower, input_v):
        """Generate acknowledgment based on content words when pendulum is borderline."""
        if any(w in chunk_lower for w in ["sick", "ill", "health", "hospital"]):
            return self._choose_unique([
                "That's stressful.",
                "That's worrying.",
                "Dealing with sickness is tough.",
            ])
        if any(w in chunk_lower for w in ["broke", "broken", "car", "rent", "money"]):
            return self._choose_unique([
                "That's the last thing you needed.",
                "That's one thing after another.",
                "That kind of stuff piles up fast.",
            ])
        if any(w in chunk_lower for w in ["fail", "failed", "exam", "test"]):
            return self._choose_unique([
                "That stings.",
                "That's disappointing.",
                "That's a tough one.",
            ])
        if any(w in chunk_lower for w in ["don't know", "can't take", "much more"]):
            return self._choose_unique([
                "That's a lot.",
                "I can hear it's piling up.",
                "That's overwhelming.",
            ])
        # Generic content-aware
        return self._choose_unique([
            "That's a lot going on.",
            "I hear you.",
            "That's not easy.",
        ])

    def _get_acknowledge(self, resp_v, resp_g, input_v, input_g):
        """Get an acknowledgment phrase based on input emotional state."""
        if input_v < 60:
            if input_g < 60:
                return self._choose_unique([
                    "That sounds really heavy.",
                    "I can feel the weight of that.",
                ])
            elif input_g > 170:
                return self._choose_unique([
                    "I can feel how fired up you are.",
                    "That's clearly hit a nerve.",
                ])
            else:
                return self._choose_unique([
                    "That sounds really rough.",
                    "I hear you. That's not easy.",
                ])
        elif input_v < 90:
            if input_g < 70:
                return self._choose_unique([
                    "That sounds exhausting.",
                    "That's wearing on you.",
                ])
            else:
                return self._choose_unique([
                    "That's frustrating.",
                    "That's not what you were hoping for.",
                ])
        elif input_v < 135:
            return self._choose_unique([
                "That's a big transition.",
                "That's a lot to process.",
                "I hear you on that.",
                "That's a lot going on.",
            ])
        else:
            return self._choose_unique(["I see.", "Got it."])

    def _get_stabilize(self, resp_d, resp_a):
        """Get a stabilizing phrase."""
        if resp_d < 90:
            return self._choose_unique([
                "I'm right here with you.",
                "You don't have to figure this out alone.",
            ])
        else:
            return self._choose_unique([
                "Let's work through this together.",
                "We can figure this out.",
            ])

    def _get_short_acknowledge(self, input_v, input_g, text, blocked=None):
        """Short acknowledgment for subsequent negative chunks.

        Respects grade guardrails via the blocked list:
        - "at_least" blocked: suppress "At least..." framing
        - "positive_spin" blocked: suppress optimistic reframes
        - "silver_lining" blocked: suppress "bright side" language
        """
        if blocked is None:
            blocked = []
        # Try to reflect the specific content
        text_lower = text.lower()

        if any(w in text_lower for w in ["miss", "leaving", "goodbye", "gone"]):
            return self._choose_unique([
                "I bet they'll miss you too.",
                "Those connections matter.",
                "That kind of bond is real.",
            ])
        if any(w in text_lower for w in ["sick", "ill", "health", "doctor"]):
            return self._choose_unique([
                "That's scary when it's someone you love.",
                "Health stuff hits different.",
            ])
        if any(w in text_lower for w in ["broke", "broken", "money", "rent", "car"]):
            return self._choose_unique([
                "And that on top of everything else.",
                "That's the last thing you needed.",
            ])
        if any(w in text_lower for w in ["fail", "failed", "exam", "test"]):
            return self._choose_unique([
                "That stings.",
                "That's disappointing.",
            ])
        if any(w in text_lower for w in ["study", "studied", "prepare"]):
            # "At least..." framing — blocked by D-range and below
            if "at_least" in blocked:
                return self._choose_unique([
                    "You see it clearly.",
                    "You know what happened.",
                ])
            return self._choose_unique([
                "At least you know what happened.",
                "Honest with yourself — that's a start.",
            ])
        if any(w in text_lower for w in ["deserve", "deserved", "guess"]):
            return self._choose_unique([
                "That's real self-awareness.",
                "You're being honest with yourself.",
            ])
        if any(w in text_lower for w in ["better", "next", "improve"]):
            # Positive spin — blocked by D-range and below
            if "positive_spin" in blocked:
                return self._choose_unique([
                    "I hear you looking ahead.",
                    "One step at a time.",
                ])
            return self._choose_unique([
                "That's the right mindset.",
                "Now that's what I like to hear.",
            ])
        if any(w in text_lower for w in ["much", "more", "take", "handle"]):
            return self._choose_unique([
                "That's a lot stacking up.",
                "One thing after another.",
            ])

        # Generic short acknowledgments
        if input_v < 60:
            return self._choose_unique([
                "And that's not easy either.",
                "That part is rough too.",
            ])
        elif input_v < 90:
            return self._choose_unique([
                "I hear that.",
                "That adds up.",
            ])
        else:
            return self._choose_unique([
                "I hear you on that.",
                "Yeah, that's real.",
            ])

    def get_arc_closer(self, arc: str, chunk_results: list,
                        grade_rules=None) -> str:
        """Select an arc-appropriate closing line, filtered by grade guardrails.

        Grade rules override the arc closer when certain strategies are blocked:
        - F-range: presence-only closers regardless of arc
        - D-range: empathy closers, no silver-lining or positive framing
        - "silver_lining" blocked: filter out optimistic closers
        """
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # F-range: override to presence-only closers
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                return self._choose_unique([
                    "I'm here.",
                    "You're not alone right now.",
                ])
            return self._choose_unique([
                "You're not alone in this.",
                "I'm here. Whatever you need.",
                "I hear you.",
            ])

        # D-range: empathy closers, block any positive/silver-lining arc closers
        if grade in ("D-", "D", "D+"):
            return self._choose_unique([
                "You're not alone in this.",
                "I'm here if you need to talk through it.",
                "That's a lot. I hear you.",
                "I'm here. What do you need right now?",
            ])

        # Normal arc-based closers with filtering
        closers = list(ARC_CLOSERS.get(arc, ARC_CLOSERS["mixed"]))

        # Filter out closers that violate blocked strategies
        if "silver_lining" in blocked or "positive_spin" in blocked:
            # Remove closers with positive/silver-lining language
            positive_words = {"silver lining", "amazing", "clicking", "looking up",
                              "congrats", "incredible", "momentum", "good stuff"}
            closers = [c for c in closers
                       if not any(pw in c.lower() for pw in positive_words)]

        # If all closers got filtered, fall back to neutral
        if not closers:
            closers = [
                "I hear you.",
                "That's a lot to hold.",
                "I'm here for all of it.",
            ]

        return self._choose_unique(closers)

    def assemble(self, responses: list, closer: str, arc: str,
                 chunk_results: list) -> str:
        """Combine chunk responses with transitions and arc closer.

        Between same-polarity chunks: ", and "
        Between opposite-polarity chunks: " But " or " Though "
        Arc closer appended at the end.
        """
        if not responses:
            return closer

        # Filter empty responses
        responses = [r for r in responses if r.strip()]
        if not responses:
            return closer

        # Build assembled text with transitions
        parts = [responses[0]]

        for i in range(1, len(responses)):
            prev_resp = responses[i - 1]
            curr_resp = responses[i]

            # Determine polarity of each response by checking if it starts
            # with reversal-style words
            curr_lower = curr_resp.lower()
            if curr_lower.startswith(("but ", "hold on", "now that", "though ")):
                # Already has a transition word — just add with space
                parts.append(curr_resp)
            elif self._response_is_positive(curr_resp) != self._response_is_positive(prev_resp):
                # Opposite polarity — use contrastive transition
                # Capitalize the response if it starts lowercase
                parts.append(curr_resp)
            else:
                # Same polarity — use additive transition
                joined = self._lowercase_start(curr_resp) if curr_resp else curr_resp
                parts.append(joined)

        # Join parts with appropriate connectors
        assembled = parts[0]
        for i in range(1, len(parts)):
            part = parts[i]
            part_lower = part.lower()
            if part_lower.startswith(("but ", "hold on", "now that", "though ")):
                assembled = self._rstrip_punct(assembled) + ". " + part
            elif self._response_is_positive(part) != self._response_is_positive(assembled.split('.')[-1]):
                assembled = self._rstrip_punct(assembled) + ". " + part
            else:
                joined = self._lowercase_start(part)
                assembled = self._rstrip_punct(assembled) + ", and " + joined

        # Append closer
        assembled = self._rstrip_punct(assembled) + ". " + closer

        return assembled

    def _rstrip_punct(self, text: str) -> str:
        """Strip trailing sentence punctuation (. ! ?) from text."""
        return text.rstrip('.!?')

    def _lowercase_start(self, text: str) -> str:
        """Lowercase the first character, but NOT if it's 'I' standing alone."""
        if not text:
            return text
        # Don't lowercase "I" when it starts a sentence as a pronoun
        if text[0] == 'I' and (len(text) == 1 or not text[1].isalpha()):
            return text
        return text[0].lower() + text[1:]

    def _response_is_positive(self, text: str) -> bool:
        """Rough check if a response text is positive in tone."""
        positive_words = {"incredible", "amazing", "great", "exciting", "love",
                          "wonderful", "good", "awesome", "fantastic", "congrats",
                          "dream", "momentum", "clicking", "looking up"}
        text_lower = text.lower()
        return any(w in text_lower for w in positive_words)


# =============================================================
# MAIN: Interactive Pipeline
# =============================================================


def run_pipeline(text: str, personality: PersonalityVector,
                 verbose: bool = True, show_trace: bool = True) -> str:
    """Run the full Clanker pipeline on input text.

    If the input has multiple emotional chunks (2+), routes to ChunkedPipeline
    for paragraph-level arc detection. Single chunks use the original pipeline.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"INPUT: \"{text}\"")
        print(f"{'='*60}")

    # Detect multi-chunk input
    splitter = ChunkSplitter()
    chunks = splitter.split(text)

    if len(chunks) >= 2:
        # Multi-chunk: use the chunked pipeline for arc-aware response
        pipeline = ChunkedPipeline()
        response, chunk_results, arc = pipeline.process(
            text, personality, verbose=verbose, show_trace=show_trace
        )
        if verbose:
            print(f"\n{'='*60}")
        return response

    # Single chunk: use original pipeline
    # Step 1: Sequential Pendulum
    pend = SequentialPendulum()
    input_vadu, history = pend.process_text(text)

    if verbose:
        print(f"\n--- STEP 1: Sequential Pendulum ---")
        if show_trace:
            print(pend.render_trace())
        print(f"\n  Pendulum settles at: {input_vadu}")
        print(f"  Reads as: {input_vadu.describe()}")
        user_emotion = nearest_emotion(input_vadu)
        print(f"  Nearest emotion: {user_emotion}")

    # Step 1.1: Tonal analysis — detect sarcasm from trajectory pattern
    tonal = TonalAnalyzer()
    intent_det = IntentDetector()
    intent_mode, _intent_conf = intent_det.detect(text)
    tone_result = tonal.analyze(history, intent_mode=intent_mode)
    tonal.display(tone_result, verbose=verbose)

    # Apply tonal V adjustment (flip positive reads on sarcastic non-emotional content)
    old_v = input_vadu.v
    new_v, new_a, new_d, new_u, new_g = apply_tonal_adjustment(
        input_vadu.v, input_vadu.a, input_vadu.d, input_vadu.u, input_vadu.g,
        tone_result, intent_mode,
    )
    if new_v != old_v:
        input_vadu = VADUG(v=new_v, a=new_a, d=new_d, u=new_u, g=new_g)
        if verbose:
            print(f"  Tonal V adjustment: {old_v} -> {new_v} (sarcasm flip)")

    # Step 1.5: Sentence grade
    grader = SentenceGrader()
    single_chunk = [{'vadug': input_vadu, 'history': history, 'text': text}]
    grade, grade_rules = grader.compute_grade(single_chunk)
    grader.display(grade, grade_rules, verbose=verbose)

    # Step 1.6: Sarcasm detection
    sarcasm = SarcasmDetector()
    sarcasm_flag = False
    is_sarcastic, sarcasm_confidence, sarcasm_signals = sarcasm.analyze_trajectory(history)

    grade_note = ""
    if is_sarcastic and sarcasm_confidence >= SarcasmDetector.LOW:
        sarcasm_flag = True
        grade, grade_note = sarcasm.adjust_grade(grade, sarcasm_confidence, grader)
        # Recompute rules with adjusted grade
        stats = grade_rules.get("stats", {})
        grade_rules = grader._get_rules(grade, stats.get("spread", 0), stats.get("trend", 0))
        grade_rules["stats"] = stats

    sarcasm.display(
        sarcasm_flag, sarcasm_confidence, sarcasm_signals,
        grade_note=grade_note, verbose=verbose
    )

    # Step 2: Metadata
    header = classify_metadata(text, input_vadu)
    if verbose:
        print(f"\n--- STEP 2: Metadata Header ---")
        print(f"  {header}")
        print(f"  9 bytes: {header.to_bytes().hex()}")

    # Step 3: Harmony
    response_vadu = compute_harmony(input_vadu, personality)
    if verbose:
        print(f"\n--- STEP 3: VADUG Harmony Response ---")
        print(f"  Input:    {input_vadu}")
        print(f"  Response: {response_vadu}")
        print(f"  Reads as: {response_vadu.describe()}")
        resp_emotion = nearest_emotion(response_vadu)
        print(f"  Nearest emotion: {resp_emotion}")

    # Step 4: Personality
    response_vadu, p_notes = apply_personality(response_vadu, input_vadu, personality)
    if verbose:
        print(f"\n--- STEP 4: Personality Filter ---")
        print(f"  Vector: {personality}")
        if p_notes:
            for n in p_notes:
                print(f"  {n}")
        else:
            print(f"  No personality overrides triggered")
        print(f"  Final VADUG: {response_vadu}")

    # Step 5: Generate Clanker + Encoding
    clanker_lines, encoding_lines = generate_clanker(text, header, response_vadu)
    if verbose:
        print(f"\n--- STEP 5: Clanker Encoding ---")
        for line in encoding_lines:
            print(f"  {line}")
        print()
        print(f"  Opcodes (human-readable):")
        for line in clanker_lines:
            print(f"    {line}")

    # Step 6: Decode via ResponseBuilder (math-based word selection)
    # Use summary mode even for single chunks to respect G-based brevity
    builder = ResponseBuilder()
    single_chunk_result = [{'vadug': input_vadu, 'text': text}]
    response = builder.build_summary_response(
        single_chunk_result, "flat_negative" if input_vadu.v < 135 else "flat_positive",
        grade, grade_rules, personality,
        verbose=verbose,
    )

    # Fallback to template system if ResponseBuilder produced empty/trivial
    if not response or response.strip() == "":
        response = decode_response(text, input_vadu, response_vadu, header.goal)

    # If sarcasm detected at moderate+ confidence, override response to address
    # the REAL emotion, not the surface positivity
    if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
        sarcasm_responses = [
            "I can tell that's not really how you feel.",
            "I hear what you're saying, but I also hear what you're not saying.",
            "The words say fine, but the feeling doesn't.",
            "I'm picking up on the frustration underneath.",
            "You don't have to pretend it's okay.",
        ]
        response = random.choice(sarcasm_responses)

    # Step 6.5: Grade guardrail — override response if grade demands it
    grade_override = None
    blocked = grade_rules.get("blocked", [])
    if grade in ("F-", "F", "F+"):
        # Crisis territory: presence only
        if grade == "F-":
            grade_override = random.choice([
                "I'm here.",
                "I hear you.",
                "You're not alone.",
            ])
        else:
            grade_override = random.choice([
                "I hear you. That's real pain.",
                "I'm here with you.",
                "You don't have to carry this alone.",
            ])
    elif grade in ("D-", "D", "D+"):
        # Check if response contains blocked strategies
        response_lower = response.lower()
        has_blocked_content = False
        if "silver_lining" in blocked or "positive_spin" in blocked:
            positive_markers = ["bright side", "at least", "silver lining",
                                "could be worse", "cheer up", "look on the",
                                "everything happens"]
            if any(m in response_lower for m in positive_markers):
                has_blocked_content = True
        if has_blocked_content:
            grade_override = random.choice([
                "I hear you. That's not easy.",
                "That sounds really heavy. I'm here.",
                "I'm sorry you're going through that.",
            ])

    if grade_override:
        response = grade_override

    if verbose:
        print(f"\n--- STEP 6: Decoded Response (ResponseBuilder) ---")
        if sarcasm_flag and sarcasm_confidence >= SarcasmDetector.MODERATE:
            print(f"  (sarcasm override — addressing real emotion)")
        if grade_override:
            print(f"  (grade {grade} guardrail — locked to {grade_rules.get('tone', '?')})")
        print(f"  \"{response}\"")
        print(f"\n{'='*60}")

    return response
