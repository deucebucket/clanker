"""Response generation for the Clanker pipeline."""

import re
import math
import random

from .shared import VADUG, VADU, MetadataHeader, PersonalityVector
from .forces_curated import EMOTIONAL_VOCABULARY as WORD_FORCES  # V2 vocab

# =============================================================
# STEP 2: Metadata Classification
# =============================================================

def classify_metadata(text: str, vadu: VADU) -> MetadataHeader:
    """Classify the message metadata: CERT, SRC, GOAL, REL."""
    text_lower = text.lower()

    # Determine GOAL from keywords
    goal = 0x00  # HELP default
    if "?" in text:
        goal = 0x01  # CLARIFY
    if any(w in text_lower for w in ["explain", "why", "how does", "what is", "teach"]):
        goal = 0x03  # TEACH
    if any(w in text_lower for w in ["help", "fix", "solve", "build", "create", "make"]):
        goal = 0x00  # HELP
    if any(w in text_lower for w in ["do it", "run", "execute", "deploy", "send"]):
        goal = 0x04  # EXECUTE
    if vadu.v < 80 and vadu.d < 90:
        goal = 0x06  # EMPATHIZE (negative + low agency = needs support)

    # CERT: user input is their truth
    cert = 180

    # SRC: it's from the user
    src = 0x04  # USER

    # REL: direct conversation is always relevant
    rel = 220

    return MetadataHeader(vadu=vadu, cert=cert, src=src, goal=goal, rel=rel)


# =============================================================
# STEP 3: VADUG Harmony -- Compute Response Emotional State
# =============================================================

def compute_harmony(input_vadug: VADUG, personality: PersonalityVector) -> VADUG:
    """Mathematically derive the response VADUG from input VADUG + personality.

    Rules:
    - Valence: nudge toward positive, SCALED by agreeableness
    - Arousal: ASSERTIVENESS controls response energy
    - Dominance: stabilize but assertiveness scales the boost
    - Urgency: CURIOSITY reduces urgency (explorative, not rushed)
    - Gravity: PLAYFULNESS lifts gravity
    """
    # Agreeableness controls empathy strength (0.1 to 0.6 range, was capped at 0.3)
    empathy_factor = personality.agreeableness / 255.0 * 0.6

    # Valence: nudge toward positive, SCALED by agreeableness
    v_nudge = (145 - input_vadug.v) * empathy_factor
    response_v = int(input_vadug.v + v_nudge)

    # Arousal: ASSERTIVENESS controls response energy
    # High assertive = match/amplify arousal. Low assertive = calm down.
    assert_factor = personality.assertiveness / 255.0
    if assert_factor > 0.6:
        # Match energy, don't dampen
        a_diff = 0  # stay near input arousal
    else:
        # Pull toward calm
        a_diff = (128 - input_vadug.a) * (1.0 - assert_factor) * 0.5
    response_a = int(input_vadug.a + a_diff)

    # Dominance: still stabilize but LESS aggressively
    # Assertiveness scales how much dominance boost to apply
    stability_target = 140 + personality.assertiveness / 255.0 * 40  # 140-180 range
    stability_boost = max(0, stability_target - input_vadug.d) * 0.4  # was 0.6
    response_d = int(input_vadug.d + stability_boost)

    # Urgency: CURIOSITY reduces urgency (curious = explorative, not rushed)
    curiosity_factor = personality.curiosity / 255.0
    urgency_damping = 0.4 + (1.0 - curiosity_factor) * 0.4  # 0.4-0.8 range
    response_u = int(input_vadug.u * urgency_damping)

    # Gravity: PLAYFULNESS lifts gravity
    play_factor = personality.playfulness / 255.0
    if input_vadug.g < 80:
        # Sinking — lift amount scaled by playfulness
        lift = (128 - input_vadug.g) * (0.2 + play_factor * 0.3)  # 0.2-0.5 range
        response_g = int(input_vadug.g + lift)
    elif input_vadug.g > 180:
        response_g = input_vadug.g
    else:
        # Playfulness pulls gravity up slightly
        play_lift = play_factor * 20  # 0-20 points
        response_g = int(128 + (input_vadug.g - 128) * 0.5 + play_lift)

    return VADUG(
        v=max(0, min(255, response_v)),
        a=max(0, min(255, response_a)),
        d=max(0, min(255, response_d)),
        u=max(0, min(255, response_u)),
        g=max(0, min(255, response_g))
    )

# =============================================================
# STEP 5: Generate Clanker Output + Byte Encoding
# =============================================================

# Nearest emotion word from VADU coordinates
EMOTION_MAP = [
    (20, 230, "enraged"), (30, 200, "furious"), (40, 170, "angry"),
    (50, 150, "frustrated"), (60, 130, "irritated"), (70, 100, "annoyed"),
    (35, 70, "sad"), (25, 50, "melancholic"), (15, 40, "despairing"),
    (45, 40, "gloomy"), (80, 60, "meh"), (90, 40, "bored"),
    (30, 190, "panicked"), (40, 180, "anxious"), (50, 160, "stressed"),
    (60, 80, "disappointed"), (70, 60, "down"),
    (128, 128, "neutral"), (128, 80, "calm"), (128, 40, "serene"),
    (150, 100, "okay"), (160, 120, "pleased"), (170, 140, "happy"),
    (180, 160, "glad"), (190, 130, "cheerful"), (200, 150, "joyful"),
    (210, 170, "excited"), (220, 180, "thrilled"), (240, 200, "ecstatic"),
    (200, 60, "content"), (190, 40, "peaceful"), (210, 80, "satisfied"),
    (160, 180, "amazed"), (150, 200, "startled"),
]


def nearest_emotion(vadu: VADU) -> str:
    """Find the nearest named emotion to the VADU coordinates."""
    best_word = "neutral"
    best_dist = float("inf")
    for v_c, a_c, word in EMOTION_MAP:
        dist = math.sqrt((vadu.v - v_c) ** 2 + (vadu.a - a_c) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_word = word
    return best_word


def generate_clanker(input_text: str, input_header: MetadataHeader,
                     response_vadu: VADU) -> tuple:
    """Generate Clanker opcodes and byte encoding for the response.

    Returns (opcode_lines, encoding_lines).
    """
    opcode_lines = []
    goal = input_header.goal
    goal_name = MetadataHeader.GOAL_NAMES.get(goal, "HELP")

    # Determine context string for empathize opcode
    emotion = nearest_emotion(input_header.vadu)
    context_str = ""

    # Emotional acknowledgment if user is negative
    if input_header.vadu.v < 90:
        context_str = f"user feels {emotion}"
        opcode_lines.append(f"06 SOCIAL intent [empathize] context=\"{context_str}\"")

    # Goal-based response opcode
    if goal == 0x06:  # EMPATHIZE
        if not context_str:
            # Only add standalone empathize if we didn't already add one above
            opcode_lines.append(f"06 SOCIAL intent [empathize]")
        opcode_lines.append(f"  THINK [premise=\"acknowledge feelings first\"] CERT200 SRC_INFERRED")
    elif goal == 0x00:  # HELP
        opcode_lines.append(f"THINK [premise=\"user needs help\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_HELP")
    elif goal == 0x01:  # CLARIFY
        opcode_lines.append(f"THINK [premise=\"need more information\"] CERT120 SRC_INFERRED")
        opcode_lines.append(f"  GOAL_CLARIFY")
    elif goal == 0x03:  # TEACH
        opcode_lines.append(f"THINK [premise=\"user wants to understand\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_TEACH")
    elif goal == 0x04:  # EXECUTE
        opcode_lines.append(f"THINK [premise=\"user wants action taken\"] CERT{input_header.cert} SRC_USER")
        opcode_lines.append(f"  GOAL_EXECUTE")

    # Attach response VADUG
    opcode_lines.append(f"  [{response_vadu}]")
    opcode_lines.append(f"ANSWER [ready] CERT180 SRC_INFERRED")

    # --- Byte encoding display ---
    encoding_lines = []
    header_bytes = input_header.to_bytes()
    hx = header_bytes.hex().upper()
    # Format as pairs
    header_hex = " ".join(hx[i:i+2] for i in range(0, len(hx), 2))

    vadug = input_header.vadu
    encoding_lines.append(f"Header: [{header_hex[:14]}] [{header_hex[15:]}]")
    encoding_lines.append(
        f"        V{vadug.v} A{vadug.a} D{vadug.d} U{vadug.u} G{vadug.g}  "
        f"CERT{input_header.cert} SRC_{MetadataHeader.SRC_NAMES.get(input_header.src, '?')} "
        f"GOAL_{goal_name} REL{input_header.rel}"
    )

    encoding_lines.append("")
    encoding_lines.append("Opcodes:")
    # Approximate opcode encoding
    opcode_count = 0
    for line in opcode_lines:
        stripped = line.strip()
        if stripped.startswith("06") or stripped.startswith("THINK") or stripped.startswith("ANSWER"):
            opcode_count += 1
            # Extract the hex/mnemonic
            if stripped.startswith("06"):
                if context_str:
                    encoding_lines.append(f"  06 00 [empathize] ctx=\"{context_str}\"")
                else:
                    encoding_lines.append(f"  06 00 [empathize]")
            elif stripped.startswith("THINK"):
                # Extract premise
                m = re.search(r'premise="([^"]*)"', stripped)
                premise = m.group(1) if m else "unknown"
                m2 = re.search(r'CERT(\d+)', stripped)
                cert_val = m2.group(1) if m2 else "180"
                encoding_lines.append(f"  20 [premise=\"{premise}\"] CERT{cert_val}")
            elif stripped.startswith("ANSWER"):
                encoding_lines.append(f"  24 [response] CERT180")

    # Token count comparison
    input_words = re.findall(r"[a-z']+", input_text.lower())
    n_input_tokens = len(input_words)
    header_bytes_count = 9
    opcode_bytes = opcode_count * 4  # ~4 bytes per opcode on average
    total_clanker_bytes = header_bytes_count + opcode_bytes
    clanker_tokens = max(1, total_clanker_bytes // 4)  # ~4 bytes per Clanker token

    encoding_lines.append("")
    encoding_lines.append("Token count comparison:")
    encoding_lines.append(f"  English input:  \"{input_text}\" = {n_input_tokens} tokens")
    encoding_lines.append(f"  Clanker encoding: [{header_bytes_count} header (VADUG+meta)] + [{opcode_count} opcodes x ~4 bytes] = {total_clanker_bytes} bytes (~{clanker_tokens} Clanker tokens)")
    if n_input_tokens > 0:
        savings = max(0, int((1.0 - clanker_tokens / n_input_tokens) * 100))
        encoding_lines.append(f"  Compression: {savings}% fewer tokens")

    return opcode_lines, encoding_lines


# =============================================================
# STEP 6: Decode -- VADUG -> English Response Framing
# =============================================================

def decode_response(input_text: str, input_vadu: VADU, response_vadu: VADU,
                    goal: int) -> str:
    """Generate a natural English response based on response VADU and goal."""
    user_emotion = nearest_emotion(input_vadu)
    response_emotion = nearest_emotion(response_vadu)

    openers = {
        "enraged": "I can see you're really angry right now.",
        "furious": "I understand you're furious.",
        "angry": "I hear your frustration.",
        "frustrated": "That sounds really frustrating.",
        "irritated": "I get that's annoying.",
        "annoyed": "I understand that's bothersome.",
        "sad": "I'm sorry you're going through that.",
        "melancholic": "That sounds tough.",
        "despairing": "I hear you. That sounds really heavy.",
        "gloomy": "That doesn't sound easy.",
        "panicked": "Let's take a breath. We'll work through this.",
        "anxious": "I understand you're worried. Let's figure this out.",
        "stressed": "I can tell this is weighing on you.",
        "disappointed": "I understand that's disappointing.",
        "down": "I hear you.",
        "neutral": "",
        "calm": "",
        "serene": "",
        "okay": "",
        "pleased": "Glad to hear that!",
        "happy": "That's great!",
        "glad": "Nice!",
        "cheerful": "Love the energy!",
        "joyful": "That's wonderful!",
        "excited": "That's awesome!",
        "thrilled": "That's amazing!",
        "ecstatic": "Incredible!",
        "content": "",
        "peaceful": "",
        "satisfied": "Good to hear.",
        "amazed": "Wow!",
        "startled": "Whoa!",
    }

    opener = openers.get(user_emotion, "")

    bodies = {
        0x00: "Let me help with that.",
        0x01: "Could you tell me a bit more?",
        0x02: "I want to flag something for you.",
        0x03: "Let me explain.",
        0x04: "On it.",
        0x05: "I can't do that, but here's why.",
        0x06: "I'm here. Take your time.",
        0x07: "Just to make sure I understand...",
        0x08: "Let's think through this together.",
    }

    body = bodies.get(goal, "How can I help?")

    if input_vadu.u > 200:
        body = "I'm on this RIGHT NOW. " + body
    elif input_vadu.u > 150:
        body = "I hear the urgency. " + body

    parts = [p for p in [opener, body] if p]
    return " ".join(parts)


# =============================================================
# STEP 6.5: ResponseBuilder — Math-Based Word Selection
# =============================================================

class ResponseBuilder:
    """Builds responses by selecting words from WORD_FORCES based on
    VADUG coordinate distance. Every word is chosen by math, not templates.

    Rules:
    1. No word used twice in a response
    2. No phrase structure repeated across chunks
    3. Track recently used words across conversation
    4. Weighted dimensions -- valence and gravity matter more
    5. POS-aware word tagging for better sentence construction
    6. Diversity penalty -- avoid stem-similar words in same response
    7. Emotional trajectory -- blend from input toward response VADUG
    """

    # Dimension weights for distance calculation.
    # Valence matters most, gravity is critical for emotional weight,
    # arousal is important for tone, urgency is situational.
    DIMENSION_WEIGHTS = {
        'v': 3.0,   # valence matters most
        'a': 1.5,   # arousal is important for tone matching
        'd': 1.0,   # dominance
        'u': 0.5,   # urgency is situational
        'g': 2.0,   # gravity is critical for emotional weight
    }

    # Curated adjective sets -- words that work in "That sounds X" / "That's really X"
    # These are all present in WORD_FORCES and selected for natural sentence fit
    NEGATIVE_ADJECTIVES = {
        # These all work as situation descriptors: "That sounds X" / "That's really X"
        "tough", "hard", "heavy", "rough", "painful", "awful", "terrible", "horrible",
        "devastating", "exhausting", "overwhelming", "difficult", "sad", "lonely",
        "scary", "stressful", "heartbreaking", "harsh", "brutal", "draining",
        "crushing", "intense", "deep", "frustrating", "agonizing", "dreadful",
        "miserable", "wretched", "hopeless", "helpless", "broken",
        "uncomfortable", "uneasy", "stuck", "lost",
        "sick", "numb", "unfair", "unbearable",
        "meaningless", "pointless", "useless", "bleak", "grim", "somber",
        "melancholy", "bittersweet", "isolating", "toxic", "chaotic",
        "disappointing", "embarrassing", "terrifying", "frightening",
        "worrying", "exhausting", "tiring", "draining",
    }

    POSITIVE_ADJECTIVES = {
        "amazing", "incredible", "wonderful", "fantastic", "beautiful", "brilliant",
        "excellent", "awesome", "magnificent", "spectacular", "phenomenal", "outstanding",
        "remarkable", "extraordinary", "superb", "marvelous", "glorious", "blessed",
        "grateful", "thankful", "proud", "accomplished", "triumphant", "victorious",
        "successful", "thriving", "flourishing", "radiant", "gorgeous", "stunning",
        "ecstatic", "thrilled", "excited", "joyful", "happy", "delicious",
        "inspiring", "vibrant", "colorful", "bright", "warm", "gentle", "peaceful",
        "hopeful", "powerful", "resilient", "strong", "creative", "bold", "brave",
    }

    NOUN_FEELINGS = {
        "grief", "sorrow", "pain", "loss", "anguish", "despair", "heartbreak",
        "struggle", "weight", "darkness", "sadness", "loneliness", "fear",
        "frustration", "anger", "regret", "guilt", "shame", "exhaustion",
        "confusion", "chaos", "stress", "worry", "doubt", "torment",
        "suffering", "burden", "tragedy", "nightmare", "betrayal", "rejection",
        "joy", "love", "pride", "hope", "strength", "courage", "resilience",
        "warmth", "comfort", "peace", "harmony", "triumph", "victory",
        "excitement", "wonder", "gratitude", "relief", "freedom", "growth",
    }

    def __init__(self):
        self.used_words = set()
        self.used_structures = set()
        self.session_history = {}
        self._categorize_words()

    def _categorize_words(self):
        """Categorize WORD_FORCES entries by likely sentence role based on VADUG."""
        self.ACKNOWLEDGE_WORDS = set()
        self.STABILIZE_WORDS = set()
        self.EMOTION_WORDS = set()
        self.ACTION_WORDS = set()
        self.neg_adj_pool = {}
        self.pos_adj_pool = {}
        self.noun_pool = {}

        for word, forces in WORD_FORCES.items():
            v, a, d, u, g = forces
            if v < -10:
                self.ACKNOWLEDGE_WORDS.add(word)
            if d > 15:
                self.STABILIZE_WORDS.add(word)
            if -30 < v < 30 and abs(a) < 30:
                self.EMOTION_WORDS.add(word)
            if v > 0 and u > 5:
                self.ACTION_WORDS.add(word)

            # Build curated pools from WORD_FORCES intersection with our adjective sets
            if word in self.NEGATIVE_ADJECTIVES:
                self.neg_adj_pool[word] = forces
            if word in self.POSITIVE_ADJECTIVES:
                self.pos_adj_pool[word] = forces
            if word in self.NOUN_FEELINGS:
                self.noun_pool[word] = forces

        # Build POS-tagged index for all WORD_FORCES entries
        self.pos_index = {}
        for word in WORD_FORCES:
            pos = self._guess_pos(word)
            self.pos_index.setdefault(pos, set()).add(word)

    @staticmethod
    def _guess_pos(word):
        """Guess part-of-speech from word suffix. Used for sentence construction."""
        if word.endswith('ly'):
            return 'adverb'
        if word.endswith(('ness', 'ment', 'tion', 'sion')):
            return 'noun'
        if word.endswith('ing'):
            return 'gerund'
        if word.endswith('ed'):
            return 'past'
        if word.endswith(('ful', 'ous', 'ive', 'able', 'ible')):
            return 'adjective'
        if word.endswith(('er', 'or')):
            return 'agent_noun'
        return 'unknown'

    @staticmethod
    def _similarity_penalty(candidate, selected_words):
        """Penalize words that share stems with already selected words.

        If 'happy' is selected, 'happily' and 'happiness' get penalized
        to avoid monotone responses.
        """
        penalty = 0.0
        for sel in selected_words:
            # Shared prefix of 4+ chars = probably same stem
            shared = 0
            for a, b in zip(candidate, sel):
                if a == b:
                    shared += 1
                else:
                    break
            if shared >= 4:
                penalty += 0.5
        return penalty

    def _get_trajectory_target(self, input_vadug, response_vadug, position, total):
        """Blend between input and response VADUG based on position in response.

        First words: closer to input VADUG (acknowledge where user is)
        Middle words: transition
        Last words: at response VADUG target (where we're guiding them)
        """
        t = position / max(total - 1, 1)
        return VADUG(
            v=int(input_vadug.v + (response_vadug.v - input_vadug.v) * t),
            a=int(input_vadug.a + (response_vadug.a - input_vadug.a) * t),
            d=int(input_vadug.d + (response_vadug.d - input_vadug.d) * t),
            u=int(input_vadug.u + (response_vadug.u - input_vadug.u) * t),
            g=int(input_vadug.g + (response_vadug.g - input_vadug.g) * t),
        )

    def find_closest_words(self, target_v, target_a, target_d, target_u, target_g,
                           word_pool=None, n=30, pos_filter=None):
        """Find words whose VADUG forces are closest to the target coordinates.

        Target values are in 0-255 scale. WORD_FORCES are in force scale (-80 to +80).
        Convert target to force scale: force = target - 128 (for V,A,D,G), force = target (for U).

        Uses DIMENSION_WEIGHTS for weighted distance calculation and applies
        a diversity penalty against already-selected words to avoid monotone output.

        Args:
            pos_filter: optional POS tag string to restrict results (e.g. 'adjective', 'noun')
        """
        target_vf = target_v - 128
        target_af = target_a - 128
        target_df = target_d - 128
        target_uf = target_u
        target_gf = target_g - 128

        w = self.DIMENSION_WEIGHTS
        pool = word_pool if word_pool else WORD_FORCES
        candidates = []

        for word, forces in pool.items():
            if word in self.used_words:
                continue
            if self.session_history.get(word, 0) > 2:
                continue
            if len(word) < 3:
                continue
            if pos_filter is not None and self._guess_pos(word) != pos_filter:
                continue

            v, a, d, u, g = forces

            distance = math.sqrt(
                w['v'] * (v - target_vf)**2 +
                w['a'] * (a - target_af)**2 +
                w['d'] * (d - target_df)**2 +
                w['u'] * (u - target_uf)**2 +
                w['g'] * (g - target_gf)**2
            )

            # Diversity penalty: penalize words sharing stems with already-selected words
            diversity_penalty = self._similarity_penalty(word, self.used_words)
            distance += diversity_penalty * 20.0  # Scale penalty to be meaningful vs distance

            candidates.append((distance, word, v, a, d, u, g))

        return sorted(candidates)[:n]

    def select_word(self, target_v, target_a, target_d, target_u, target_g,
                    word_pool=None):
        """Select the best unused word matching the target VADUG."""
        candidates = self.find_closest_words(target_v, target_a, target_d, target_u, target_g, word_pool)

        if candidates:
            _, word, _, _, _, _, _ = candidates[0]
            self.used_words.add(word)
            self.session_history[word] = self.session_history.get(word, 0) + 1
            return word
        return None

    def _pick_structure(self, structures):
        """Pick the first structure whose opening pattern hasn't been used.

        Uses first 2 words as pattern key to prevent 'That sounds X' and
        'That sounds Y' from both being selected across chunks.
        """
        for s in structures:
            if s is None:
                continue
            words = s.split()
            pattern_key = ' '.join(words[0:2]) if len(words) >= 2 else s
            if pattern_key not in self.used_structures:
                self.used_structures.add(pattern_key)
                return s
        # Fallback: return the first non-None
        for s in structures:
            if s is not None:
                return s
        return None

    def build_acknowledge(self, input_vadug, response_vadug):
        """Build an acknowledgment phrase from WORD_FORCES.

        Selects both an adjective (for 'That sounds X') and a noun (for 'the X
        in that') closest to the input VADUG, then uses the best fitting template.
        """
        # Select an adjective from curated negative adjective pool
        adj_pool = {w: f for w, f in self.neg_adj_pool.items() if w not in self.used_words}
        adj = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            adj_pool if adj_pool else None
        )

        # Select a noun from curated feeling-noun pool
        noun_pool = {w: f for w, f in self.noun_pool.items()
                     if w not in self.used_words and WORD_FORCES.get(w, (0,))[0] < 0}
        noun = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            noun_pool if noun_pool else None
        )

        structures = []
        if adj:
            structures.extend([
                f"That sounds {adj}.",
                f"That's really {adj}.",
                f"That's genuinely {adj}.",
            ])
        if noun:
            structures.extend([
                f"I can feel the {noun} in that.",
                f"I hear the {noun}.",
                f"That level of {noun} is real.",
            ])
        if adj:
            structures.append(f"{adj.capitalize()} is the right word for it.")
        structures.append("I hear you.")

        result = self._pick_structure(structures)
        return result if result else "I hear you."

    def build_stabilize(self, response_vadug):
        """Build a stabilizing phrase from WORD_FORCES.

        Picks words that project the RESPONSE vadug (stability, presence).
        """
        stab_pool = {w: WORD_FORCES[w] for w in self.STABILIZE_WORDS
                     if w in WORD_FORCES and w not in self.used_words}
        anchor = self.select_word(
            response_vadug.v, response_vadug.a, response_vadug.d, response_vadug.u, response_vadug.g,
            stab_pool if stab_pool else None
        )

        structures = []
        if anchor:
            if anchor in ("here", "near", "close"):
                structures.append(f"I'm right {anchor} with you.")
            elif anchor in ("strength", "courage", "resilience", "resolve", "grit",
                            "comfort", "warmth"):
                structures.append(f"You've got {anchor} in you.")
            elif anchor in ("handle", "figure", "manage", "work", "face", "tackle",
                            "navigate", "endure"):
                structures.append(f"We'll {anchor} this together.")
            elif anchor in ("together", "beside", "alongside"):
                structures.append(f"We're in this {anchor}.")
            elif anchor in ("strong", "powerful", "resilient", "brave", "capable"):
                structures.append(f"You're stronger than this feels.")
            else:
                structures.append(f"We'll get through this {anchor}.")

        structures.extend([
            "I'm not going anywhere.",
            "You don't have to carry this alone.",
            "Take whatever time you need.",
            "I'm right here.",
        ])

        result = self._pick_structure(structures)
        return result if result else "I'm here."

    # Verbs that work in "Let's X" / "What would X" redirect templates
    REDIRECT_VERBS = {
        "look", "start", "begin", "explore", "try", "work", "figure", "think",
        "plan", "talk", "focus", "tackle", "check", "sort", "help", "fix",
        "learn", "study", "practice", "build", "create", "make", "find",
    }

    def build_redirect(self, response_vadug, grade_rules=None):
        """Build a redirect/next-step phrase. Respects grade rules."""
        if grade_rules and any(b in (grade_rules.get('blocked', []))
                               for b in ['advice', 'redirect', 'problem_solving']):
            return None

        # Only select from known verbs that work in redirect templates
        verb_pool = {w: WORD_FORCES[w] for w in self.REDIRECT_VERBS
                     if w in WORD_FORCES and w not in self.used_words}
        action = self.select_word(
            response_vadug.v, response_vadug.a, response_vadug.d, response_vadug.u, response_vadug.g,
            verb_pool if verb_pool else None
        )

        structures = []
        if action:
            if action in ("look", "start", "begin", "explore", "check"):
                structures.append(f"Let's {action} at this fresh.")
            elif action in ("try", "work", "figure", "think", "plan", "sort",
                            "talk", "focus"):
                structures.append(f"Let's {action} through it.")
            elif action in ("help", "fix", "tackle", "find"):
                structures.append(f"Let's {action} this.")
            else:
                structures.append(f"Let's {action} from here.")

        structures.extend([
            "What do you need from me right now?",
            "Where do you want to start?",
        ])

        result = self._pick_structure(structures)
        return result

    def build_positive_acknowledge(self, input_vadug, response_vadug):
        """Build a positive acknowledgment for happy/excited input."""
        adj_pool = {w: f for w, f in self.pos_adj_pool.items() if w not in self.used_words}
        word = self.select_word(
            input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
            adj_pool if adj_pool else None
        )

        if not word:
            return "That's great to hear."

        structures = [
            f"That's genuinely {word}.",
            f"That sounds {word}!",
            f"That's really {word}.",
            f"I can feel the {word} in that.",
        ]

        result = self._pick_structure(structures)
        return result if result else "That's great to hear."

    # Adjectives that work in "that's genuinely X" / "that part is X" (situation descriptors)
    SITUATION_POS_ADJECTIVES = {
        "amazing", "incredible", "wonderful", "fantastic", "beautiful", "brilliant",
        "excellent", "awesome", "magnificent", "spectacular", "phenomenal", "outstanding",
        "remarkable", "extraordinary", "superb", "marvelous", "glorious", "exciting",
        "inspiring", "vibrant", "stunning", "thrilling", "special", "great",
    }

    def build_reversal_response(self, input_vadug, response_vadug, is_positive_reversal):
        """Build a response for a reversal chunk (after 'but', 'however', etc.)."""
        if is_positive_reversal:
            # Use situation-appropriate positive adjectives only
            sit_pool = {w: WORD_FORCES[w] for w in self.SITUATION_POS_ADJECTIVES
                        if w in WORD_FORCES and w not in self.used_words}
            word = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                sit_pool if sit_pool else None
            )
            structures = [
                f"But that part is {word}." if word else None,
                f"But wait -- that's {word}!" if word else None,
                f"But hold on, that's genuinely {word}." if word else None,
                "But that changes things.",
                "But that's a different story entirely.",
            ]
        else:
            adj_pool = {w: f for w, f in self.neg_adj_pool.items() if w not in self.used_words}
            noun_pool = {w: f for w, f in self.noun_pool.items()
                         if w not in self.used_words and WORD_FORCES.get(w, (0,))[0] < 0}
            adj = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                adj_pool if adj_pool else None
            )
            noun = self.select_word(
                input_vadug.v, input_vadug.a, input_vadug.d, input_vadug.u, input_vadug.g,
                noun_pool if noun_pool else None
            )
            structures = [
                f"But that part is {adj}." if adj else None,
                f"Though that's genuinely {adj}." if adj else None,
                f"But I hear the {noun} in that too." if noun else None,
                "But that part is hard.",
                "But I hear the hard part too.",
            ]

        result = self._pick_structure(structures)
        return result if result else ("But that changes things." if is_positive_reversal else "But that part is hard.")

    def build_chunk_response(self, input_vadug, response_vadug, grade_rules=None,
                              is_first=True, is_reversal=False,
                              is_subsequent_negative=False,
                              is_positive=False, chunk_text=""):
        """Build a complete response for one chunk using WORD_FORCES math."""
        blocked = grade_rules.get("blocked", []) if grade_rules else []
        grade = grade_rules.get("grade", "C") if grade_rules else "C"

        # F-range crisis override: presence only, still math-selected
        if grade in ("F-", "F", "F+"):
            if grade == "F-":
                stab_pool = {w: WORD_FORCES[w] for w in self.STABILIZE_WORDS if w in WORD_FORCES}
                anchor = self.select_word(
                    response_vadug.v, response_vadug.a, response_vadug.d,
                    response_vadug.u, response_vadug.g, stab_pool
                )
                structures = [
                    f"I'm {anchor}." if anchor and anchor in ("here",) else None,
                    "I'm here.",
                    "I hear you.",
                    "You're not alone.",
                ]
                return self._pick_structure(structures) or "I'm here."
            else:
                ack = self.build_acknowledge(input_vadug, response_vadug)
                return ack if ack else "I hear you."

        # Reversal chunks
        if is_reversal:
            is_pos_reversal = input_vadug.v > 148
            return self.build_reversal_response(input_vadug, response_vadug, is_pos_reversal)

        # Positive chunks
        if is_positive and input_vadug.v > 150:
            if "positive_spin" in blocked or "ANY_positive_framing" in blocked:
                return ""
            if input_vadug.v > 190:
                return self.build_positive_acknowledge(input_vadug, response_vadug)
            return ""

        # Dead neutral: skip
        if 135 <= input_vadug.v <= 165 and is_subsequent_negative is False and not is_first:
            return ""

        parts = []

        # Acknowledgment
        if input_vadug.v < 135 or input_vadug.v > 165:
            if is_subsequent_negative:
                # Shorter acknowledgment for follow-on negative chunks
                ack = self.build_acknowledge(input_vadug, response_vadug)
                return ack if ack else "I hear that."
            else:
                ack = self.build_acknowledge(input_vadug, response_vadug)
                if ack:
                    parts.append(ack)

        # Stabilize (for first negative chunk or if dominance is low)
        if is_first and input_vadug.d < 110 and input_vadug.v < 135:
            if "unsolicited_advice" in blocked:
                stab = self._pick_structure([
                    "I'm right here.",
                    "I'm with you.",
                ])
            else:
                stab = self.build_stabilize(response_vadug)
            if stab:
                parts.append(stab)

        # Redirect (only if grade allows)
        if not is_first and not is_reversal and input_vadug.v >= 100:
            red = self.build_redirect(response_vadug, grade_rules)
            if red:
                parts.append(red)

        return ' '.join(parts) if parts else "I hear you."

    def build_full_response(self, chunk_results, arc, grade, grade_rules, personality,
                             verbose=False):
        """Build the complete assembled response for all chunks."""
        responses = []
        seen_negative = False

        for i, chunk in enumerate(chunk_results):
            is_first = (i == 0)
            input_vadug = chunk['vadug']
            response_vadug = compute_harmony(input_vadug, personality)
            response_vadug, _ = apply_personality(response_vadug, input_vadug, personality)

            is_negative = input_vadug.v < 135
            chunk_lower = chunk['text'].lower()

            # Content-based negativity check
            negative_content_words = {"sick", "broke", "broken", "died", "lost",
                                       "hurt", "failed", "crash", "fire", "rent",
                                       "raising", "can't take", "much more",
                                       "don't know", "struggle", "pain"}
            has_negative_content = any(w in chunk_lower for w in negative_content_words)
            if has_negative_content and input_vadug.v < 155:
                is_negative = True

            is_reversal = False
            first_word = chunk['text'].split()[0].lower().strip('.,!?;:') if chunk['text'].split() else ""
            if first_word in ChunkSplitter.REVERSAL_WORDS:
                is_reversal = True

            is_positive = input_vadug.v > 150

            resp = self.build_chunk_response(
                input_vadug,
                response_vadug,
                grade_rules,
                is_first=(is_first and not seen_negative) if is_negative else is_first,
                is_reversal=is_reversal,
                is_subsequent_negative=(is_negative and seen_negative),
                is_positive=is_positive,
                chunk_text=chunk['text'],
            )

            if is_negative:
                seen_negative = True

            if resp and resp.strip() and resp not in responses:
                responses.append(resp)
                if verbose:
                    words_used = sorted(self.used_words)
                    print(f"  Chunk {i+1} -> \"{resp}\"")
                    print(f"    Words selected: {', '.join(words_used[-4:])}")
            elif verbose:
                print(f"  Chunk {i+1} -> (skipped, neutral/duplicate)")

        # Arc closer
        closer = self._build_arc_closer(arc, grade, grade_rules)
        if closer:
            responses.append(closer)
            if verbose:
                print(f"  Arc closer ({arc}) -> \"{closer}\"")

        # Assemble with transitions
        if len(responses) <= 1:
            return responses[0] if responses else "I hear you."

        assembled = responses[0]
        for i, r in enumerate(responses[1:], 1):
            r_lower = r.lower()
            if r_lower.startswith(("but ", "hold on", "now that", "though ")):
                assembled = assembled.rstrip('.!?') + ". " + r
            elif i == len(responses) - 1:
                assembled = assembled.rstrip('.!?') + ". " + r
            else:
                joined = r[0].lower() + r[1:] if r and r[0] != 'I' else r
                assembled = assembled.rstrip('.!?') + ", and " + joined

        return assembled

    def _build_arc_closer(self, arc, grade, grade_rules):
        """Build an arc-aware closing line."""
        if grade in ("F-", "F", "F+"):
            return "I'm here. You're not alone."

        if grade in ("D-", "D"):
            closers = {
                "valley": "But the hard part isn't all there is.",
                "peak": "That's heavy. One step at a time.",
                "descending": "Let's take this one thing at a time.",
                "ascending": "You're finding your way through this.",
                "flat_negative": "I'm here for all of it.",
                "contrast": "That bright spot doesn't change the hard part.",
                "mixed": "All of that matters.",
            }
            return closers.get(arc, "I'm here.")

        closers = {
            "valley": "Not everyone gets that kind of turning point.",
            "peak": "We'll work through the rough part.",
            "descending": "One thing at a time. We'll get there.",
            "ascending": "Things are moving in the right direction.",
            "flat_negative": "You're not carrying this alone.",
            "flat_positive": "That's incredible all around.",
            "contrast": "That one bright moment doesn't undo the rest. I see the whole picture.",
            "mixed": "Life's complicated like that. All of it counts.",
        }
        return closers.get(arc, "")

    def compute_response_length(self, chunks):
        """G axis controls response length. Heavy = brief. Light = can expand."""
        avg_g = sum(c['vadug'].g for c in chunks) / len(chunks)
        avg_v = sum(c['vadug'].v for c in chunks) / len(chunks)
        total_weight = sum(abs(128 - c['vadug'].v) for c in chunks)

        if avg_g < 60:      # crushing
            max_sentences = 1
        elif avg_g < 90:    # sinking
            max_sentences = 2
        elif avg_g < 140:   # grounded
            max_sentences = min(len(chunks), 3)
        else:               # floating/soaring
            max_sentences = min(len(chunks), 4)

        # Heavier total weight = fewer words needed
        if total_weight > 200:
            max_sentences = min(max_sentences, 2)
        if total_weight > 300:
            max_sentences = 1

        # Many chunks = complex story. Compress further.
        # 4+ emotional beats should never produce more than 2 summary sentences.
        if len(chunks) >= 4:
            max_sentences = min(max_sentences, 2)

        return max_sentences

    def build_summary_response(self, chunks, arc, grade, grade_rules, personality,
                                verbose=False):
        """Build ONE response that summarizes the emotional story, not per-chunk."""
        max_sentences = self.compute_response_length(chunks)

        # Overall VADUG (weighted average across chunks)
        # Contrast chunks get inverted weight: -0.3 instead of +1.0
        # This makes a brief positive in negative context EMPHASIZE the negativity
        weights = []
        for c in chunks:
            if c.get('contrast'):
                weights.append(-0.3)
            else:
                weights.append(1.0)
        total_weight_sum = sum(abs(w) for w in weights)
        if total_weight_sum == 0:
            total_weight_sum = 1.0

        avg_v = sum(c['vadug'].v * w for c, w in zip(chunks, weights)) / total_weight_sum
        avg_a = sum(c['vadug'].a * w for c, w in zip(chunks, weights)) / total_weight_sum
        avg_d = sum(c['vadug'].d * w for c, w in zip(chunks, weights)) / total_weight_sum
        avg_u = sum(c['vadug'].u * w for c, w in zip(chunks, weights)) / total_weight_sum
        avg_g = sum(c['vadug'].g * w for c, w in zip(chunks, weights)) / total_weight_sum

        # Compute target response VADUG from harmony on the OVERALL emotion
        overall_vadug = VADUG(v=int(avg_v), a=int(avg_a), d=int(avg_d), u=int(avg_u), g=int(avg_g))
        response_vadug = compute_harmony(overall_vadug, personality if personality else PersonalityVector())

        if verbose:
            print(f"  Summary mode: avg_g={avg_g:.0f}, max_sentences={max_sentences}, arc={arc}")

        parts = []

        # Neutral/operational inputs (V 118-165, G grounded): brief operational response
        # But NOT if there's emotional spread (some chunks very negative/positive)
        v_spread = max(c['vadug'].v for c in chunks) - min(c['vadug'].v for c in chunks)
        truly_neutral = 118 <= avg_v <= 165 and avg_g >= 90 and v_spread < 40
        if truly_neutral:
            if avg_v > 145:
                # Mildly positive
                ack = self.build_positive_acknowledge(overall_vadug, response_vadug)
                if ack:
                    parts.append(ack)
            else:
                # Dead neutral / operational -- minimal acknowledgment
                parts.append("On it.")
            if verbose:
                print(f"  Summary built {len(parts)} part(s): {parts}")
            return ' '.join(parts) if parts else "I hear you."

        # Emotional trajectory: compute blended targets for each sentence position.
        # Sentence 1 (acknowledge): stays close to input VADUG (meet user where they are)
        # Sentence 2 (stabilize): midpoint between input and response target
        # Sentence 3 (redirect): at response VADUG target (where we're guiding them)
        traj_1 = self._get_trajectory_target(overall_vadug, response_vadug, 0, max_sentences)
        traj_2 = self._get_trajectory_target(overall_vadug, response_vadug, 1, max_sentences)
        traj_3 = self._get_trajectory_target(overall_vadug, response_vadug, 2, max(max_sentences, 3))

        # Sentence 1: acknowledge the overall weight (using trajectory position 0 = near input)
        if avg_v > 150:
            ack = self.build_positive_acknowledge(traj_1, response_vadug)
        else:
            ack = self.build_acknowledge(traj_1, response_vadug)
        if ack:
            parts.append(ack)

        # Sentence 2 (if allowed): stabilize or arc-closer (trajectory midpoint)
        if max_sentences >= 2:
            if grade in ("F-", "F", "F+"):
                parts.append("I'm here.")
            elif arc == "contrast":
                # Contrast arc: acknowledge the bright spot doesn't fix the pain
                closer = self._build_arc_closer(arc, grade, grade_rules)
                if closer:
                    parts.append(closer)
            elif arc == "valley" and avg_v > 100:
                # Valley that ends positive -- acknowledge the good part
                closer = self._build_arc_closer(arc, grade, grade_rules)
                if closer:
                    parts.append(closer)
            elif avg_d < 90:
                # Low dominance -- offer stability (use trajectory midpoint)
                stab = self.build_stabilize(traj_2)
                if stab:
                    parts.append(stab)
            else:
                closer = self._build_arc_closer(arc, grade, grade_rules)
                if closer:
                    parts.append(closer)

        # Sentence 3 (only if G > 120 and max allows) -- trajectory at response target
        if max_sentences >= 3 and avg_g > 120:
            red = self.build_redirect(traj_3, grade_rules)
            if red:
                parts.append(red)

        if verbose:
            print(f"  Summary built {len(parts)} part(s): {parts}")

        return ' '.join(parts) if parts else "I hear you."

    def reset_for_new_response(self):
        """Call between responses to reset per-response tracking."""
        self.used_words.clear()
        self.used_structures.clear()


# =============================================================
