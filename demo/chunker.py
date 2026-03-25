"""Paragraph chunking for the Clanker pipeline."""

import re

# =============================================================
# STEP 7: Emotional Chunking — Paragraph-Level Arc Detection
# =============================================================

class ChunkSplitter:
    """Splits input text at natural emotional boundaries.

    Boundary types:
    - Sentence endings: . ! ?
    - Conjunctive reversals: but, however, although, yet, though
    - Causal links: because, since
    - "so" when followed by a subject pronoun (so I, so we, so they)

    Rules:
    - Minimum chunk size: 2 words
    - Maximum chunk size: ~20 words (split at commas if needed)
    - Splitting word stays with the NEW chunk
    """

    # Words that trigger a split — the word goes with the NEW chunk
    REVERSAL_WORDS = {"but", "however", "although", "yet", "though"}
    CAUSAL_WORDS = {"because", "since"}
    SUBJECT_PRONOUNS = {"i", "we", "they", "he", "she", "it", "you"}

    MIN_CHUNK_WORDS = 2
    MAX_CHUNK_WORDS = 20

    def split(self, text: str) -> list:
        """Split text into emotional chunks. Returns list of strings."""
        # First, split at sentence boundaries (. ! ?)
        # Preserve the punctuation with the preceding chunk
        sentence_chunks = self._split_sentences(text)

        # Then split each sentence at emotional boundaries
        final_chunks = []
        for sentence in sentence_chunks:
            sub_chunks = self._split_at_boundaries(sentence)
            final_chunks.extend(sub_chunks)

        # Enforce max chunk size by splitting at commas
        sized_chunks = []
        for chunk in final_chunks:
            if self._word_count(chunk) > self.MAX_CHUNK_WORDS:
                sized_chunks.extend(self._split_at_commas(chunk))
            else:
                sized_chunks.append(chunk)

        # Merge any too-small chunks with neighbors
        merged = self._merge_small_chunks(sized_chunks)

        return [c.strip() for c in merged if c.strip()]

    def _split_sentences(self, text: str) -> list:
        """Split at sentence boundaries (. ! ?) while preserving punctuation."""
        # Split but keep the delimiter with the preceding text
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_at_boundaries(self, text: str) -> list:
        """Split a sentence at emotional boundary words."""
        words = text.split()
        if len(words) <= self.MIN_CHUNK_WORDS:
            return [text]

        chunks = []
        current_start = 0

        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')

            is_boundary = False

            # Check reversal words
            if word_lower in self.REVERSAL_WORDS:
                is_boundary = True

            # Check causal words
            elif word_lower in self.CAUSAL_WORDS:
                is_boundary = True

            # Check "so" + subject pronoun
            elif word_lower == "so" and i + 1 < len(words):
                next_word = words[i + 1].lower().strip('.,!?;:')
                if next_word in self.SUBJECT_PRONOUNS:
                    is_boundary = True

            if is_boundary and i > current_start:
                # Only split if the preceding chunk has enough words
                preceding = words[current_start:i]
                if len(preceding) >= self.MIN_CHUNK_WORDS:
                    # Strip trailing comma from the preceding chunk
                    chunk_text = " ".join(preceding)
                    chunk_text = chunk_text.rstrip(',').rstrip()
                    chunks.append(chunk_text)
                    current_start = i

        # Add the remaining words
        if current_start < len(words):
            chunks.append(" ".join(words[current_start:]))

        return chunks

    def _split_at_commas(self, text: str) -> list:
        """Split long chunks at commas to enforce max size."""
        parts = text.split(',')
        chunks = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
            test = (current + ", " + part).strip(', ') if current else part
            if self._word_count(test) > self.MAX_CHUNK_WORDS and current:
                chunks.append(current.strip())
                current = part
            else:
                current = test

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _merge_small_chunks(self, chunks: list) -> list:
        """Merge chunks that are too small with their neighbors."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if self._word_count(chunk) < self.MIN_CHUNK_WORDS:
                if merged:
                    # Merge with previous
                    merged[-1] = merged[-1].rstrip() + " " + chunk
                elif i + 1 < len(chunks):
                    # Merge with next
                    chunks[i + 1] = chunk + " " + chunks[i + 1]
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)
            i += 1

        return merged

    def _word_count(self, text: str) -> int:
        return len(text.split())

