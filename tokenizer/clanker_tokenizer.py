"""
ClankerTokenizer — Direct-mapping tokenizer for Clanker bytecode IR.

This is NOT a BPE/subword tokenizer. Each Clanker opcode, variable reference,
parameter type, and structural element maps directly to a single token ID.
The vocabulary is ~960 tokens covering the full Clanker instruction set.

For string values not in the vocabulary, character-level fallback is used
with ASCII printable characters in the vocab.

HuggingFace PreTrainedTokenizer compatible.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

    class PreTrainedTokenizer:
        """Minimal stub when transformers is not installed."""

        def __init__(self, **kwargs):
            self._added_tokens_encoder = {}
            self._added_tokens_decoder = {}

        def __len__(self):
            return self.vocab_size

        @property
        def vocab_size(self):
            raise NotImplementedError

        def save_pretrained(self, save_directory, **kwargs):
            pass


_TOKENIZER_DIR = os.path.dirname(os.path.abspath(__file__))

# Regex patterns for Clanker text format tokenization
_VADUG_HEADER_RE = re.compile(r'\[(V|A|D|U|G)(\d{1,3})\]')
_EMOTION_ANNOT_RE = re.compile(r'!\[v:(\d+)\s+a:(\d+)\s+d:(\d+)\s+u:(\d+)\s+g:(\d+)\]')
_VARREF_RE = re.compile(r'\$(_|\d{1,3})')
_HEX_OPCODE_RE = re.compile(r'0x[0-9A-Fa-f]{2}')
_PARAM_BLOCK_RE = re.compile(r'\{([^}]*)\}')
_PARAM_KV_RE = re.compile(r'(\w+):\s*("(?:[^"\\]|\\.)*"|true|false|-?\d+(?:\.\d+)?|\$(?:_|\d{1,3}))')

# Sentinel used by the tokenizer to mark character-level tokens
_CHAR_PREFIX = "chr_"


class ClankerTokenizer(PreTrainedTokenizer if HAS_TRANSFORMERS else PreTrainedTokenizer):
    """
    A direct-mapping tokenizer for Clanker bytecode IR.

    Each opcode, variable reference, type name, and structural element
    maps 1:1 to a token ID. String values use character-level fallback
    when not present as whole tokens in the vocabulary.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        special_tokens_file: Optional[str] = None,
        **kwargs,
    ):
        # Load vocab
        if vocab_file is None:
            vocab_file = os.path.join(_TOKENIZER_DIR, "vocab.json")
        with open(vocab_file, "r") as f:
            self._vocab: Dict[str, int] = json.load(f)

        self._id_to_token: Dict[int, str] = {v: k for k, v in self._vocab.items()}

        # Load special tokens map
        if special_tokens_file is None:
            special_tokens_file = os.path.join(_TOKENIZER_DIR, "special_tokens_map.json")
        with open(special_tokens_file, "r") as f:
            self._special_tokens_map = json.load(f)

        # Build sets for fast lookup
        self._special_tokens = set()
        self._special_tokens.add(self._special_tokens_map["pad_token"])
        self._special_tokens.add(self._special_tokens_map["bos_token"])
        self._special_tokens.add(self._special_tokens_map["eos_token"])
        self._special_tokens.add(self._special_tokens_map["unk_token"])
        self._special_tokens.add(self._special_tokens_map["sep_token"])
        for t in self._special_tokens_map.get("additional_special_tokens", []):
            self._special_tokens.add(t)

        # Build char lookup for single-character tokens
        self._char_tokens: Dict[str, int] = {}
        for token, tid in self._vocab.items():
            if len(token) == 1 and 32 <= ord(token) < 127:
                self._char_tokens[token] = tid

        if HAS_TRANSFORMERS:
            super().__init__(
                pad_token=self._special_tokens_map["pad_token"],
                bos_token=self._special_tokens_map["bos_token"],
                eos_token=self._special_tokens_map["eos_token"],
                unk_token=self._special_tokens_map["unk_token"],
                sep_token=self._special_tokens_map["sep_token"],
                additional_special_tokens=self._special_tokens_map.get(
                    "additional_special_tokens", []
                ),
                **kwargs,
            )
        else:
            super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Core properties
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #

    def _tokenize_string_value(self, s: str) -> List[str]:
        """
        Tokenize a string value. If the whole string is in the vocab,
        return it as one token. Otherwise, split into characters.
        """
        if s in self._vocab:
            return [s]
        # Character-level fallback
        chars = []
        for c in s:
            if c in self._char_tokens:
                chars.append(c)
            else:
                chars.append("<unk>")
        return chars

    def _tokenize_instruction(self, line: str) -> List[str]:
        """Tokenize a single Clanker instruction line."""
        tokens: List[str] = []
        line = line.strip()
        if not line:
            return tokens

        # Check for VADUG header lines like [V65] [A221] [D163] [U49] [G174]
        vadug_matches = list(_VADUG_HEADER_RE.finditer(line))
        if vadug_matches and line.startswith("["):
            for m in vadug_matches:
                axis = m.group(1)
                value = m.group(2)
                tokens.append(f"[{axis}")
                tokens.append(value)
                tokens.append("]")
            return tokens

        # Check for instruction prefix @
        if line.startswith("@"):
            tokens.append("@")
            line = line[1:].strip()

        # Extract emotion annotation if present
        emotion_match = _EMOTION_ANNOT_RE.search(line)
        emotion_tokens = []
        if emotion_match:
            emotion_tokens.append("![")
            emotion_tokens.append("v:")
            emotion_tokens.append(emotion_match.group(1))
            emotion_tokens.append("a:")
            emotion_tokens.append(emotion_match.group(2))
            emotion_tokens.append("d:")
            emotion_tokens.append(emotion_match.group(3))
            emotion_tokens.append("u:")
            emotion_tokens.append(emotion_match.group(4))
            emotion_tokens.append("g:")
            emotion_tokens.append(emotion_match.group(5))
            emotion_tokens.append("]")
            line = line[: emotion_match.start()].strip()

        # Extract param blocks {key: "value"}
        param_blocks = []
        param_positions = []
        for m in _PARAM_BLOCK_RE.finditer(line):
            param_blocks.append(m.group(0))
            param_positions.append((m.start(), m.end()))

        # Get the non-param prefix part
        if param_positions:
            prefix = line[: param_positions[0][0]].strip()
        else:
            prefix = line.strip()

        # Parse prefix: opcode target src param_count
        prefix_parts = prefix.split()
        for part in prefix_parts:
            if _HEX_OPCODE_RE.fullmatch(part):
                tokens.append("0x" + part[2:].upper())
            elif _VARREF_RE.fullmatch(part):
                tokens.append(part)
            elif part in self._vocab:
                tokens.append(part)
            else:
                upper = part.upper()
                if upper in self._vocab:
                    tokens.append(upper)
                else:
                    tokens.append(part)

        # Parse param blocks
        for block in param_blocks:
            tokens.append("{")
            inner = block[1:-1].strip()
            for kv_match in _PARAM_KV_RE.finditer(inner):
                key = kv_match.group(1)
                value = kv_match.group(2)
                # Tokenize the key
                tokens.extend(self._tokenize_string_value(key))
                tokens.append(":")
                if value.startswith('"') and value.endswith('"'):
                    tokens.append('"')
                    string_content = value[1:-1]
                    tokens.extend(self._tokenize_string_value(string_content))
                    tokens.append('"')
                elif value in ("true", "false"):
                    tokens.append(value)
                elif _VARREF_RE.fullmatch(value):
                    tokens.append(value)
                else:
                    # Numeric or other literal
                    tokens.extend(self._tokenize_string_value(value))
            tokens.append("}")

        tokens.extend(emotion_tokens)
        return tokens

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a Clanker program (one or more lines) into token strings.
        """
        all_tokens: List[str] = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line_tokens = self._tokenize_instruction(line)
            if line_tokens:
                all_tokens.extend(line_tokens)
                if i < len(lines) - 1:
                    all_tokens.append("<newline>")
        if all_tokens and all_tokens[-1] == "<newline>":
            all_tokens.pop()
        return all_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a single token string to its ID."""
        if token in self._vocab:
            return self._vocab[token]
        upper = token.upper()
        if upper in self._vocab:
            return self._vocab[upper]
        return self._vocab.get("<unk>", 3)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert a single token ID back to its string."""
        return self._id_to_token.get(index, "<unk>")

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(t) for t in tokens]

    # Structural special tokens that must survive skip_special_tokens
    _STRUCTURAL_SPECIAL = frozenset({"<newline>"})

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            token = self._convert_id_to_token(ids)
            if skip_special_tokens and token in self._special_tokens and token not in self._STRUCTURAL_SPECIAL:
                return ""
            return token
        result = []
        for i in ids:
            token = self._convert_id_to_token(i)
            if skip_special_tokens and token in self._special_tokens and token not in self._STRUCTURAL_SPECIAL:
                continue
            result.append(token)
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        """
        Encode a Clanker program string into a list of token IDs.

        Args:
            text: Clanker program text (one or more instruction lines).
            add_special_tokens: If True, wrap with <bos> and <eos>.

        Returns:
            List of integer token IDs.
        """
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            bos_id = self._vocab["<bos>"]
            eos_id = self._vocab["<eos>"]
            ids = [bos_id] + ids + [eos_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        """
        Decode a list of token IDs back into Clanker program text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, omit special tokens from output.

        Returns:
            Reconstructed Clanker program text.
        """
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        return self._reconstruct(tokens)

    def _is_char_token(self, token: str) -> bool:
        """Check if a token is a single ASCII printable character (char-level)."""
        return len(token) == 1 and 32 <= ord(token) < 127

    def _reconstruct(self, tokens: List[str]) -> str:
        """Reconstruct Clanker text from a list of token strings."""
        result_parts: List[str] = []
        i = 0

        def _append_spaced(text: str):
            """Append text with a preceding space if needed."""
            if result_parts and result_parts[-1] not in ("\n", ""):
                result_parts.append(" ")
            result_parts.append(text)

        while i < len(tokens):
            token = tokens[i]

            if token == "<newline>":
                result_parts.append("\n")
                i += 1
                continue

            # VADUG header: [V + number + ]
            if token.startswith("[") and len(token) == 2 and token[1] in "VADUG":
                if i + 2 < len(tokens) and tokens[i + 2] == "]":
                    _append_spaced(f"{token}{tokens[i+1]}{tokens[i+2]}")
                    i += 3
                    continue

            # Emotion annotation: ![ v: N a: N d: N u: N g: N ]
            if token == "![":
                parts = ["!["]
                j = i + 1
                while j < len(tokens) and tokens[j] != "]":
                    t = tokens[j]
                    if t in ("v:", "a:", "d:", "u:", "g:"):
                        if len(parts) > 1:
                            parts.append(" ")
                        parts.append(t)
                    else:
                        parts.append(t)
                    j += 1
                if j < len(tokens):
                    parts.append("]")
                    j += 1
                _append_spaced("".join(parts))
                i = j
                continue

            # Instruction prefix
            if token == "@":
                _append_spaced("@")
                i += 1
                continue

            # Param block: { key : value }
            if token == "{":
                j = i + 1
                block_str = self._reconstruct_param_block(tokens, j)
                _append_spaced(block_str[0])
                i = block_str[1]
                continue

            # Default: space-separated token
            _append_spaced(token)
            i += 1

        return "".join(result_parts).strip()

    def _reconstruct_param_block(self, tokens: List[str], start: int) -> Tuple[str, int]:
        """
        Reconstruct a param block from tokens starting after the opening '{'.
        Returns (block_string, next_index_after_closing_brace).
        """
        parts = ["{"]
        j = start

        while j < len(tokens) and tokens[j] != "}":
            # Read key: could be a single token or char-level sequence
            key_chars = []
            while j < len(tokens) and tokens[j] not in (":", "}"):
                key_chars.append(tokens[j])
                j += 1
            key = "".join(key_chars)
            parts.append(key)

            if j < len(tokens) and tokens[j] == ":":
                parts.append(": ")
                j += 1

                # Read value
                if j < len(tokens) and tokens[j] == '"':
                    # Quoted string value
                    j += 1
                    val_chars = []
                    while j < len(tokens) and tokens[j] != '"':
                        val_chars.append(tokens[j])
                        j += 1
                    val = "".join(val_chars)
                    parts.append('"')
                    parts.append(val)
                    parts.append('"')
                    if j < len(tokens) and tokens[j] == '"':
                        j += 1
                elif j < len(tokens) and tokens[j] != "}":
                    # Non-string value (bool, number, varref) — may be char-level
                    val_chars = []
                    while j < len(tokens) and tokens[j] not in ("{", "}", '"'):
                        # Check if next token starts a new key (followed by ':')
                        if (j + 1 < len(tokens) and tokens[j + 1] == ":"
                                and not self._is_char_token(tokens[j])):
                            break
                        # Also break if the current token is a known param name
                        # and the next is ':'
                        if (j + 1 < len(tokens) and tokens[j + 1] == ":"
                                and tokens[j] in self._vocab
                                and len(tokens[j]) > 1):
                            break
                        val_chars.append(tokens[j])
                        j += 1
                    parts.append("".join(val_chars))

        if j < len(tokens) and tokens[j] == "}":
            parts.append("}")
            j += 1

        return ("".join(parts), j)

    # ------------------------------------------------------------------ #
    # Persistence (HuggingFace compatible)
    # ------------------------------------------------------------------ #

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_path = os.path.join(save_directory, f"{prefix}vocab.json")
        special_path = os.path.join(save_directory, f"{prefix}special_tokens_map.json")

        with open(vocab_path, "w") as f:
            json.dump(self._vocab, f, indent=2)
        with open(special_path, "w") as f:
            json.dump(self._special_tokens_map, f, indent=2)

        return (vocab_path, special_path)

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    @property
    def pad_token_id(self) -> int:
        return self._vocab.get("<pad>", 0)

    @property
    def bos_token_id(self) -> int:
        return self._vocab.get("<bos>", 1)

    @property
    def eos_token_id(self) -> int:
        return self._vocab.get("<eos>", 2)

    @property
    def unk_token_id(self) -> int:
        return self._vocab.get("<unk>", 3)

    def get_opcode_id(self, opcode: str) -> Optional[int]:
        """Get the token ID for an opcode by name or hex code."""
        if opcode in self._vocab:
            return self._vocab[opcode]
        upper = opcode.upper()
        if upper in self._vocab:
            return self._vocab[upper]
        return None

    def get_vadug_bucket(self, axis: str, value: int) -> str:
        """
        Map a VADUG axis value (0-255) to a bucket token.

        Buckets: neg_high(0-36), neg_med(37-73), neg_low(74-109),
                 neutral(110-146), pos_low(147-182), pos_med(183-219),
                 pos_high(220-255)
        """
        boundaries = [37, 74, 110, 147, 183, 220]
        names = ["neg_high", "neg_med", "neg_low", "neutral", "pos_low", "pos_med", "pos_high"]
        bucket_idx = 0
        for b in boundaries:
            if value >= b:
                bucket_idx += 1
            else:
                break
        return f"{axis}_{names[bucket_idx]}"

    def encode_vadug(self, v: int, a: int, d: int, u: int, g: int) -> List[int]:
        """Encode a VADUG vector as bucket tokens."""
        tokens = []
        for axis, val in [("V", v), ("A", a), ("D", d), ("U", u), ("G", g)]:
            bucket = self.get_vadug_bucket(axis, val)
            if bucket in self._vocab:
                tokens.append(self._vocab[bucket])
        return tokens

    def __repr__(self) -> str:
        return f"ClankerTokenizer(vocab_size={self.vocab_size})"
