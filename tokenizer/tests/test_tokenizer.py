"""Comprehensive tests for the ClankerTokenizer."""

import json
import os
import sys
import tempfile
import unittest

# Ensure the tokenizer package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tokenizer.clanker_tokenizer import ClankerTokenizer


class TestClankerTokenizerBasic(unittest.TestCase):
    """Basic tokenizer properties and special tokens."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def test_vocab_size_reasonable(self):
        """Vocab should be in the ~500-800 range for Clanker's ~512 token design."""
        self.assertGreater(self.tok.vocab_size, 400)
        self.assertLess(self.tok.vocab_size, 1000)

    def test_special_tokens_exist(self):
        """All required special tokens must be in the vocab."""
        for token in ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>",
                       "<emotion>", "<round>", "<context>"]:
            self.assertIn(token, self.tok.get_vocab(),
                          f"Missing special token: {token}")

    def test_special_token_ids(self):
        """Special token IDs should be deterministic."""
        self.assertEqual(self.tok.pad_token_id, 0)
        self.assertEqual(self.tok.bos_token_id, 1)
        self.assertEqual(self.tok.eos_token_id, 2)
        self.assertEqual(self.tok.unk_token_id, 3)

    def test_repr(self):
        self.assertIn("ClankerTokenizer", repr(self.tok))
        self.assertIn("vocab_size=", repr(self.tok))


class TestOpcodeTokens(unittest.TestCase):
    """Every defined opcode must have both hex and name tokens."""

    def setUp(self):
        self.tok = ClankerTokenizer()
        self.vocab = self.tok.get_vocab()

    def test_core_opcodes_present(self):
        core_names = [
            "NOP_DONE", "WAIT", "SET", "EMIT", "LOG", "CALL", "RETURN",
            "LOAD", "STORE", "DELETE", "ASSERT", "DEFINE", "LABEL", "GOTO",
            "REGISTER", "END", "COMMENT", "YIELD", "SPAWN", "JOIN",
            "SEND", "RECEIVE",
        ]
        for name in core_names:
            self.assertIn(name, self.vocab, f"Missing core opcode: {name}")

    def test_reasoning_opcodes_present(self):
        names = ["THINK", "CHECK", "INFER", "DERIVE", "ANSWER", "DOUBT", "ASSUME"]
        for name in names:
            self.assertIn(name, self.vocab, f"Missing reasoning opcode: {name}")

    def test_hardware_opcodes_present(self):
        names = [
            "GPIO_SET", "GPIO_READ", "PWM_SET", "I2C_WRITE", "I2C_READ",
            "SPI_TRANSFER", "UART_SEND", "UART_RECV", "ADC_READ",
            "SENSOR_READ", "MOTOR_SET", "DISPLAY_WRITE",
        ]
        for name in names:
            self.assertIn(name, self.vocab, f"Missing hardware opcode: {name}")

    def test_web_opcodes_present(self):
        names = [
            "ENDPOINT", "RESPOND", "REQUEST", "HEADER",
            "MIDDLEWARE", "WEBSOCKET", "AUTH", "RATE_LIMIT",
        ]
        for name in names:
            self.assertIn(name, self.vocab, f"Missing web opcode: {name}")

    def test_data_opcodes_present(self):
        names = ["QUERY", "TRANSFORM", "VALIDATE", "MAP", "FILTER"]
        for name in names:
            self.assertIn(name, self.vocab, f"Missing data opcode: {name}")

    def test_logic_opcodes_present(self):
        names = ["WHEN", "MATCH", "REPEAT", "TRY"]
        for name in names:
            self.assertIn(name, self.vocab, f"Missing logic opcode: {name}")

    def test_hex_codes_present(self):
        """Every opcode must have a hex token like 0x00, 0xC0, etc."""
        expected_hex = [
            "0x00", "0x01", "0x02", "0x03", "0x04", "0x05", "0x06",
            "0x07", "0x08", "0x09", "0x0A", "0x0B", "0x0C", "0x0D",
            "0x0E", "0x0F", "0x10", "0x11", "0x12", "0x13", "0x14",
            "0x15", "0x20", "0x21", "0x22", "0x23", "0x24", "0x25",
            "0x26", "0xA0", "0xA1", "0xA2", "0xA3", "0xA4", "0xA5",
            "0xA6", "0xA7", "0xA8", "0xA9", "0xAA", "0xAB",
            "0xC0", "0xC1", "0xC2", "0xC3", "0xC4", "0xC5", "0xC6",
            "0xC7", "0xD0", "0xD1", "0xD2", "0xD3", "0xD4",
            "0xE0", "0xE1", "0xE2", "0xE3",
        ]
        for h in expected_hex:
            self.assertIn(h, self.vocab, f"Missing hex opcode token: {h}")

    def test_get_opcode_id(self):
        """get_opcode_id should resolve both hex and name forms."""
        self.assertIsNotNone(self.tok.get_opcode_id("SET"))
        self.assertIsNotNone(self.tok.get_opcode_id("0x02"))
        self.assertIsNotNone(self.tok.get_opcode_id("THINK"))
        self.assertIsNone(self.tok.get_opcode_id("NONEXISTENT_OP"))


class TestVariableRefTokens(unittest.TestCase):
    """Variable reference tokens $0-$255 and $_."""

    def setUp(self):
        self.vocab = ClankerTokenizer().get_vocab()

    def test_unused_slot(self):
        self.assertIn("$_", self.vocab)

    def test_var_refs_full_range(self):
        for i in range(256):
            self.assertIn(f"${i}", self.vocab, f"Missing varref ${i}")

    def test_var_ref_ids_unique(self):
        ids = [self.vocab[f"${i}"] for i in range(256)]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate varref token IDs")


class TestParamTypeTokens(unittest.TestCase):
    """Parameter type tokens."""

    def setUp(self):
        self.vocab = ClankerTokenizer().get_vocab()

    def test_all_param_types(self):
        types = ["str", "str_ext", "int", "float", "bool", "duration",
                 "bytes", "list", "varref", "map", "null"]
        for t in types:
            self.assertIn(t, self.vocab, f"Missing param type: {t}")


class TestVADUGTokens(unittest.TestCase):
    """Emotional vector (VADUG) tokens."""

    def setUp(self):
        self.tok = ClankerTokenizer()
        self.vocab = self.tok.get_vocab()

    def test_vadug_axis_tokens(self):
        for axis in ["[V", "[A", "[D", "[U", "[G"]:
            self.assertIn(axis, self.vocab, f"Missing VADUG axis: {axis}")
        self.assertIn("]", self.vocab)

    def test_vadug_bucket_tokens(self):
        buckets = ["neg_high", "neg_med", "neg_low", "neutral",
                    "pos_low", "pos_med", "pos_high"]
        for axis in ["V", "A", "D", "U", "G"]:
            for b in buckets:
                token = f"{axis}_{b}"
                self.assertIn(token, self.vocab, f"Missing bucket: {token}")

    def test_get_vadug_bucket(self):
        self.assertEqual(self.tok.get_vadug_bucket("V", 0), "V_neg_high")
        self.assertEqual(self.tok.get_vadug_bucket("V", 36), "V_neg_high")
        self.assertEqual(self.tok.get_vadug_bucket("V", 37), "V_neg_med")
        self.assertEqual(self.tok.get_vadug_bucket("V", 128), "V_neutral")
        self.assertEqual(self.tok.get_vadug_bucket("V", 255), "V_pos_high")
        self.assertEqual(self.tok.get_vadug_bucket("A", 200), "A_pos_med")
        self.assertEqual(self.tok.get_vadug_bucket("G", 220), "G_pos_high")

    def test_encode_vadug_vector(self):
        ids = self.tok.encode_vadug(65, 221, 163, 49, 174)
        self.assertEqual(len(ids), 5)
        # Verify each bucket
        tokens = self.tok.convert_ids_to_tokens(ids)
        self.assertEqual(tokens[0], "V_neg_med")       # 65
        self.assertEqual(tokens[1], "A_pos_high")      # 221
        self.assertEqual(tokens[2], "D_pos_low")       # 163
        self.assertEqual(tokens[3], "U_neg_med")       # 49
        self.assertEqual(tokens[4], "G_pos_low")       # 174

    def test_emotion_annotation_tokens(self):
        self.assertIn("![", self.vocab)
        for p in ["v:", "a:", "d:", "u:", "g:"]:
            self.assertIn(p, self.vocab, f"Missing emotion prefix: {p}")


class TestTokenization(unittest.TestCase):
    """Tokenize Clanker text format."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def test_simple_instruction(self):
        text = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        tokens = self.tok.tokenize(text)
        self.assertIn("@", tokens)
        self.assertIn("0x02", tokens)
        self.assertIn("$0", tokens)
        self.assertIn("$_", tokens)

    def test_vadug_header(self):
        text = "[V65] [A221] [D163] [U49] [G174]"
        tokens = self.tok.tokenize(text)
        self.assertIn("[V", tokens)
        self.assertIn("65", tokens)
        self.assertIn("]", tokens)
        self.assertIn("[A", tokens)
        self.assertIn("221", tokens)

    def test_emotion_annotation(self):
        text = '@ 0xC1 $1 $2 01 {status: 500} ![v:28 a:248 d:88 u:240 g:100]'
        tokens = self.tok.tokenize(text)
        self.assertIn("![", tokens)
        self.assertIn("v:", tokens)
        self.assertIn("28", tokens)
        self.assertIn("a:", tokens)
        self.assertIn("248", tokens)

    def test_multiline_program(self):
        program = """@ 0x02 $0 $_ 02 {name: "greeting"} {value: "hello"}
@ 0x03 $0 $_ 01 {value: "hello world"}"""
        tokens = self.tok.tokenize(program)
        self.assertIn("<newline>", tokens)
        # Should have two @ tokens
        self.assertEqual(tokens.count("@"), 2)


class TestEncoding(unittest.TestCase):
    """Encode Clanker text to token IDs."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def test_encode_with_special_tokens(self):
        text = "@ 0x00 $_ $_ 00"
        ids = self.tok.encode(text, add_special_tokens=True)
        self.assertEqual(ids[0], self.tok.bos_token_id)
        self.assertEqual(ids[-1], self.tok.eos_token_id)

    def test_encode_without_special_tokens(self):
        text = "@ 0x00 $_ $_ 00"
        ids = self.tok.encode(text, add_special_tokens=False)
        self.assertNotEqual(ids[0], self.tok.bos_token_id)

    def test_all_ids_valid(self):
        text = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/api/health"}'
        ids = self.tok.encode(text, add_special_tokens=False)
        for token_id in ids:
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.tok.vocab_size)

    def test_unknown_token_maps_to_unk(self):
        """Tokens not in vocab should map to <unk> ID."""
        unk_id = self.tok.unk_token_id
        result = self.tok._convert_token_to_id("TOTALLY_NONEXISTENT_TOKEN_XYZ")
        self.assertEqual(result, unk_id)


class TestDecoding(unittest.TestCase):
    """Decode token IDs back to Clanker text."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def test_decode_skips_special_tokens(self):
        ids = [self.tok.bos_token_id, self.tok.eos_token_id]
        result = self.tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(result, "")

    def test_decode_preserves_special_tokens(self):
        ids = [self.tok.bos_token_id]
        result = self.tok.decode(ids, skip_special_tokens=False)
        self.assertIn("<bos>", result)


class TestRoundTrip(unittest.TestCase):
    """Encode -> Decode round-trip must be lossless for valid Clanker programs."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def _round_trip(self, text: str) -> str:
        ids = self.tok.encode(text, add_special_tokens=True)
        return self.tok.decode(ids, skip_special_tokens=True)

    def test_simple_nop(self):
        original = "@ 0x00 $_ $_ 00"
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_set_instruction(self):
        original = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_web_endpoint(self):
        original = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/api/health"}'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_vadug_header(self):
        original = "[V65] [A221] [D163] [U49] [G174]"
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_emotion_annotation(self):
        original = '@ 0xC1 $1 $2 01 {status: 500} ![v:28 a:248 d:88 u:240 g:100]'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_multiline_program(self):
        original = '@ 0x02 $0 $_ 02 {name: "greeting"} {value: "hello"}\n@ 0x03 $0 $_ 01 {value: "hello world"}'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_reasoning_chain(self):
        original = '@ 0x20 $_ $_ 01 {premise: "sort list"}'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_logic_when(self):
        original = '@ 0xE0 $_ $_ 01 {condition: "x > 0"}'
        result = self._round_trip(original)
        self.assertEqual(result, original)

    def test_hardware_gpio(self):
        original = '@ 0xA0 $_ $_ 02 {pin: 13} {state: true}'
        result = self._round_trip(original)
        self.assertEqual(result, original)


class TestMetadataTokens(unittest.TestCase):
    """Metadata header tokens (CERT, SRC, GOAL, REL)."""

    def setUp(self):
        self.vocab = ClankerTokenizer().get_vocab()

    def test_cert_src_goal_rel(self):
        for t in ["CERT", "SRC", "GOAL", "REL"]:
            self.assertIn(t, self.vocab)

    def test_src_values(self):
        for t in ["SRC_UNKNOWN", "SRC_TRAINED", "SRC_RAG", "SRC_INFERRED",
                   "SRC_USER", "SRC_EXTERNAL", "SRC_VERIFIED"]:
            self.assertIn(t, self.vocab)

    def test_goal_values(self):
        for t in ["GOAL_HELP", "GOAL_CLARIFY", "GOAL_WARN", "GOAL_TEACH",
                   "GOAL_EXECUTE", "GOAL_REFUSE", "GOAL_EMPATHIZE",
                   "GOAL_CONFIRM", "GOAL_EXPLORE"]:
            self.assertIn(t, self.vocab)


class TestPersonalityTokens(unittest.TestCase):
    """Personality vector tokens."""

    def setUp(self):
        self.vocab = ClankerTokenizer().get_vocab()

    def test_personality_dimensions(self):
        for t in ["GULLIBILITY", "AGREEABLENESS", "SUGGESTIBILITY",
                   "TRUTHFULNESS", "SAFETY", "CURIOSITY",
                   "ASSERTIVENESS", "PLAYFULNESS"]:
            self.assertIn(t, self.vocab)


class TestSaveLoad(unittest.TestCase):
    """Save and reload the tokenizer."""

    def test_save_and_reload(self):
        tok = ClankerTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save_vocabulary(tmpdir)
            vocab_path = os.path.join(tmpdir, "vocab.json")
            special_path = os.path.join(tmpdir, "special_tokens_map.json")
            self.assertTrue(os.path.exists(vocab_path))
            self.assertTrue(os.path.exists(special_path))

            # Reload
            tok2 = ClankerTokenizer(vocab_file=vocab_path, special_tokens_file=special_path)
            self.assertEqual(tok.vocab_size, tok2.vocab_size)

            # Round-trip should still work
            text = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
            ids1 = tok.encode(text, add_special_tokens=False)
            ids2 = tok2.encode(text, add_special_tokens=False)
            self.assertEqual(ids1, ids2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def setUp(self):
        self.tok = ClankerTokenizer()

    def test_empty_input(self):
        ids = self.tok.encode("", add_special_tokens=False)
        self.assertEqual(ids, [])

    def test_empty_with_special(self):
        ids = self.tok.encode("", add_special_tokens=True)
        self.assertEqual(ids, [self.tok.bos_token_id, self.tok.eos_token_id])

    def test_max_varref(self):
        text = "@ 0x02 $255 $_ 00"
        tokens = self.tok.tokenize(text)
        self.assertIn("$255", tokens)

    def test_boolean_param_values(self):
        text = '@ 0xA0 $_ $_ 02 {pin: 13} {state: true}'
        tokens = self.tok.tokenize(text)
        self.assertIn("true", tokens)

    def test_numeric_param_values(self):
        # Numbers 0-255 are single tokens; larger numbers use char-level fallback
        text = '@ 0xC1 $1 $2 01 {status: 200}'
        tokens = self.tok.tokenize(text)
        self.assertIn("200", tokens)

    def test_large_numeric_char_split(self):
        # Numbers > 255 are not in vocab, so they get char-level split
        text = '@ 0xC1 $1 $2 01 {status: 500}'
        tokens = self.tok.tokenize(text)
        self.assertIn("5", tokens)
        self.assertIn("0", tokens)


if __name__ == "__main__":
    unittest.main()
