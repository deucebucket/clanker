"""
Tests for the Clanker binary compiler and decompiler.

Covers:
- Single instruction compilation
- Multi-instruction script compilation
- Round-trip (compile -> decompile -> matches original)
- All example files in examples/
- Binary size vs text size
- Magic bytes and version correctness
- VADUG emotional header encoding
- Error handling for invalid input
"""

import os
import re
import pytest
from pathlib import Path

from clanker_decoder.compiler import (
    ClankerCompiler,
    ClankerDecompiler,
    CompilerError,
)


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"


@pytest.fixture
def compiler():
    return ClankerCompiler()


@pytest.fixture
def decompiler():
    return ClankerDecompiler()


def normalize_text(text: str) -> str:
    """Normalize whitespace and quoting for round-trip comparison.

    Strips comments, blank lines, and trailing whitespace from each line.
    Also normalizes unquoted param values to quoted form, since the binary
    format doesn't preserve quoting style — the decompiler always emits
    quoted values.
    """
    quote_re = re.compile(r'\{(\w+):\s+([^}"]+)\}')
    lines = []
    for line in text.strip().split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            # Normalize unquoted values: {key: value} -> {key: "value"}
            stripped = quote_re.sub(
                lambda m: '{' + m.group(1) + ': "' + m.group(2).strip() + '"}',
                stripped,
            )
            lines.append(stripped)
    return '\n'.join(lines)


class TestMagicAndVersion:
    """Verify binary header structure."""

    def test_magic_bytes(self, compiler):
        binary = compiler.compile("@ 0x00")
        assert binary[:2] == b'\x43\x4C', "Magic bytes should be 'CL' (0x434C)"

    def test_version_byte(self, compiler):
        binary = compiler.compile("@ 0x00")
        assert binary[2] == 0x01, "Version byte should be 0x01"

    def test_header_length(self, compiler):
        binary = compiler.compile("@ 0x00")
        # 2 (magic) + 1 (version) + 4 (instruction: opcode + target + source + param_count)
        assert len(binary) == 7


class TestSingleInstruction:
    """Compile individual instructions to correct binary."""

    def test_bare_opcode(self, compiler):
        """Bare opcode like '@ 0x00' compiles to opcode + 0xFF + 0xFF + 0x00."""
        binary = compiler.compile("@ 0x00")
        instruction = binary[3:]  # skip header
        assert instruction == bytes([0x00, 0xFF, 0xFF, 0x00])

    def test_bare_opcode_0x0f(self, compiler):
        binary = compiler.compile("@ 0x0F")
        instruction = binary[3:]
        assert instruction == bytes([0x0F, 0xFF, 0xFF, 0x00])

    def test_instruction_with_params(self, compiler):
        text = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        binary = compiler.compile(text)
        instruction = binary[3:]

        assert instruction[0] == 0x02  # opcode
        assert instruction[1] == 0x00  # target $0
        assert instruction[2] == 0xFF  # source $_
        assert instruction[3] == 0x02  # param_count

        # First param: name="x"
        pos = 4
        assert instruction[pos] == 4  # key length ("name")
        pos += 1
        assert instruction[pos:pos + 4] == b'name'
        pos += 4
        assert instruction[pos:pos + 2] == b'\x00\x01'  # value length 1
        pos += 2
        assert instruction[pos:pos + 1] == b'x'
        pos += 1

        # Second param: value="42"
        assert instruction[pos] == 5  # key length ("value")
        pos += 1
        assert instruction[pos:pos + 5] == b'value'
        pos += 5
        assert instruction[pos:pos + 2] == b'\x00\x02'  # value length 2
        pos += 2
        assert instruction[pos:pos + 2] == b'42'

    def test_variable_slots(self, compiler):
        text = '@ 0x02 $3 $7 00'
        binary = compiler.compile(text)
        assert binary[4] == 3   # target $3
        assert binary[5] == 7   # source $7

    def test_underscore_vars(self, compiler):
        text = '@ 0x04 $_ $_ 01 {message: "hello"}'
        binary = compiler.compile(text)
        assert binary[4] == 0xFF  # target $_
        assert binary[5] == 0xFF  # source $_

    def test_single_param(self, compiler):
        text = '@ 0x04 $_ $_ 01 {message: "hello"}'
        binary = compiler.compile(text)
        instruction = binary[3:]
        assert instruction[3] == 0x01  # param_count


class TestMultiInstruction:
    """Compile multi-line scripts."""

    def test_two_instructions(self, compiler):
        text = '@ 0x02 $0 $_ 01 {name: "x"}\n@ 0x00'
        binary = compiler.compile(text)
        # Header (3) + first instruction + second instruction
        assert binary[:2] == b'\x43\x4C'
        assert binary[2] == 0x01
        # First instruction starts at 3
        assert binary[3] == 0x02
        # Second instruction should be 0x00 somewhere after first

    def test_comments_and_blanks_skipped(self, compiler):
        text = """
# this is a comment
@ 0x02 $0 $_ 01 {name: "x"}

# another comment

@ 0x00
"""
        binary = compiler.compile(text)
        # Should produce same result as without comments
        clean_text = '@ 0x02 $0 $_ 01 {name: "x"}\n@ 0x00'
        clean_binary = compiler.compile(clean_text)
        assert binary == clean_binary

    def test_hello_world_structure(self, compiler):
        text = """@ 0x10 $_ $_ 01 {text: "Hello World in Clank"}
@ 0x02 $0 $_ 02 {name: "greeting"} {value: "Hello, World!"}
@ 0x03 $0 $_ 01 {value: "greeting"}
@ 0x00 $_ $_ 01 {message: "program complete"}"""
        binary = compiler.compile(text)
        assert binary[:2] == b'\x43\x4C'
        # Should have 4 instructions
        assert binary[3] == 0x10  # first opcode


class TestRoundTrip:
    """Compile -> decompile produces identical text (whitespace-normalized)."""

    def test_bare_opcode_roundtrip(self, compiler, decompiler):
        original = "@ 0x00"
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_single_param_roundtrip(self, compiler, decompiler):
        original = '@ 0x04 $_ $_ 01 {message: "hello"}'
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_multi_param_roundtrip(self, compiler, decompiler):
        original = '@ 0x02 $0 $_ 02 {name: "greeting"} {value: "Hello, World!"}'
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_multi_instruction_roundtrip(self, compiler, decompiler):
        original = """@ 0x10 $_ $_ 01 {text: "Hello World in Clank"}
@ 0x02 $0 $_ 02 {name: "greeting"} {value: "Hello, World!"}
@ 0x03 $0 $_ 01 {value: "greeting"}
@ 0x00 $_ $_ 01 {message: "program complete"}"""
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_variable_slots_roundtrip(self, compiler, decompiler):
        original = '@ 0xD0 $1 $_ 02 {source: "users"} {filter: "status=active"}'
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_mixed_bare_and_param_roundtrip(self, compiler, decompiler):
        original = """@ 0xE3 $_ $_ 01 {error_var: "$5"}
@ 0x08 $4 $_ 02 {key: "results"} {value: "processed_data"}
@ 0x0F
@ 0x00"""
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)


class TestExampleFiles:
    """Compile each example file in examples/ directory."""

    def _example_files(self):
        if not EXAMPLES_DIR.exists():
            pytest.skip(f"Examples directory not found: {EXAMPLES_DIR}")
        files = sorted(EXAMPLES_DIR.glob("*.clank.txt"))
        if not files:
            pytest.skip("No example .clank.txt files found")
        return files

    def test_all_examples_compile(self, compiler):
        for path in self._example_files():
            text = path.read_text(encoding='utf-8')
            binary = compiler.compile(text)
            assert len(binary) >= 3, f"{path.name}: binary too short"
            assert binary[:2] == ClankerCompiler.MAGIC, f"{path.name}: bad magic"

    def test_all_examples_roundtrip(self, compiler, decompiler):
        for path in self._example_files():
            text = path.read_text(encoding='utf-8')
            binary = compiler.compile(text)
            recovered = decompiler.decompile(binary)
            assert normalize_text(recovered) == normalize_text(text), (
                f"Round-trip failed for {path.name}"
            )

    def test_binary_smaller_than_text(self, compiler):
        for path in self._example_files():
            text = path.read_text(encoding='utf-8')
            text_size = len(text.encode('utf-8'))
            binary = compiler.compile(text)
            binary_size = len(binary)
            ratio = binary_size / text_size
            assert binary_size < text_size, (
                f"{path.name}: binary ({binary_size}B) not smaller than "
                f"text ({text_size}B), ratio={ratio:.2f}"
            )


class TestVADUG:
    """VADUG emotional header encoding and decoding."""

    def test_vadug_compile(self, compiler):
        text = '@ 0xC1 $1 $2 01 {status: "500"} ![v:28 a:248 d:88 u:240 g:100]'
        binary = compiler.compile(text)
        # Find the VADUG marker 0xFE in the binary
        assert 0xFE in binary, "VADUG marker 0xFE not found in binary"

    def test_vadug_values(self, compiler):
        text = '@ 0xC1 $1 $2 01 {status: "500"} ![v:28 a:248 d:88 u:240 g:100]'
        binary = compiler.compile(text)
        # Find VADUG marker position
        idx = binary.index(0xFE)
        assert binary[idx + 1] == 28   # V
        assert binary[idx + 2] == 248  # A
        assert binary[idx + 3] == 88   # D
        assert binary[idx + 4] == 240  # U
        assert binary[idx + 5] == 100  # G

    def test_vadug_roundtrip(self, compiler, decompiler):
        original = '@ 0xC1 $1 $2 01 {status: "500"} ![v:28 a:248 d:88 u:240 g:100]'
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_vadug_extended_fields(self, compiler, decompiler):
        original = (
            '@ 0xC1 $1 $2 01 {status: "200"} '
            '![v:200 a:108 d:188 u:10 g:180 cert:220 src:150 goal:200 rel:180]'
        )
        binary = compiler.compile(original)
        recovered = decompiler.decompile(binary)
        assert normalize_text(recovered) == normalize_text(original)

    def test_vadug_marker_byte(self, compiler):
        text = '@ 0xC1 $1 $2 01 {status: "200"} ![v:128 a:128 d:128 u:128 g:128]'
        binary = compiler.compile(text)
        idx = binary.index(0xFE)
        # 0xFE followed by V A D U G cert src goal rel (9 bytes)
        assert len(binary) > idx + 9

    def test_no_vadug_when_absent(self, compiler):
        text = '@ 0xC1 $1 $2 01 {status: "200"}'
        binary = compiler.compile(text)
        assert 0xFE not in binary, "VADUG marker should not appear without annotation"


class TestErrorHandling:
    """Invalid input produces clear errors."""

    def test_param_count_mismatch(self, compiler):
        text = '@ 0x02 $0 $_ 03 {name: "x"} {value: "42"}'
        with pytest.raises(CompilerError, match="Declared 3 params but found 2"):
            compiler.compile(text)

    def test_invalid_var_reference(self, compiler):
        # $abc doesn't match the instruction regex (\$[\d_]+), so the line
        # is rejected as unrecognized format rather than an invalid var ref
        text = '@ 0x02 $abc $_ 00'
        with pytest.raises(CompilerError, match="Unrecognized instruction format"):
            compiler.compile(text)

    def test_unrecognized_line(self, compiler):
        text = 'not a valid instruction'
        with pytest.raises(CompilerError, match="Unrecognized instruction format"):
            compiler.compile(text)

    def test_decompile_bad_magic(self, decompiler):
        with pytest.raises(CompilerError, match="Invalid magic bytes"):
            decompiler.decompile(b'\x00\x00\x01')

    def test_decompile_too_short(self, decompiler):
        with pytest.raises(CompilerError, match="too short"):
            decompiler.decompile(b'\x43')

    def test_decompile_bad_version(self, decompiler):
        with pytest.raises(CompilerError, match="Unsupported binary version"):
            decompiler.decompile(b'\x43\x4C\x99')

    def test_var_slot_out_of_range(self, compiler):
        text = '@ 0x02 $999 $_ 00'
        with pytest.raises(CompilerError, match="out of range"):
            compiler.compile(text)


class TestCompressionRatio:
    """Verify binary is meaningfully smaller than text."""

    def test_hello_world_compression(self, compiler):
        text = """# hello_world.clank.txt
@ 0x10 $_ $_ 01 {text: "Hello World in Clank"}
@ 0x02 $0 $_ 02 {name: "greeting"} {value: "Hello, World!"}
@ 0x03 $0 $_ 01 {value: "greeting"}
@ 0x00 $_ $_ 01 {message: "program complete"}
"""
        text_size = len(text.encode('utf-8'))
        binary = compiler.compile(text)
        binary_size = len(binary)
        ratio = binary_size / text_size
        # Binary should be meaningfully smaller — opcode numbers, var refs,
        # and braces/colons all get compressed away
        assert ratio < 1.0, f"Compression ratio {ratio:.2f} — binary not smaller"
        print(f"Compression ratio: {ratio:.2%} ({binary_size}B / {text_size}B)")
