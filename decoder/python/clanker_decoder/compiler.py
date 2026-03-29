"""
Clanker-Lang Compiler — Binary format compiler and decompiler.

Converts between human-readable .clank.txt text format and compact .clank
binary bytecode.

Text format (input):
    @ 0x02 $0 $_ 02 {name: "x"} {value: "42"}
    @ 0x04 $0 $_ 01 {message: "hello"}
    @ 0x00

Binary format (output):
    [MAGIC] [VERSION] [INSTRUCTION]... [0x00]

    MAGIC:   0x434C ("CL" for ClankerLang)
    VERSION: 0x01

    Each instruction:
        [OPCODE:      1 byte]
        [TARGET_VAR:  1 byte]  (0-255, 0xFF for $_)
        [SOURCE_VAR:  1 byte]  (0-255, 0xFF for $_)
        [PARAM_COUNT: 1 byte]
        [PARAMS: variable length]
            Each param:
                [KEY_LENGTH:   1 byte]
                [KEY:          N bytes]
                [VALUE_LENGTH: 2 bytes big-endian]
                [VALUE:        N bytes]

    VADUG emotional header (optional, per instruction):
        [0xFE] [V] [A] [D] [U] [G] [CERT] [SRC] [GOAL] [REL]
        0xFE = emotional header marker, followed by 9 bytes.
"""

import re
import struct
from pathlib import Path
from typing import List, Optional, Tuple


class CompilerError(Exception):
    """Raised when compilation encounters invalid input."""

    def __init__(self, message: str, line_num: int = 0, line_text: str = ""):
        self.line_num = line_num
        self.line_text = line_text
        super().__init__(
            f"Line {line_num}: {message}" if line_num else message
        )


class ClankerCompiler:
    """Compile .clank.txt to .clank binary format."""

    MAGIC = b'\x43\x4C'  # "CL"
    VERSION = 0x01

    # Regex for full instruction line
    INSTRUCTION_RE = re.compile(
        r'^@\s+(0x[0-9A-Fa-f]{2})\s+(\$[\d_]+)\s+(\$[\d_]+)\s+(\d{2})\s*(.*)?$'
    )

    # Regex for bare instruction (opcode only, no params)
    BARE_INSTRUCTION_RE = re.compile(
        r'^@\s+(0x[0-9A-Fa-f]{2})\s*$'
    )

    # Regex for extracting {key: "value"} or {key: value} parameters
    PARAM_RE = re.compile(
        r'\{(\w+):\s*"([^"]*)"\}|\{(\w+):\s*([^}]+)\}'
    )

    # Regex for VADUG emotional annotation: ![v:28 a:248 d:88 u:240 g:100]
    VADUG_RE = re.compile(
        r'!\[v:(\d+)\s+a:(\d+)\s+d:(\d+)\s+u:(\d+)\s+g:(\d+)\]'
    )

    # Regex for VADUG with extended fields (cert, src, goal, rel)
    VADUG_EXT_RE = re.compile(
        r'!\[v:(\d+)\s+a:(\d+)\s+d:(\d+)\s+u:(\d+)\s+g:(\d+)'
        r'(?:\s+cert:(\d+)\s+src:(\d+)\s+goal:(\d+)\s+rel:(\d+))?\]'
    )

    VADUG_MARKER = 0xFE

    def compile(self, text: str) -> bytes:
        """Compile text-format Clanker to binary bytes."""
        output = bytearray()
        output.extend(self.MAGIC)
        output.append(self.VERSION)

        lines = text.strip().split('\n')
        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            instruction_bytes = self._compile_instruction(line, line_num)
            output.extend(instruction_bytes)

        return bytes(output)

    def compile_file(self, input_path: str, output_path: str = None):
        """Compile a .clank.txt file to .clank binary.

        Args:
            input_path: Path to the .clank.txt source file.
            output_path: Path for the output .clank file. If None, replaces
                         .clank.txt extension with .clank.
        """
        if output_path is None:
            output_path = input_path.replace('.clank.txt', '.clank')

        source = Path(input_path).read_text(encoding='utf-8')
        binary = self.compile(source)
        Path(output_path).write_bytes(binary)
        return output_path

    def _compile_instruction(self, line: str, line_num: int) -> bytes:
        """Compile a single instruction line to binary bytes."""
        output = bytearray()

        # Try full instruction format first
        match = self.INSTRUCTION_RE.match(line)
        if match:
            opcode = int(match.group(1), 16)
            target = self._parse_var(match.group(2), line_num)
            source = self._parse_var(match.group(3), line_num)
            param_count = int(match.group(4))
            param_str = match.group(5) or ""

            # Check for VADUG annotation
            vadug = self._parse_vadug(param_str)

            # Parse parameters (strip VADUG annotation from param string)
            clean_param_str = self.VADUG_EXT_RE.sub('', param_str).strip()
            params = self._parse_params(clean_param_str, line_num)

            if len(params) != param_count:
                raise CompilerError(
                    f"Declared {param_count} params but found {len(params)}",
                    line_num, line
                )

            output.append(opcode)
            output.append(target)
            output.append(source)
            output.append(param_count)

            for key, value in params:
                output.extend(self._encode_param(key, value, line_num))

            # Append VADUG header if present
            if vadug is not None:
                output.extend(self._encode_vadug(vadug))

            return bytes(output)

        # Try bare instruction (opcode only)
        bare_match = self.BARE_INSTRUCTION_RE.match(line)
        if bare_match:
            opcode = int(bare_match.group(1), 16)
            output.append(opcode)
            output.append(0xFF)  # target: $_
            output.append(0xFF)  # source: $_
            output.append(0x00)  # param_count: 0
            return bytes(output)

        raise CompilerError(
            f"Unrecognized instruction format: {line!r}",
            line_num, line
        )

    def _parse_var(self, var_str: str, line_num: int) -> int:
        """Parse a variable reference ($N or $_) to a byte value."""
        if var_str == '$_':
            return 0xFF
        try:
            slot = int(var_str[1:])  # strip the '$'
        except ValueError:
            raise CompilerError(
                f"Invalid variable reference: {var_str!r}",
                line_num
            )
        if slot < 0 or slot > 254:
            raise CompilerError(
                f"Variable slot {slot} out of range (0-254)",
                line_num
            )
        return slot

    def _parse_params(
        self, param_str: str, line_num: int
    ) -> List[Tuple[str, str]]:
        """Parse parameter string into ordered list of (key, value) tuples."""
        params = []
        for match in self.PARAM_RE.finditer(param_str):
            if match.group(1):
                # {key: "value"} form
                params.append((match.group(1), match.group(2)))
            elif match.group(3):
                # {key: value} form
                params.append((match.group(3), match.group(4).strip()))
        return params

    def _parse_vadug(self, param_str: str) -> Optional[tuple]:
        """Parse VADUG annotation from param string, if present.

        Returns a tuple of (V, A, D, U, G, cert, src, goal, rel) or None.
        Extended fields default to 0 if not present.
        """
        match = self.VADUG_EXT_RE.search(param_str)
        if not match:
            return None
        v = int(match.group(1))
        a = int(match.group(2))
        d = int(match.group(3))
        u = int(match.group(4))
        g = int(match.group(5))
        cert = int(match.group(6)) if match.group(6) else 0
        src = int(match.group(7)) if match.group(7) else 0
        goal = int(match.group(8)) if match.group(8) else 0
        rel = int(match.group(9)) if match.group(9) else 0
        return (v, a, d, u, g, cert, src, goal, rel)

    def _encode_param(self, key: str, value: str, line_num: int) -> bytes:
        """Encode a single parameter as binary."""
        output = bytearray()
        key_bytes = key.encode('utf-8')
        value_bytes = value.encode('utf-8')

        if len(key_bytes) > 255:
            raise CompilerError(
                f"Parameter key too long ({len(key_bytes)} bytes, max 255)",
                line_num
            )
        if len(value_bytes) > 65535:
            raise CompilerError(
                f"Parameter value too long ({len(value_bytes)} bytes, max 65535)",
                line_num
            )

        output.append(len(key_bytes))
        output.extend(key_bytes)
        output.extend(struct.pack('>H', len(value_bytes)))
        output.extend(value_bytes)
        return bytes(output)

    def _encode_vadug(self, vadug: tuple) -> bytes:
        """Encode a VADUG emotional header.

        Format: [0xFE] [V] [A] [D] [U] [G] [CERT] [SRC] [GOAL] [REL]
        """
        output = bytearray()
        output.append(self.VADUG_MARKER)
        for val in vadug:
            if val < 0 or val > 255:
                raise CompilerError(
                    f"VADUG value {val} out of range (0-255)"
                )
            output.append(val)
        return bytes(output)


class ClankerDecompiler:
    """Decompile .clank binary back to .clank.txt text format."""

    MAGIC = ClankerCompiler.MAGIC
    VADUG_MARKER = ClankerCompiler.VADUG_MARKER

    def decompile(self, data: bytes) -> str:
        """Decompile binary bytes to text format.

        Args:
            data: Raw binary .clank bytes.

        Returns:
            Text-format Clanker script.
        """
        pos = 0

        # Validate magic bytes
        if len(data) < 3:
            raise CompilerError("Binary data too short (missing header)")
        if data[0:2] != self.MAGIC:
            raise CompilerError(
                f"Invalid magic bytes: expected 0x434C, got "
                f"0x{data[0]:02X}{data[1]:02X}"
            )

        version = data[2]
        if version != 0x01:
            raise CompilerError(
                f"Unsupported binary version: {version}"
            )
        pos = 3

        lines = []
        while pos < len(data):
            line, pos = self._decompile_instruction(data, pos)
            lines.append(line)

        return '\n'.join(lines) + '\n'

    def _decompile_instruction(
        self, data: bytes, pos: int
    ) -> Tuple[str, int]:
        """Decompile a single instruction from binary data.

        Returns:
            Tuple of (text line, new position).
        """
        if pos >= len(data):
            raise CompilerError("Unexpected end of binary data")

        opcode = data[pos]
        pos += 1

        if pos + 2 >= len(data):
            raise CompilerError(
                f"Truncated instruction at opcode 0x{opcode:02X}"
            )

        target = data[pos]
        pos += 1
        source = data[pos]
        pos += 1
        param_count = data[pos]
        pos += 1

        # Parse parameters
        params = []
        for _ in range(param_count):
            key, value, pos = self._decode_param(data, pos)
            params.append((key, value))

        # Check for VADUG emotional header
        vadug = None
        if pos < len(data) and data[pos] == self.VADUG_MARKER:
            vadug, pos = self._decode_vadug(data, pos)

        # Format the text line
        target_str = '$_' if target == 0xFF else f'${target}'
        source_str = '$_' if source == 0xFF else f'${source}'

        # Check if this is a bare instruction (no params, $_ $_, and
        # opcode like 0x00 or 0x0F commonly written bare)
        if param_count == 0 and target == 0xFF and source == 0xFF and vadug is None:
            line = f'@ 0x{opcode:02X}'
        else:
            param_strs = []
            for key, value in params:
                param_strs.append(f'{{{key}: "{value}"}}')
            parts = [
                f'@ 0x{opcode:02X}',
                target_str,
                source_str,
                f'{param_count:02d}',
            ]
            if param_strs:
                parts.append(' '.join(param_strs))

            line = ' '.join(parts)

            if vadug is not None:
                v, a, d, u, g, cert, src, goal, rel = vadug
                vadug_str = f'![v:{v} a:{a} d:{d} u:{u} g:{g}'
                if any(x != 0 for x in (cert, src, goal, rel)):
                    vadug_str += (
                        f' cert:{cert} src:{src} goal:{goal} rel:{rel}'
                    )
                vadug_str += ']'
                line += f' {vadug_str}'

        return line, pos

    def _decode_param(
        self, data: bytes, pos: int
    ) -> Tuple[str, str, int]:
        """Decode a single parameter from binary data.

        Returns:
            Tuple of (key, value, new position).
        """
        if pos >= len(data):
            raise CompilerError("Unexpected end of data while reading param key length")

        key_len = data[pos]
        pos += 1

        if pos + key_len > len(data):
            raise CompilerError("Unexpected end of data while reading param key")
        key = data[pos:pos + key_len].decode('utf-8')
        pos += key_len

        if pos + 2 > len(data):
            raise CompilerError("Unexpected end of data while reading param value length")
        value_len = struct.unpack('>H', data[pos:pos + 2])[0]
        pos += 2

        if pos + value_len > len(data):
            raise CompilerError("Unexpected end of data while reading param value")
        value = data[pos:pos + value_len].decode('utf-8')
        pos += value_len

        return key, value, pos

    def _decode_vadug(
        self, data: bytes, pos: int
    ) -> Tuple[tuple, int]:
        """Decode a VADUG emotional header from binary data.

        Returns:
            Tuple of ((V,A,D,U,G,cert,src,goal,rel), new position).
        """
        if pos >= len(data) or data[pos] != self.VADUG_MARKER:
            raise CompilerError("Expected VADUG marker 0xFE")
        pos += 1

        if pos + 9 > len(data):
            raise CompilerError("Truncated VADUG header")

        values = tuple(data[pos:pos + 9])
        pos += 9
        return values, pos
