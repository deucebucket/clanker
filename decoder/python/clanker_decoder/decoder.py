"""
Clanker-Lang Decoder — Core decoding logic.

Takes Clanker text-format scripts (@ lines) and decodes them to any
loaded dictionary language using template substitution.
"""

import re
from typing import Optional
from .loader import DictionaryLoader


class ClankerDecoder:
    """Decodes Clanker text-format scripts using dictionary lookup."""

    # Regex for parsing a Clank instruction line
    # Format: @ <opcode> <target> <src> <param_count> {key: "value"} ...
    INSTRUCTION_RE = re.compile(
        r'^@\s+(0x[0-9A-Fa-f]{2})\s+(\$[\d_]+)\s+(\$[\d_]+)\s+(\d{2})\s*(.*)?$'
    )

    # Regex for extracting {key: "value"} or {key: value} parameters
    PARAM_RE = re.compile(
        r'\{(\w+):\s*"([^"]*)"\}|\{(\w+):\s*([^}]+)\}'
    )

    def __init__(self, loader: Optional[DictionaryLoader] = None):
        """
        Initialize the decoder.

        Args:
            loader: A DictionaryLoader instance. If None, creates a default one.
        """
        self.loader = loader or DictionaryLoader()

    def decode(self, clank_text: str, lang: str) -> str:
        """
        Decode a Clanker text-format script to the specified language.

        Args:
            clank_text: Clank script as a string (lines starting with @).
            lang: Target language code (e.g., 'en', 'zh', 'python').

        Returns:
            Decoded output as a string.
        """
        dictionary = self.loader.load(lang)
        raw_opcodes = dictionary.get("opcodes", {})
        # Normalize keys: YAML may parse 0x00 as int 0, so convert all keys to "0xNN" strings
        opcodes = {}
        for k, v in raw_opcodes.items():
            if isinstance(k, int):
                opcodes[f"0x{k:02X}"] = v
            else:
                # Normalize string keys like "0xc0" -> "0xC0"
                key_int = int(str(k), 16)
                opcodes[f"0x{key_int:02X}"] = v

        lines = clank_text.strip().split("\n")
        output_lines = []
        indent_level = 0

        # Track which opcodes open blocks (for indentation)
        block_openers = {"0xE0", "0xE1", "0xE2", "0xE3", "0x0B", "0xC0"}

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            match = self.INSTRUCTION_RE.match(line)
            if not match:
                # Try simpler format: @ <opcode> (no params)
                simple_match = re.match(r'^@\s+(0x[0-9A-Fa-f]{2})$', line)
                if simple_match:
                    opcode = simple_match.group(1)
                    params = {}
                else:
                    continue
            else:
                opcode = match.group(1)
                target = match.group(2)
                src = match.group(3)
                param_count = int(match.group(4))
                param_str = match.group(5) or ""
                params = self._parse_params(param_str)

            # Normalize opcode to "0xNN" format with uppercase hex digits
            opcode_int = int(opcode, 16)
            opcode = f"0x{opcode_int:02X}"

            # Handle END — decrease indent before rendering
            if opcode == "0x0F":
                indent_level = max(0, indent_level - 1)

            # Look up the opcode in the dictionary
            entry = opcodes.get(opcode)
            if entry is None:
                output_lines.append("  " * indent_level + f"[unknown opcode: {opcode}]")
                continue

            # Choose template: with_param if optional params are present and with_param exists
            template = entry.get("template", f"[{opcode}]")
            if params and "with_param" in entry:
                # Use with_param if it references any param that we have
                wp = entry["with_param"]
                # Check if with_param template uses any of our params that aren't in the base template
                with_param_keys = set(re.findall(r'\{(\w+)\}', wp))
                base_keys = set(re.findall(r'\{(\w+)\}', template))
                extra_keys = with_param_keys - base_keys
                if extra_keys and any(k in params for k in extra_keys):
                    template = wp

            # Substitute parameters into template
            rendered = self._render_template(template, params)
            output_lines.append("  " * indent_level + rendered)

            # Handle block openers — increase indent after rendering
            if opcode in block_openers:
                indent_level += 1

        return "\n".join(output_lines)

    def _parse_params(self, param_str: str) -> dict:
        """
        Parse parameter string into a dict.

        Handles formats:
            {key: "value"}   -> {'key': 'value'}
            {key: value}     -> {'key': 'value'}
        """
        params = {}
        for match in self.PARAM_RE.finditer(param_str):
            if match.group(1):
                # {key: "value"} form
                params[match.group(1)] = match.group(2)
            elif match.group(3):
                # {key: value} form
                params[match.group(3)] = match.group(4).strip()

        return params

    def _render_template(self, template: str, params: dict) -> str:
        """
        Substitute {param_name} placeholders in a template with actual values.

        Unresolved placeholders are left as-is.
        """
        result = template
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


# Module-level convenience function
_default_decoder = None


def decode(clank_text: str, lang: str, loader: Optional[DictionaryLoader] = None) -> str:
    """
    Decode a Clanker text-format script to the specified language.

    Convenience function that uses a module-level decoder instance.

    Args:
        clank_text: Clank script as a string (lines starting with @).
        lang: Target language code (e.g., 'en', 'zh', 'python').
        loader: Optional custom DictionaryLoader.

    Returns:
        Decoded output as a string.
    """
    global _default_decoder
    if loader:
        decoder = ClankerDecoder(loader)
        return decoder.decode(clank_text, lang)

    if _default_decoder is None:
        _default_decoder = ClankerDecoder()
    return _default_decoder.decode(clank_text, lang)
