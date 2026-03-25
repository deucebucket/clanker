"""
Clank-Lang Validator — Validates Clank scripts against opcode definitions.

Checks for:
- Valid opcode references
- Required parameters present
- Block nesting depth and matching
- Variable slot references in range
"""

import re
import yaml
from pathlib import Path
from typing import List, Tuple, Optional


class ValidationError:
    """Represents a single validation error."""

    def __init__(self, line_num: int, message: str, severity: str = "error"):
        self.line_num = line_num
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __repr__(self):
        return f"[{self.severity.upper()}] line {self.line_num}: {self.message}"


class ClankValidator:
    """Validates Clank text-format scripts."""

    INSTRUCTION_RE = re.compile(
        r'^@\s+(0x[0-9A-Fa-f]{2})\s+(\$[\d_]+)\s+(\$[\d_]+)\s+(\d{2})\s*(.*)?$'
    )
    SIMPLE_RE = re.compile(r'^@\s+(0x[0-9A-Fa-f]{2})$')
    PARAM_RE = re.compile(r'\{(\w+):\s*"([^"]*)"\}|\{(\w+):\s*([^}]+)\}')

    MAX_NESTING_DEPTH = 4
    BLOCK_OPENERS = {"0x0B", "0xC0", "0xE0", "0xE1", "0xE2", "0xE3"}

    def __init__(self, opcode_dir: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            opcode_dir: Path to the opcodes/ directory. If None, uses default.
        """
        if opcode_dir:
            self._opcode_dir = Path(opcode_dir)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            self._opcode_dir = project_root / "opcodes"

        self._opcodes = self._load_opcodes()

    def _load_opcodes(self) -> dict:
        """Load all opcode definitions from YAML files."""
        opcodes = {}
        if not self._opcode_dir.exists():
            return opcodes

        for f in self._opcode_dir.glob("*.yaml"):
            if f.name.startswith("_"):
                continue
            with open(f, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if data and "opcodes" in data:
                for code, defn in data["opcodes"].items():
                    opcodes[code] = defn
        return opcodes

    def validate(self, clank_text: str) -> List[ValidationError]:
        """
        Validate a Clank text-format script.

        Args:
            clank_text: The Clank script to validate.

        Returns:
            List of ValidationError objects. Empty list means valid.
        """
        errors = []
        lines = clank_text.strip().split("\n")
        nesting_depth = 0
        block_stack = []

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            match = self.INSTRUCTION_RE.match(line)
            simple = self.SIMPLE_RE.match(line) if not match else None

            if not match and not simple:
                errors.append(ValidationError(
                    line_num, f"Invalid instruction format: {line}"
                ))
                continue

            if match:
                opcode = match.group(1).upper().replace("X", "x")
                target = match.group(2)
                src = match.group(3)
                param_count = int(match.group(4))
                param_str = match.group(5) or ""
                params = self._parse_params(param_str)
            else:
                opcode = simple.group(1).upper().replace("X", "x")
                target = "$_"
                src = "$_"
                param_count = 0
                params = {}

            # Check opcode exists
            if opcode not in self._opcodes:
                errors.append(ValidationError(
                    line_num, f"Unknown opcode: {opcode}"
                ))
                continue

            defn = self._opcodes[opcode]

            # Check required params
            if "params" in defn:
                for p in defn["params"]:
                    if p.get("required") and p["name"] not in params:
                        errors.append(ValidationError(
                            line_num,
                            f"Missing required parameter '{p['name']}' for {defn['name']}"
                        ))

            # Check param count matches
            actual_count = len(params)
            if actual_count != param_count:
                errors.append(ValidationError(
                    line_num,
                    f"Declared param count {param_count:02d} but found {actual_count} params",
                    severity="warning"
                ))

            # Check variable slots
            for var in [target, src]:
                if var != "$_":
                    slot_match = re.match(r'^\$(\d+)$', var)
                    if slot_match:
                        slot = int(slot_match.group(1))
                        if slot > 31:
                            errors.append(ValidationError(
                                line_num,
                                f"Variable slot {var} out of range (max $31)"
                            ))

            # Handle block nesting
            if opcode == "0x0F":
                if nesting_depth == 0:
                    errors.append(ValidationError(
                        line_num, "Unmatched END: no block to close"
                    ))
                else:
                    nesting_depth -= 1
                    block_stack.pop()
            elif opcode in self.BLOCK_OPENERS:
                nesting_depth += 1
                if nesting_depth > self.MAX_NESTING_DEPTH:
                    errors.append(ValidationError(
                        line_num,
                        f"Block nesting depth {nesting_depth} exceeds maximum of {self.MAX_NESTING_DEPTH}"
                    ))
                block_stack.append((line_num, opcode))

        # Check for unclosed blocks
        for open_line, open_op in block_stack:
            errors.append(ValidationError(
                open_line,
                f"Unclosed block opened by {self._opcodes.get(open_op, {}).get('name', open_op)}"
            ))

        return errors

    def _parse_params(self, param_str: str) -> dict:
        """Parse parameter string into a dict."""
        params = {}
        for match in self.PARAM_RE.finditer(param_str):
            if match.group(1):
                params[match.group(1)] = match.group(2)
            elif match.group(3):
                params[match.group(3)] = match.group(4).strip()
        return params
