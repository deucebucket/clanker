"""Physical source and AST metrics, including large single-line payloads."""
from __future__ import annotations
import ast
from pathlib import Path


def source_metrics(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    tree = ast.parse(text, filename=str(path))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    largest = max(functions, key=lambda n: n.end_lineno - n.lineno, default=None)
    return {
        'lines': len(text.splitlines()), 'bytes': len(raw),
        'max_function_lines': largest.end_lineno - largest.lineno + 1 if largest else 0,
        'largest_function': largest.name if largest else None,
        'functions': len(functions),
    }


def inventory(root: Path, roots: list[str]) -> dict[str, dict]:
    paths: set[Path] = set()
    for directory in roots:
        base = root / directory
        if base.is_file():
            paths.add(base)
        elif base.is_dir():
            paths.update(base.rglob('*.py'))
    return {str(p.relative_to(root)): source_metrics(p) for p in sorted(paths)}
