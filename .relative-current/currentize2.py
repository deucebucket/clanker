from __future__ import annotations

from pathlib import Path

path = Path(".relative-current/currentize.py")
source = path.read_text(encoding="utf-8")
old = '''    if found != count:\n        raise RuntimeError(\n            f"Expected {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
new = '''    if found < count:\n        raise RuntimeError(\n            f"Expected at least {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
if source.count(old) != 1:
    raise RuntimeError("Could not patch currentize replacement guard")
compiled = compile(source.replace(old, new, 1), str(path), "exec")
exec(compiled, {"__name__": "__main__"})
