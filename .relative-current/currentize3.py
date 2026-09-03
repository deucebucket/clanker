from __future__ import annotations

from pathlib import Path

path = Path(".relative-current/currentize.py")
source = path.read_text(encoding="utf-8")
old_guard = '''    if found != count:\n        raise RuntimeError(\n            f"Expected {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
new_guard = '''    if found < count:\n        raise RuntimeError(\n            f"Expected at least {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
if source.count(old_guard) != 1:
    raise RuntimeError("Could not patch currentize replacement guard")
source = source.replace(old_guard, new_guard, 1)

marker = 'compiled = compile(source, str(SOURCE), "exec")\n'
if source.count(marker) != 1:
    raise RuntimeError("Could not locate staged-source compile boundary")
extra = '''# Current #49 memory snapshots format the relation counter over three lines.\nswap(\n    ''' + "'''" + '''        memory._relation_counter = int(data.get("relation_counter", len(memory.relations)))\n        return memory\n''' + "'''" + ''',\n    ''' + "'''" + '''        memory._relation_counter = int(\n            data.get("relation_counter", len(memory.relations))\n        )\n        return memory\n''' + "'''" + ''',\n)\nswap(\n    ''' + "'''" + '''        memory._relation_counter = int(data.get("relation_counter", len(memory.relations)))\n        memory._modifier_counter = int(\n            data.get("modifier_counter", len(memory.modifiers))\n        )\n        return memory\n''' + "'''" + ''',\n    ''' + "'''" + '''        memory._relation_counter = int(\n            data.get("relation_counter", len(memory.relations))\n        )\n        memory._modifier_counter = int(\n            data.get("modifier_counter", len(memory.modifiers))\n        )\n        return memory\n''' + "'''" + ''',\n)\n\n'''
source = source.replace(marker, extra + marker, 1)
compiled = compile(source, str(path), "exec")
exec(compiled, {"__name__": "__main__"})
