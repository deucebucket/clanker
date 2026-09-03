from __future__ import annotations

from pathlib import Path


# Normalize the old staged implementation before executing it.
staged_path = Path("/tmp/relative-apply.py")
staged = staged_path.read_text(encoding="utf-8")

old_tokens = "segments = self._split_assertion_segments(tokens)"
if staged.count(old_tokens) != 2:
    raise RuntimeError(
        f"Expected two stale parser token references, found {staged.count(old_tokens)}"
    )
staged = staged.replace(
    old_tokens,
    "segments = self._split_assertion_segments(clean)",
)

obsolete_decorator = '''# The inserted RelativeSplit declaration needs its dataclass decorator.
replace_once(
    "clanker_lm/parser.py",
    ''' + "'''" + '''\n\nclass RelativeSplit:\n''' + "'''" + ''',
    ''' + "'''" + '''\n\n@dataclass\nclass RelativeSplit:\n''' + "'''" + ''',
)

'''
if staged.count(obsolete_decorator) != 1:
    raise RuntimeError("Could not locate stale RelativeSplit decorator transform")
staged_path.write_text(
    staged.replace(obsolete_decorator, "", 1),
    encoding="utf-8",
)

# Rebase the staged implementation onto the exact current-main seams.
wrapper_path = Path(".relative-current/currentize.py")
wrapper = wrapper_path.read_text(encoding="utf-8")

old_guard = '''    if found != count:\n        raise RuntimeError(\n            f"Expected {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
new_guard = '''    if found < count:\n        raise RuntimeError(\n            f"Expected at least {count} staged-source matches, found {found}: {old[:120]!r}"\n        )\n    source = source.replace(old, new, count)\n'''
if wrapper.count(old_guard) != 1:
    raise RuntimeError("Could not patch currentizer replacement guard")
wrapper = wrapper.replace(old_guard, new_guard, 1)

compile_marker = 'compiled = compile(source, str(SOURCE), "exec")\n'
if wrapper.count(compile_marker) != 1:
    raise RuntimeError("Could not locate staged implementation compile boundary")

memory_rebase = '''# Current #49 snapshots format the relation counter over multiple lines.\nswap(\n    ''' + "'''" + '''        memory._relation_counter = int(data.get("relation_counter", len(memory.relations)))\n        return memory\n''' + "'''" + ''',\n    ''' + "'''" + '''        memory._relation_counter = int(\n            data.get("relation_counter", len(memory.relations))\n        )\n        return memory\n''' + "'''" + ''',\n)\nswap(\n    ''' + "'''" + '''        memory._relation_counter = int(data.get("relation_counter", len(memory.relations)))\n        memory._modifier_counter = int(\n            data.get("modifier_counter", len(memory.modifiers))\n        )\n        return memory\n''' + "'''" + ''',\n    ''' + "'''" + '''        memory._relation_counter = int(\n            data.get("relation_counter", len(memory.relations))\n        )\n        memory._modifier_counter = int(\n            data.get("modifier_counter", len(memory.modifiers))\n        )\n        return memory\n''' + "'''" + ''',\n)\n\n'''
wrapper = wrapper.replace(
    compile_marker,
    memory_rebase + compile_marker,
    1,
)

old_hash_guard = '''if parser_text.count(old_hash) != 2:
    raise RuntimeError(f"Expected two nondeterministic relative IDs, found {parser_text.count(old_hash)}")
parser_text = parser_text.replace(old_hash, new_hash)
'''
new_hash_guard = '''import re as _currentize_re
parser_text, replacement_count = _currentize_re.subn(
    r"str\\(abs\\(hash\\(tuple\\(token\\.norm for token in items\\)\\)\\)\\)",
    'hashlib.sha256(" ".join(token.norm for token in items).encode("utf-8")).hexdigest()[:16]',
    parser_text,
)
if replacement_count != 2:
    raise RuntimeError(
        f"Expected two nondeterministic relative IDs, found {replacement_count}"
    )
'''
if wrapper.count(old_hash_guard) != 1:
    raise RuntimeError("Could not locate deterministic-ID replacement guard")
wrapper = wrapper.replace(old_hash_guard, new_hash_guard, 1)

compiled = compile(wrapper, str(wrapper_path), "exec")
exec(compiled, {"__name__": "__main__"})
