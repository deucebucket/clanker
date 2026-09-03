from __future__ import annotations

from pathlib import Path

staged = Path("/tmp/relative-apply.py")
source = staged.read_text(encoding="utf-8")
obsolete = '''# The inserted RelativeSplit declaration needs its dataclass decorator.
replace_once(
    "clanker_lm/parser.py",
    ''' + "'''" + '''\n\nclass RelativeSplit:\n''' + "'''" + ''',
    ''' + "'''" + '''\n\n@dataclass\nclass RelativeSplit:\n''' + "'''" + ''',
)

'''
if source.count(obsolete) != 1:
    raise RuntimeError(
        "Could not locate obsolete RelativeSplit decorator transform"
    )
staged.write_text(source.replace(obsolete, "", 1), encoding="utf-8")

wrapper = Path(".relative-current/currentize3.py")
compiled = compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec")
exec(compiled, {"__name__": "__main__"})
