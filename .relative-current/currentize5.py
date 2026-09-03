from __future__ import annotations

from pathlib import Path

staged = Path("/tmp/relative-apply.py")
source = staged.read_text(encoding="utf-8")
old = "segments = self._split_assertion_segments(tokens)"
new = "segments = self._split_assertion_segments(clean)"
count = source.count(old)
if count != 2:
    raise RuntimeError(
        f"Expected two assertion-loop token references, found {count}"
    )
staged.write_text(source.replace(old, new), encoding="utf-8")

wrapper = Path(".relative-current/currentize4.py")
compiled = compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec")
exec(compiled, {"__name__": "__main__"})
