# PR #124 integration checkpoint — preserve full evaluation ancestry

The first exact-head PR CI exposed a pre-existing workflow interaction:
checkout requested full history, but the later additive-corpus gate fetched
`main` with `--depth=1`. This marked the measured production commit as a shallow
boundary and made its parent unavailable to the report-provenance regression.

The decoder's 51 native tests and both one-CPU, 64 MiB address-space-cap jobs
passed. The full suite correctly remained red. Python 3.12 recorded 2,949
passes, one failure, one intentional unpublished-baseline skip, and two
pre-existing expected V8 failures. The failed assertion was a `git rev-parse`
call for the parent of the measured production commit, not a changed corpus
label or failed decoder output.

The repair removes depth truncation from both additive-gate fetches and
asserts that the checkout remains non-shallow. A workflow regression guard
protects this requirement. No evaluator code, corpus generation, label, test
expectation, or production byte is changed to hide the failure.

Evidence:
- Initial PR head: `0366cb45113f99f28e67ab2e9296c35900911027`.
- Full CI run: https://github.com/deucebucket/clanker/actions/runs/34320988969
- Native decoder CI: https://github.com/deucebucket/clanker/actions/runs/34320989087
- Local repair check: 52 passes (51 decoder tests plus one workflow guard).

Final review and merge must use the repaired exact head, not the earlier run.
