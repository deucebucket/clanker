"""Run: python -m tools.architecture [--root PATH] [--output JSON]."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .checks import check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    policy = json.loads((root / 'architecture_policy.json').read_text(encoding='utf-8'))
    report = check(root, policy)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'violations': report['violations'],
                      'source_modules': len(report['inventory']),
                      'legacy_debt_files': len(report['legacy_debt'])}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
