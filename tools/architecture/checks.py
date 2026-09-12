"""Explicit layering plus non-growth limits; metrics are not a cohesion proof."""
from __future__ import annotations
from pathlib import Path
from .metrics import inventory
from .imports import cycles, dependencies, module_name


def under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + '.')


def check(root: Path, policy: dict) -> dict:
    if policy.get('version') != 1:
        raise ValueError('unsupported architecture policy')
    measured = inventory(root, policy['roots'])
    graph, hazards = dependencies(root, list(measured))
    violations: list[dict] = []
    for path, values in measured.items():
        limits = policy['legacy'].get(path, {}).get('limits', policy['defaults'])
        for metric in ('lines', 'bytes', 'max_function_lines'):
            if values[metric] > limits[metric]:
                violations.append({'path': path, 'rule': metric, 'actual': values[metric], 'limit': limits[metric]})
    for path, entry in policy['legacy'].items():
        if path not in measured:
            violations.append({'path': path, 'rule': 'remove retired legacy exception'})
            continue
        if not entry.get('reason') or not entry.get('issue'):
            violations.append({'path': path, 'rule': 'legacy debt must have reason and issue'})
        for metric, cap in entry['limits'].items():
            if cap > policy['defaults'][metric] and measured[path][metric] < cap:
                violations.append({'path': path, 'rule': 'lower legacy ceiling after shrink', 'metric': metric,
                                   'actual': measured[path][metric], 'limit': cap})
    for prefix, allowed in policy['layers'].items():
        for name, targets in graph.items():
            if not under(name, prefix):
                continue
            for target in sorted(targets):
                if target.startswith(('clanker_lm', 'engine', 'tools')) and not any(under(target, p) for p in allowed):
                    violations.append({'path': name, 'rule': 'forbidden dependency', 'target': target})
    controlled = tuple(policy['layers'])
    for cycle in cycles(graph):
        if any(under(n, p) for n in cycle for p in controlled):
            violations.append({'path': cycle, 'rule': 'controlled module cycle'})
    for path, found in hazards.items():
        if any(under(module_name(path), p) for p in controlled):
            violations.extend({'path': path, 'rule': item} for item in found)
    return {'version': 1, 'passed': not violations, 'violations': violations,
            'inventory': measured, 'legacy_debt': policy['legacy'],
            'dependencies': {k: sorted(v) for k, v in sorted(graph.items())}}
