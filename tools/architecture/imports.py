"""Static dependency discovery; local imports count, unknown dynamic edges do not."""
from __future__ import annotations
import ast
from pathlib import Path


def module_name(path: str) -> str:
    name = path.removesuffix('.py').replace('/', '.')
    return name.removesuffix('.__init__')


def dependencies(root: Path, paths: list[str]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    modules = {module_name(p): p for p in paths}
    graph: dict[str, set[str]] = {name: set() for name in modules}
    hazards: dict[str, list[str]] = {}
    for name, path in modules.items():
        tree = ast.parse((root / path).read_text(encoding='utf-8'))
        package = name if path.endswith('/__init__.py') else name.rpartition('.')[0]
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == '*' for a in node.names):
                    hazards.setdefault(path, []).append(f'wildcard import at line {node.lineno}')
                if node.level:
                    pieces = package.split('.')
                    if node.level > len(pieces):
                        hazards.setdefault(path, []).append(f'invalid relative import at line {node.lineno}')
                        continue
                    prefix = '.'.join(pieces[:len(pieces) - node.level + 1])
                    base = prefix + ('.' + node.module if node.module else '')
                else:
                    base = node.module or ''
                for alias in node.names:
                    child = base + '.' + alias.name
                    targets.append(child if child in modules else base)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval'}:
                hazards.setdefault(path, []).append(f'dynamic code execution at line {node.lineno}')
            for target in targets:
                # Retain external names for layer enforcement; cycles only use
                # discovered modules. Prefix package initialization is not a
                # separate edge for every imported child module.
                if target and target != name:
                    graph[name].add(target)
    return graph, hazards


def cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan SCCs: module order cannot hide mutual imports."""
    counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    numbers: dict[str, int] = {}
    lows: dict[str, int] = {}
    result: list[list[str]] = []

    def visit(name: str) -> None:
        nonlocal counter
        numbers[name] = lows[name] = counter
        counter += 1
        stack.append(name)
        on_stack.add(name)
        for other in sorted(graph[name]):
            if other not in graph:
                continue
            if other not in numbers:
                visit(other)
                lows[name] = min(lows[name], lows[other])
            elif other in on_stack:
                lows[name] = min(lows[name], numbers[other])
        if lows[name] == numbers[name]:
            group = []
            while True:
                other = stack.pop()
                on_stack.remove(other)
                group.append(other)
                if other == name:
                    break
            if len(group) > 1:
                result.append(sorted(group))
    for name in sorted(graph):
        if name not in numbers:
            visit(name)
    return sorted(result)
