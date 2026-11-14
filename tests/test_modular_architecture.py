"""Modularity is executable policy, not a filename convention."""
from __future__ import annotations
import ast
import copy
from dataclasses import fields, is_dataclass
import importlib
import json
from pathlib import Path
import pickle
from typing import get_type_hints

import pytest

from tools.architecture.checks import check
from tools.architecture.imports import cycles, dependencies
from tools.architecture.metrics import source_metrics

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return {'version': 1, 'roots': ['clanker_lm'],
            'defaults': {'lines': 40, 'bytes': 2000, 'max_function_lines': 10},
            'legacy': {},
            'layers': {'clanker_lm.semantic_types': ['clanker_lm.semantic_types'],
                       'clanker_lm.parsing': ['clanker_lm.parsing', 'clanker_lm.semantic_types']}}


def put(root, relative, text):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def test_actual_source_satisfies_architecture_contract():
    declared = json.loads((ROOT / 'architecture_policy.json').read_text())
    report = check(ROOT, declared)
    assert report['passed'], report['violations']
    assert all(not path.startswith(('clanker_lm/parsing/', 'clanker_lm/semantic_types/',
                                   'clanker_lm/answering/', 'clanker_lm/realization/'))
               for path in report['legacy_debt'])


@pytest.mark.parametrize('mode', ['lines', 'bytes', 'max_function_lines'])
def test_new_monolith_is_rejected_even_if_its_filename_is_modular(tmp_path, mode):
    text = {'lines': '# responsibility\n' * 41,
            'bytes': 'VALUE = ' + repr('x' * 2100),
            'max_function_lines': 'def task():\n' + '    pass\n' * 11}[mode]
    put(tmp_path, 'clanker_lm/parsing/innocent.py', text)
    report = check(tmp_path, policy())
    assert any(v['rule'] == mode for v in report['violations'])


def test_legacy_debt_cannot_grow_and_cannot_keep_an_obsolete_ceiling(tmp_path):
    path = 'clanker_lm/old.py'
    source = put(tmp_path, path, '# existing\n' * 50)
    p = policy()
    p['legacy'][path] = {'limits': {'lines': 50, 'bytes': 2000, 'max_function_lines': 10},
                         'reason': 'tracked extraction', 'issue': 161}
    assert check(tmp_path, p)['passed']
    source.write_text('# existing\n' * 51)
    assert any(v['rule'] == 'lines' for v in check(tmp_path, p)['violations'])
    source.write_text('# now smaller\n')
    assert any(v['rule'] == 'lower legacy ceiling after shrink' for v in check(tmp_path, p)['violations'])
    source.unlink()
    assert any(v['rule'] == 'remove retired legacy exception' for v in check(tmp_path, p)['violations'])


def test_debt_requires_an_explicit_reason_and_issue(tmp_path):
    put(tmp_path, 'clanker_lm/old.py', 'pass\n')
    p = policy()
    p['legacy']['clanker_lm/old.py'] = {'limits': dict(p['defaults'])}
    assert any('reason and issue' in v['rule'] for v in check(tmp_path, p)['violations'])


@pytest.mark.parametrize('statement', ['from ..runtime import ClankerLM',
                                      'import clanker_lm.runtime',
                                      'def late():\n    from ..runtime import ClankerLM'])
def test_data_layer_cannot_import_runtime_even_through_local_import(tmp_path, statement):
    put(tmp_path, 'clanker_lm/semantic_types/entity.py', statement + '\n')
    assert any(v['rule'] == 'forbidden dependency' for v in check(tmp_path, policy())['violations'])


def test_data_cannot_depend_on_parser_but_parser_can_depend_on_data(tmp_path):
    data = put(tmp_path, 'clanker_lm/semantic_types/item.py', 'pass\n')
    put(tmp_path, 'clanker_lm/parsing/step.py', 'from ..semantic_types.item import Record\n')
    assert check(tmp_path, policy())['passed']
    data.write_text('from ..parsing.step import parse\n')
    report = check(tmp_path, policy())
    assert any(v['rule'] == 'forbidden dependency' for v in report['violations'])
    assert any(v['rule'] == 'controlled module cycle' for v in report['violations'])


def test_relative_import_cycle_is_not_hidden_by_a_facade(tmp_path):
    put(tmp_path, 'clanker_lm/parsing/__init__.py', 'from .a import A\n')
    put(tmp_path, 'clanker_lm/parsing/a.py', 'from .b import B\n')
    put(tmp_path, 'clanker_lm/parsing/b.py', 'from .a import A\n')
    assert any(v['rule'] == 'controlled module cycle' for v in check(tmp_path, policy())['violations'])
    assert cycles({'a': {'b'}, 'b': {'external'}}) == []


@pytest.mark.parametrize('source,flag', [
    ('from .other import *\n', 'wildcard import'),
    ('exec("pass")\n', 'dynamic code execution'),
    ('eval("1")\n', 'dynamic code execution'),
])
def test_no_wildcard_namespace_or_runtime_code_injection(tmp_path, source, flag):
    put(tmp_path, 'clanker_lm/parsing/example.py', source)
    assert any(flag in v['rule'] for v in check(tmp_path, policy())['violations'])


def test_invalid_relative_import_is_reported(tmp_path):
    put(tmp_path, 'clanker_lm/parsing/example.py', 'from ....missing import thing\n')
    assert any('invalid relative import' in v['rule'] for v in check(tmp_path, policy())['violations'])


def test_public_model_exports_are_real_definitions_not_duplicate_copies():
    from clanker_lm import model
    names = []
    for path in (ROOT / 'clanker_lm/semantic_types').glob('*.py'):
        tree = ast.parse(path.read_text())
        mod = importlib.import_module('clanker_lm.semantic_types.' + path.stem)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                assert getattr(model, node.name) is getattr(mod, node.name)
                names.append(node.name)
                item = getattr(mod, node.name)
                if is_dataclass(item):
                    # Resolve postponed/forward annotations in the real defining
                    # module, not a wildcard namespace added just for tests.
                    get_type_hints(item)
    assert sorted(names) == sorted(model.__all__)
    assert len(names) == len(set(names))


def test_existing_public_imports_serialization_and_real_implementation_paths():
    from clanker_lm.model import EventFrame, SemanticRef, EntityKind
    from clanker_lm.parser import SemanticParser, NPResult
    from clanker_lm.qa import QuestionAnswerer
    from clanker_lm.realize import SurfaceRealizer, Part
    event = EventFrame('borrow', {'agent': SemanticRef.entity('mira_1', 'Mira'),
                                 'patient': SemanticRef.entity('car_2', 'car', EntityKind.THING)})
    assert EventFrame.from_dict(event.to_dict()).to_dict() == event.to_dict()
    assert pickle.loads(pickle.dumps(event)).to_dict() == event.to_dict()
    assert SemanticParser._parse_input.__module__ == 'clanker_lm.parsing.pipeline'
    assert NPResult.__module__ == 'clanker_lm.parsing.types'
    assert QuestionAnswerer.answer.__module__ == 'clanker_lm.answering.routing'
    assert SurfaceRealizer.realize.__module__ == 'clanker_lm.realization.routing'


def test_public_legacy_pickle_paths_still_resolve():
    # Trusted protocol-0 GLOBAL fixtures identify classes under the original
    # public modules. No untrusted pickle is accepted by the application here.
    from clanker_lm.model import EventFrame
    from clanker_lm.parser import NPResult
    assert pickle.loads(b'cclanker_lm.model\nEventFrame\n.') is EventFrame
    assert pickle.loads(b'cclanker_lm.parser\nNPResult\n.') is NPResult


def test_components_do_not_accidentally_override_each_others_methods():
    from clanker_lm.parser import SemanticParser
    from clanker_lm.qa import QuestionAnswerer
    from clanker_lm.realize import SurfaceRealizer
    for composed in (SemanticParser, QuestionAnswerer, SurfaceRealizer):
        owners = {}
        for component in composed.__bases__:
            for name, value in vars(component).items():
                if callable(value) or isinstance(value, (staticmethod, classmethod)):
                    if name.startswith('__') and name != '__init__':
                        continue
                    assert name not in owners, (composed.__name__, name, owners.get(name), component)
                    owners[name] = component


def test_handler_dispatch_is_ordered_and_really_executes(monkeypatch):
    from clanker_lm.parser import SemanticParser
    from clanker_lm.memory import ConversationMemory
    from clanker_lm.parsing import steps
    expected = ['embedded', 'gerund', 'infinitival', 'content', 'relative', 'appositive', 'subordinate', 'ordinary']
    assert [f.__module__.split('.')[-1] for f in steps.STAGES] == expected
    called = []
    def traced(fn):
        def run(parser, clause, connector, ctx):
            called.append((fn.__module__.split('.')[-1], id(ctx)))
            return fn(parser, clause, connector, ctx)
        return run
    monkeypatch.setattr(steps, 'STAGES', tuple(traced(f) for f in steps.STAGES))
    parsed = SemanticParser().parse('Sarah said John left.', ConversationMemory())
    assert parsed.understood and parsed.contents
    assert [n for n, _ in called] == expected[:4]
    assert len({i for _, i in called}) == 1


def test_dispatch_has_no_silent_fallback_if_its_terminal_stage_is_removed(monkeypatch):
    from clanker_lm.parser import SemanticParser
    from clanker_lm.memory import ConversationMemory
    from clanker_lm.parsing import steps
    monkeypatch.setattr(steps, 'STAGES', ())
    with pytest.raises(RuntimeError, match='no terminal handler'):
        SemanticParser().parse('Sarah called John.', ConversationMemory())


def test_subclass_extension_seam_and_call_local_state_remain_intact():
    from clanker_lm.parser import SemanticParser
    from clanker_lm.memory import ConversationMemory
    class Traced(SemanticParser):
        def _parse_clause(self, tokens, raw, memory):
            out = super()._parse_clause(tokens, raw, memory)
            out.diagnostics.append('subclass seam reached')
            return out
    parser = Traced()
    before = copy.deepcopy(vars(parser))
    a, b = ConversationMemory(), ConversationMemory()
    left = parser.parse('Sarah called John.', a)
    right = parser.parse('Mira bought a camera.', b)
    assert 'subclass seam reached' in left.diagnostics
    assert 'subclass seam reached' in right.diagnostics
    assert vars(parser) == before
    assert not b.find_by_alias('Sarah').resolved
    assert not a.find_by_alias('Mira').resolved
    assert left.events[0].arguments['agent'].surface == 'Sarah'


def test_new_source_contains_no_second_semantic_parser_definition():
    found = []
    for path in (ROOT / 'clanker_lm').rglob('*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name == 'SemanticParser':
                found.append(path.relative_to(ROOT).as_posix())
    assert found == ['clanker_lm/parser.py']
