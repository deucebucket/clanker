"""Best-first activation using the SAME task/plane gates as ActivationIndex.

The strongest path wins, never a sum of repeated support. A lower-salience,
shallower path may still expand because it has more remaining depth. Queue,
edge and node work remain bounded. Labels and memory payloads are absent here.
"""
from __future__ import annotations
from dataclasses import asdict
import heapq
from types import MappingProxyType
from typing import Mapping
from ..activation import ActivationIndex, Policy, Request, canonical, fingerprint
from .types import SCHEMA, SCALE, WaveConfig, gain


class WaveIndex:
    def __init__(self, graph: ActivationIndex, *, strengths: Mapping[str, int] | None = None):
        if not isinstance(graph, ActivationIndex):
            raise TypeError('existing ActivationIndex required')
        supplied = dict(strengths or {})
        if supplied.keys() - graph.links.keys():
            raise ValueError('unbound edge weight')
        self.graph = graph
        self.strengths = MappingProxyType({k: gain(supplied.get(k, SCALE)) for k in graph.links})
        self.generation = fingerprint({'graph': graph.generation,
                                       'strengths': dict(self.strengths), 'schema': SCHEMA})

    def select(self, request: Request, policy: Policy, config: WaveConfig | None = None) -> dict:
        self.graph.validate_request(request, policy)  # Before consulting any root.
        cfg = WaveConfig() if config is None else config
        if not isinstance(cfg, WaveConfig):
            raise TypeError('typed wave configuration required')
        queue = []
        best = {}  # Candidate mode/key/depth -> strongest encountered path.
        expanded = {}  # key -> non-dominated (depth, salience) expansions.
        active, refs, scores, depths = set(), set(), {}, {}
        steps, reached, stops = [], set(), set()
        work = {'edge_pings': 0, 'queue_pops': 0, 'expansions': 0,
                'peak_queue': 0, 'dominated_paths': 0, 'salience_pruned': 0,
                'policy_blocked': 0, 'payloads_read': 0}

        def record(action, **fields):
            steps.append({'step': len(steps), 'action': action, **fields})

        def offer(key, depth, score, mode, source='', edge=''):
            marker = (mode, key, depth)
            if best.get(marker, -1) >= score:
                return False
            best[marker] = score
            heapq.heappush(queue, (-score, depth, key, mode, source, edge))
            work['peak_queue'] = max(work['peak_queue'], len(queue))
            return True

        for root in sorted(request.roots):
            offer(root, 0, SCALE, 'active')
            record('seed', target=root, depth=0, salience=SCALE)
        while queue:
            neg_score, depth, node, mode, source, via = heapq.heappop(queue)
            score = -neg_score
            work['queue_pops'] += 1
            if best.get((mode, node, depth)) != score:
                work['dominated_paths'] += 1
                continue
            if mode == 'reference_only' and node in active:
                if via:
                    reached.add(via)
                continue
            old_expansions = expanded.get(node, []) if mode == 'active' else []
            if any(d <= depth and s >= score for d, s in old_expansions):
                work['dominated_paths'] += 1
                if via:
                    reached.add(via)
                record('pruned', target=node, source=source, edge=via, depth=depth,
                       salience=score, reason='dominated_path_not_additional_evidence')
                continue
            if node not in active and node not in refs and len(active) + len(refs) >= request.max_nodes:
                stops.add('node_budget')
                record('pruned', target=node, source=source, edge=via, depth=depth,
                       salience=score, reason='node_budget')
                continue
            if via:
                reached.add(via)
            scores[node] = max(score, scores.get(node, 0))
            depths[node] = min(depth, depths.get(node, depth))
            if mode == 'reference_only':
                refs.add(node)
                record('reference_only', target=node, source=source, edge=via,
                       depth=depth, salience=score, reason='context_label_only_no_expansion')
                continue
            active.add(node)
            refs.discard(node)
            expanded[node] = [(d, s) for d, s in old_expansions
                              if not (depth <= d and score >= s)] + [(depth, score)]
            work['expansions'] += 1
            record('expanded', target=node, source=source, edge=via, depth=depth, salience=score)
            for edge in self.graph.adjacency[node]:
                if work['edge_pings'] >= request.max_edges:
                    stops.add('edge_budget')
                    queue.clear()
                    record('stopped', source=node, reason='edge_budget')
                    break
                work['edge_pings'] += 1
                target = edge.target if edge.source == node else edge.source
                target_mode, reason = self.graph.edge_policy(node, edge, request.task)
                fields = dict(source=node, edge=edge.key, target=target, depth=depth + 1, edge_ping=True)
                if target_mode is None:
                    work['policy_blocked'] += 1
                    record('blocked', **fields, reason=reason)
                    continue
                if depth >= request.max_depth:
                    if target not in active and target not in refs:
                        stops.add('depth_budget')
                    record('pruned', **fields, reason='depth_budget')
                    continue
                strength = self.strengths[edge.key]
                candidate = score * strength * cfg.decay // (SCALE * SCALE)
                if candidate < cfg.minimum_salience:
                    work['salience_pruned'] += 1
                    record('pruned', **fields, reason='below_salience_floor',
                           salience=candidate, source_salience=score, edge_strength=strength)
                    continue
                added = offer(target, depth + 1, candidate, target_mode, node, edge.key)
                record('queued' if added else 'pruned', **fields,
                       reason=reason if added else 'duplicate_path_not_additional_evidence',
                       mode=target_mode, salience=candidate,
                       source_salience=score, edge_strength=strength)
        selected = active | refs
        reached = {k for k in reached if self.graph.links[k].source in selected
                   and self.graph.links[k].target in selected}
        result = {'schema': SCHEMA, 'context': self.graph.context,
                  'graph_generation': self.graph.generation, 'wave_generation': self.generation,
                  'policy': asdict(policy), 'request': asdict(request), 'config': asdict(cfg),
                  'active': sorted(active), 'reference_only': sorted(refs),
                  'salience': {k: scores[k] for k in sorted(selected)},
                  'depth': {k: depths[k] for k in sorted(selected)},
                  'traversed_edges': sorted(reached), 'steps': steps, 'work': work,
                  'status': 'incomplete' if stops else 'complete_within_declared_selection_policy',
                  'stop_reasons': sorted(stops), 'context_edges_are_proof_premises': False,
                  'salience_is_confidence': False, 'selection_is_exhaustive_truth_search': False}
        result['digest'] = fingerprint(result)
        return result

    def verify(self, receipt, request, policy, config=None):
        if not isinstance(receipt, dict) or canonical(receipt) != canonical(self.select(request, policy, config)):
            raise ValueError('wave receipt does not reproduce under this graph, policy and configuration')
