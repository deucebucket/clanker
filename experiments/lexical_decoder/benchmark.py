"""Measure real native response construction, not artificial streaming speed.

Run under OS resource caps, e.g. ulimit -v 262144 and taskset on one permitted
CPU. This original development workload is not a language-quality evaluation.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone

from .affinity import canonical
from .runtime import ReceiptChat


WORKLOAD = (
    "Hello.",
    "My brother opened a blue box yesterday.",
    "Who opened the box?",
    "I am disappointed.",
    "Thanks.",
    "What is 11 * 7?",
)


def summarize(values):
    ordered = sorted(values)
    return {"mean": statistics.fmean(values), "p50": statistics.median(values),
            "p95": ordered[max(0, int(len(ordered)*.95)-1)],
            "p99": ordered[max(0, int(len(ordered)*.99)-1)]}


def main():
    elapsed, first, init, receipt_sizes, node_counts, token_counts = [], [], [], [], [], []
    sample = []
    backend = ""
    code_hash = pack_hash = ""
    pack_bytes = 0
    for session in range(20):
        start = time.perf_counter_ns()
        with ReceiptChat(clock=lambda: datetime(2026, 9, 9, 12, 34, tzinfo=timezone.utc)) as chat:
            init.append((time.perf_counter_ns()-start)/1e6)
            for text in WORKLOAD:
                start = time.perf_counter_ns()
                result = chat.process(text)
                elapsed.append((time.perf_counter_ns()-start)/1e6)
                chunks = chat.committed_chunks()
                head = next(chunks)
                first.append((time.perf_counter_ns()-start)/1e6)
                assert head + "".join(chunks) == result.response
                r = chat.receipt
                assert r["coverage"] == "supported", r["coverage"]
                assert r["backend"] == "clanker-v8", "real V8 is mandatory for this measurement"
                receipt_sizes.append(len(canonical(r).encode()))
                node_counts.append(r['nodes_used']);token_counts.append(len(r['selected_word_ids']))
                backend, code_hash, pack_hash = r['backend'], r['runtime_sha256'], r['pack_sha256']
                pack_bytes = chat.pack.connection.execute('PRAGMA page_count').fetchone()[0] * chat.pack.connection.execute('PRAGMA page_size').fetchone()[0]
                if session == 0:
                    sample.append({"input": text, "response": result.response, "status": result.contract.status.value})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == 'Darwin' else peak*1024
    data = {"scope":"original development smoke workload; no quality/generalization claim",
            "backend":backend,"python":platform.python_version(),"platform":platform.platform(),
            "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            "address_space_limit_bytes":list(resource.getrlimit(resource.RLIMIT_AS)),
            "runtime_sha256":code_hash,"pack_sha256":pack_hash,"pack_sqlite_bytes":pack_bytes,"sessions":20,"turns":len(elapsed),
            "initialization_ms":summarize(init),"construction_ms":summarize(elapsed),
            "first_validated_chunk_ms":summarize(first),"max_rss_bytes":peak_bytes,
            "receipt_bytes":summarize(receipt_sizes),"max_nodes":max(node_counts),
            "total_tokens":sum(token_counts),"tokens_per_construction_second":sum(token_counts)/(sum(elapsed)/1000),
            "delivery":"all tokens held until complete validation; no network measurement or sleeps",
            "sample":sample}
    print(json.dumps(data,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
