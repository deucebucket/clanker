# Clanker-LM Hugging Face Space Readiness Benchmark

Date: 2026-09-03

Commit under test: `b61b19c28a67dcfff46cc500c6a2878755b8fcbe`, based directly on merged `main` commit `e29ccfd0eb7a54bbda2fda875f13f9f26f0e8719` plus the benchmark workflow only.

Environment:

- GitHub-hosted Ubuntu 24.04 runner
- CPython 3.12.14
- package installed with `python -m pip install -e .`
- no Gradio, model, tensor, GPU, or external runtime dependency installed

## Source payload

| Component | Bytes |
|---|---:|
| Existing `engine/` tree, including its tests | 962,110 |
| `clanker_lm/` runtime | 417,023 |
| Combined | 1,379,133 |
| Combined MiB | 1.3152 |

The production engine payload is smaller than the first row because that measurement deliberately included the existing `engine/tests/` tree.

## Runtime benchmark

The benchmark initialized the real V8-backed `ClankerLM`, ran six warm-up turns covering conversational memory, Q&A, a live-time resolver, unknown-word probing, lexical teaching, and learned-word retrieval, then timed 600 full runtime turns across six representative inputs.

| Metric | Result |
|---|---:|
| Python package import | 350.166 ms |
| `ClankerLM()` initialization | 8.245 ms |
| Timed full turns | 600 |
| Mean latency | 1.9738 ms |
| Median latency | 1.8102 ms |
| p95 latency | 2.9479 ms |
| Mean throughput | 506.63 turns/s |
| Maximum resident memory | 62.602 MiB |

The latency includes semantic parsing, memory updates, lexical overlay handling, resolver routing, contextual gates, atomic compositional realization, V8 response scoring, and trajectory bookkeeping.

## Learning-state footprint

A separate file-backed SQLite run performed an unknown-word probe, taught a definition, and queried the learned meaning.

| Metric | Result |
|---|---:|
| SQLite file after learning demonstration | 135,168 bytes |
| Learned terms in in-memory benchmark session | 1 |
| Adaptive overlay export | successful |

## Sample observed behavior

```text
My sister bought a used Honda yesterday.
Who bought the Honda?
→ Your sister bought a used Honda yesterday.

That movie was glorp.
→ What does glorp mean?

Negative, like disappointing and overhyped.
→ Glorp means negative evaluation.

What does glorp mean?
→ Glorp means negative evaluation.
```

## Interpretation

These measurements support a CPU-only demonstration. They do not include the memory and startup overhead of a web framework such as Gradio, which is expected to dominate the application wrapper. They also do not establish multi-user isolation, persistent-volume behavior, abuse resistance, or production service availability; those are deployment-layer concerns rather than core engine compute limits.
