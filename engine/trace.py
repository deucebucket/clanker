"""Pipeline debug trace — watch each engine stage in real time.

Every stage in the pipeline logs what it did: inputs, decisions, outputs.
Enable with `compute_vadug(text, trace=True)` or `Pipeline(trace=True)`.

Usage:
    from engine.pendulum import compute_vadug

    # Normal (no trace):
    vadug, meta = compute_vadug("im fine")

    # With trace:
    vadug, meta = compute_vadug("im fine", trace=True)
    for entry in meta["pipeline_trace"]:
        print(f"[{entry['stage']}] {entry['summary']}")

    # Pretty print:
    from engine.trace import print_trace
    print_trace(meta["pipeline_trace"])
"""

from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class TraceEntry:
    """One stage's contribution to the pipeline trace."""
    stage: str          # TOKENIZE, CLASSIFY, PROXIMITY, FORCES, etc.
    summary: str        # one-line human-readable summary
    details: dict = field(default_factory=dict)  # full data for debugging
    input_state: dict = field(default_factory=dict)   # state BEFORE this stage
    output_state: dict = field(default_factory=dict)  # state AFTER this stage


class PipelineTrace:
    """Accumulates trace entries across pipeline stages."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.entries: List[TraceEntry] = []

    def log(self, stage: str, summary: str, **kwargs):
        """Log a trace entry. No-op if tracing is disabled."""
        if not self.enabled:
            return
        self.entries.append(TraceEntry(
            stage=stage,
            summary=summary,
            details=kwargs.get("details", {}),
            input_state=kwargs.get("input_state", {}),
            output_state=kwargs.get("output_state", {}),
        ))

    def to_list(self) -> list:
        """Export trace as list of dicts (for meta["pipeline_trace"])."""
        return [
            {
                "stage": e.stage,
                "summary": e.summary,
                "details": e.details,
                "input_state": e.input_state,
                "output_state": e.output_state,
            }
            for e in self.entries
        ]


def print_trace(trace_list: list, verbose: bool = False):
    """Pretty-print a pipeline trace."""
    for entry in trace_list:
        stage = entry["stage"]
        summary = entry["summary"]
        print(f"  [{stage:12s}] {summary}")
        if verbose and entry.get("details"):
            for k, v in entry["details"].items():
                print(f"  {'':14s} {k}: {v}")


# ── Stage-specific trace helpers ──────────────────────────────

def trace_tokenize(trace: PipelineTrace, words_in: list, words_out: list, compounds: list):
    """Log tokenization stage."""
    if not compounds:
        trace.log("TOKENIZE", f"{len(words_in)} words, no compounds")
    else:
        trace.log("TOKENIZE",
                  f"{len(words_in)}→{len(words_out)} words, compounds: {compounds}",
                  details={"compounds": compounds})


def trace_classify(trace: PipelineTrace, roles: list):
    """Log classification stage."""
    role_counts = {}
    for r in roles:
        role_counts[r.role] = role_counts.get(r.role, 0) + 1
    emotional = [r for r in roles if r.role == "EMOTIONAL"]
    trace.log("CLASSIFY",
              f"{len(roles)} words: {dict(role_counts)}",
              details={"emotional_words": [(r.word, r.force[0] if r.force else 0) for r in emotional]})


def trace_proximity(trace: PipelineTrace, word: str, role: str, coeff: float, force: tuple):
    """Log proximity coefficient for one word."""
    if abs(coeff - 1.0) > 0.05 or (force and abs(force[0]) > 10):
        trace.log("PROXIMITY",
                  f"{word}({role}): coeff={coeff:.2f}, dV={force[0] if force else 0:+d}",
                  details={"word": word, "coefficient": round(coeff, 3)})


def trace_forces(trace: PipelineTrace, state_v: float, state_a: float, state_d: float,
                 state_w: float, m_eff: float):
    """Log force accumulation result."""
    trace.log("FORCES",
              f"V={state_v:.0f} A={state_a:.0f} D={state_d:.0f} W={state_w:.0f} M_eff={m_eff:.3f}",
              output_state={"V": round(state_v), "A": round(state_a),
                           "D": round(state_d), "W": round(state_w)})


def trace_structure(trace: PipelineTrace, pattern: str, confidence: float,
                    v_weight: float, state_v_before: float, state_v_after: float):
    """Log a structural pattern firing."""
    trace.log("STRUCTURE",
              f"{pattern} (conf={confidence:.2f}, v_weight={v_weight:+.0f}) "
              f"V: {state_v_before:.0f}→{state_v_after:.0f}",
              details={"pattern": pattern, "confidence": confidence, "v_weight": v_weight})


def trace_anomaly(trace: PipelineTrace, anomaly_type: str, severity: int, description: str):
    """Log an anomaly detection."""
    trace.log("ANOMALY",
              f"[{anomaly_type}] severity={severity}: {description[:60]}",
              details={"type": anomaly_type, "severity": severity})


def trace_saturate(trace: PipelineTrace, state_v_before: float, state_v_after: float):
    """Log tanh saturation effect."""
    delta = abs(state_v_after - state_v_before)
    if delta > 1:
        trace.log("SATURATE",
                  f"V: {state_v_before:.0f}→{state_v_after:.0f} (compressed {delta:.0f})",
                  details={"before": round(state_v_before), "after": round(state_v_after)})
    else:
        trace.log("SATURATE", "no compression needed")


def trace_final(trace: PipelineTrace, vadug):
    """Log final VADUGWI output."""
    trace.log("OUTPUT",
              f"V={vadug.v} A={vadug.a} D={vadug.d} U={vadug.u} G={vadug.g} W={vadug.w} I={vadug.i}")
