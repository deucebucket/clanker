"""Ensemble — Engine + Model working together.

The engine handles the 66% it's confident about.
The model handles the 34% the engine can't resolve.
Together they cover more ground than either alone.

When the engine says HIGH confidence: trust the engine (fast, auditable)
When the engine says NULL/LOW: ask the model (slower, handles pragmatics)

Usage:
    from demo.ensemble import Ensemble
    ens = Ensemble()
    result = ens.process("Whatever makes you happy")
    # result.source = "engine" or "model" or "consensus"
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from demo.shared import VADUG
from demo.pendulum_v2 import PendulumV2
from demo.confidence import ConfidenceScorer


@dataclass
class EnsembleResult:
    """Result from the engine+model ensemble."""
    vadug: VADUG
    source: str               # "engine", "model", "consensus", "weighted"
    engine_vadug: VADUG
    engine_confidence: int
    model_vadug: Optional[VADUG]
    trace: list
    agreement: Optional[str]  # "agree", "disagree", None


class Ensemble:
    """Engine + Model ensemble for full emotional understanding."""

    # Thresholds: original 70/30 with engine-weighted consensus
    ENGINE_ONLY_THRESHOLD = 70    # HIGH confidence = engine only
    MODEL_REQUIRED_THRESHOLD = 30  # below this = model takes over

    def __init__(self, engine=None, model_dir="training/checkpoints/best"):
        self.engine = engine or PendulumV2()
        self.confidence_scorer = ConfidenceScorer()
        self.model = None
        self.tokenizer = None
        self._model_dir = model_dir
        self._model_available = os.path.exists(
            os.path.join(model_dir, "model.pt"))
        self._bucket_thresholds = None
        self._vadug_dims = None

    def _load_model(self):
        """Lazy-load the neural model (only when needed)."""
        if self.model is not None:
            return True
        if not self._model_available:
            return False

        try:
            import torch
            from transformers import AutoTokenizer
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from training.train import ClankerClassifier, VADUG_DIMS, BUCKET_THRESHOLDS

            self.model = ClankerClassifier.load_pretrained(self._model_dir)
            self.model.set_eval = lambda: None  # compat
            self._bucket_thresholds = BUCKET_THRESHOLDS
            self._vadug_dims = VADUG_DIMS
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
            return True
        except Exception:
            self._model_available = False
            return False

    def _run_model(self, text):
        """Run the neural model on text. Returns VADUG or None."""
        if not self._load_model():
            return None

        import torch

        enc = self.tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=128)
        emotion_pos = torch.tensor([enc["input_ids"].shape[1] - 1])

        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"], emotion_pos)

        preds = {}
        for dim in self._vadug_dims:
            bucket = torch.argmax(logits[dim], dim=-1).item()
            if bucket > 0:
                preds[dim] = int(self._bucket_thresholds[bucket - 1] + 18)
            else:
                preds[dim] = 18

        return VADUG(
            v=preds["V"], a=preds["A"], d=preds["D"],
            u=preds["U"], g=preds["G"],
        )

    def process(self, text):
        """Process text through the ensemble.

        HIGH confidence (70+): engine only (fast, auditable)
        MEDIUM (30-69): run both, build consensus
        LOW/NULL (0-29): model required (engine can't resolve)
        """
        engine_vadug, trace = self.engine.process_text(text)
        conf = self.confidence_scorer.score(engine_vadug, trace)

        # HIGH: trust the engine
        if conf.score >= self.ENGINE_ONLY_THRESHOLD:
            return EnsembleResult(
                vadug=engine_vadug, source="engine",
                engine_vadug=engine_vadug, engine_confidence=conf.score,
                model_vadug=None, trace=trace, agreement=None,
            )

        # MEDIUM or LOW: bring in the model
        model_vadug = self._run_model(text)

        if model_vadug is None:
            return EnsembleResult(
                vadug=engine_vadug, source="engine",
                engine_vadug=engine_vadug, engine_confidence=conf.score,
                model_vadug=None, trace=trace, agreement=None,
            )

        # Compare engine and model
        v_agree = abs(engine_vadug.v - model_vadug.v) < 40
        d_agree = abs(engine_vadug.d - model_vadug.d) < 40

        if v_agree and d_agree:
            # Consensus: weight toward engine (more granular, more precise)
            # Engine 70%, model 30%
            final = VADUG(
                v=int(engine_vadug.v * 0.7 + model_vadug.v * 0.3),
                a=int(engine_vadug.a * 0.7 + model_vadug.a * 0.3),
                d=int(engine_vadug.d * 0.7 + model_vadug.d * 0.3),
                u=int(engine_vadug.u * 0.7 + model_vadug.u * 0.3),
                g=int(engine_vadug.g * 0.7 + model_vadug.g * 0.3),
            )
            return EnsembleResult(
                vadug=final, source="consensus",
                engine_vadug=engine_vadug, engine_confidence=conf.score,
                model_vadug=model_vadug, trace=trace, agreement="agree",
            )
        else:
            # Disagree: weight by engine confidence
            if conf.score < self.MODEL_REQUIRED_THRESHOLD:
                final = model_vadug
                source = "model"
            else:
                w_e = conf.score / 100
                w_m = 1.0 - w_e
                final = VADUG(
                    v=int(engine_vadug.v * w_e + model_vadug.v * w_m),
                    a=int(engine_vadug.a * w_e + model_vadug.a * w_m),
                    d=int(engine_vadug.d * w_e + model_vadug.d * w_m),
                    u=int(engine_vadug.u * w_e + model_vadug.u * w_m),
                    g=int(engine_vadug.g * w_e + model_vadug.g * w_m),
                )
                source = "weighted"

            return EnsembleResult(
                vadug=final, source=source,
                engine_vadug=engine_vadug, engine_confidence=conf.score,
                model_vadug=model_vadug, trace=trace, agreement="disagree",
            )
