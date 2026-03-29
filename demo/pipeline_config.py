"""Pipeline configuration for toggling individual layers on/off."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Toggle individual layers on/off for testing and tuning."""

    # Input preprocessing
    nonsense_filter: bool = True
    fuzzy_matching: bool = False  # disabled: ablation showed -3.3pp accuracy (edit distance too aggressive with 46K entries)
    intent_detection: bool = True

    # Emotional processing
    exponential_decay: bool = True
    pendulum: bool = True
    idiom_detection: bool = True
    context_modifiers: bool = True
    crisis_momentum_lock: bool = True
    morpheme_decomposition: bool = True
    recency_weighting: bool = True
    shift_markers: bool = True
    bookend_parsing: bool = True
    word_role_detection: bool = True
    bigram_detection: bool = True

    # Multi-path processing
    multipath_processing: bool = True

    # Analysis
    tonal_analysis: bool = True
    emotional_chunking: bool = True
    arc_detection: bool = True
    contrast_detection: bool = True

    # Post-processing
    grading: bool = True
    personality_filter: bool = True

    # Normalization (optional — density hurts benchmark accuracy, keep raw V as default)
    density_normalization: bool = False

    # Response generation
    content_extraction: bool = True
    delivery_modes: bool = True

    # Presets
    @classmethod
    def minimal(cls):
        """Just pendulum, nothing else."""
        return cls(
            nonsense_filter=False, fuzzy_matching=False, intent_detection=False,
            idiom_detection=False, context_modifiers=False, crisis_momentum_lock=False,
            morpheme_decomposition=False, recency_weighting=False, shift_markers=False,
            multipath_processing=False,
            bookend_parsing=False, word_role_detection=False, bigram_detection=False,
            tonal_analysis=False, emotional_chunking=False, arc_detection=False,
            contrast_detection=False, grading=False, personality_filter=False,
            content_extraction=False, delivery_modes=False,
        )

    @classmethod
    def full(cls):
        """Everything enabled (default)."""
        return cls()

    @classmethod
    def production(cls):
        """Recommended production settings."""
        return cls()  # currently same as full

    def disabled_layers(self) -> list[str]:
        """Return list of disabled layer names."""
        return [name for name, val in self.__dict__.items() if val is False]

    def enabled_layers(self) -> list[str]:
        """Return list of enabled layer names."""
        return [name for name, val in self.__dict__.items() if val is True]
