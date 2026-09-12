"""Bounded fixed-point salience. These numbers are NOT confidence percentages."""
from dataclasses import dataclass

SCALE = 1024
SCHEMA = 'clanker-attention-wave-v1'


@dataclass(frozen=True)
class WaveConfig:
    decay: int = 896
    minimum_salience: int = 96

    def __post_init__(self):
        if type(self.decay) is not int or not 1 <= self.decay < SCALE:
            raise ValueError('wave decay must be a positive integer below 1024')
        if (type(self.minimum_salience) is not int
                or not 1 <= self.minimum_salience <= SCALE):
            raise ValueError('bounded positive salience floor required')


def gain(value: int) -> int:
    if type(value) is not int or not 0 <= value <= SCALE:
        raise ValueError('edge salience must be an integer from 0 to 1024')
    return value
