"""Noise and non-ideal effect models for optical neural network simulation.

This module adds shot noise, thermal noise, and crosstalk models that can be
layered on top of the ideal device models to study accuracy degradation.
"""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for noise injection during simulation.

    shot_noise_enabled: Enable shot noise on photodetector output.
    thermal_noise_std: Standard deviation of additive thermal noise (in mW).
    crosstalk_fraction: Fraction of signal power coupled to adjacent channels.
    random_seed: Seed for reproducibility; None for random behaviour.
    """

    shot_noise_enabled: bool = False
    thermal_noise_std: float = 0.0
    crosstalk_fraction: float = 0.0
    random_seed: int | None = 42

    def get_rng(self) -> random.Random:
        """Return a seeded random number generator."""
        return random.Random(self.random_seed)


def add_shot_noise(optical_power_mw: float, rng: random.Random) -> float:
    """Approximate shot noise as Gaussian with variance proportional to signal.

    For simplicity, shot noise standard deviation ~ sqrt(power).
    """
    if optical_power_mw <= 0:
        return 0.0
    std = math.sqrt(optical_power_mw) * 0.01  # scale factor for simulation
    noisy = optical_power_mw + rng.gauss(0, std)
    return max(noisy, 0.0)


def add_thermal_noise(value: float, std: float, rng: random.Random) -> float:
    """Add Gaussian thermal noise to a signal."""
    if std <= 0:
        return value
    return value + rng.gauss(0, std)


def apply_crosstalk(signals: list[float], fraction: float) -> list[float]:
    """Apply nearest-neighbour crosstalk coupling.

    Each channel receives ``fraction`` of its immediate neighbours' power.
    """
    if fraction <= 0 or len(signals) <= 1:
        return list(signals)

    result = list(signals)
    n = len(signals)
    for i in range(n):
        leak = 0.0
        if i > 0:
            leak += signals[i - 1] * fraction
        if i < n - 1:
            leak += signals[i + 1] * fraction
        result[i] = signals[i] * (1 - fraction) + leak
    return result
