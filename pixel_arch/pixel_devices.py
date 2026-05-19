"""Physical device models customized for the PIXEL architecture.

Includes MRRs with thermal drift and TIAs with dynamic decision thresholds.
"""

from dataclasses import dataclass
import random
from enum import Enum
from typing import Callable

from devices.mrr import _validate_bit


class ThresholdStrategy(Enum):
    """Strategies for dynamic decision thresholding."""
    FIXED = "fixed"                # Fixed threshold (e.g., 0.5)
    MEAN_POWER = "mean_power"      # (Expected ON + Expected OFF) / 2
    SCALED_ON = "scaled_on"        # Expected ON * scaling_factor


@dataclass(frozen=True)
class PixelMRRConfig:
    """MRR configuration including thermal drift for organic error modeling."""
    insertion_loss_db: float = 0.0
    extinction_ratio_db: float = 20.0
    thermal_drift_std_db: float = 0.5  # Standard deviation for Gaussian noise added to IL/ER
    
    # Thresholding configuration
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.MEAN_POWER
    fixed_threshold: float = 0.5
    threshold_scale_factor: float = 0.5  # Used for SCALED_ON strategy

    def get_dynamic_threshold(self) -> float:
        """Calculate the decision threshold based on the selected strategy."""
        expected_on = 10 ** (-self.insertion_loss_db / 10)
        expected_off = expected_on * 10 ** (-self.extinction_ratio_db / 10)

        if self.threshold_strategy == ThresholdStrategy.FIXED:
            return self.fixed_threshold
        elif self.threshold_strategy == ThresholdStrategy.MEAN_POWER:
            return (expected_on + expected_off) / 2.0
        elif self.threshold_strategy == ThresholdStrategy.SCALED_ON:
            return expected_on * self.threshold_scale_factor
        else:
            raise ValueError(f"Unknown threshold strategy: {self.threshold_strategy}")


def pixel_optical_transmission(
    input_bit: int, 
    weight_bit: int, 
    config: PixelMRRConfig,
    rng: random.Random | None = None
) -> float:
    """Return normalized optical output power, incorporating thermal drift."""
    _validate_bit(input_bit, "input_bit")
    _validate_bit(weight_bit, "weight_bit")

    if input_bit == 0:
        return 0.0

    # Add thermal drift (Gaussian noise) to the ideal dB parameters
    if rng is None:
        rng = random.Random()
        
    drift_il = 0.0
    drift_er = 0.0
    if config.thermal_drift_std_db > 0:
        # Drift affects the resonance wavelength, which randomly degrades both IL and ER
        # We model this as independent Gaussian perturbations to the dB values.
        # Absolute value for IL drift to ensure it always strictly increases loss (or oscillates around 0 if we allow small negative which is unphysical, so max(0, ...))
        drift_il = max(0, rng.gauss(0, config.thermal_drift_std_db))
        # ER typically degrades with drift
        drift_er = rng.gauss(0, config.thermal_drift_std_db)

    effective_il = config.insertion_loss_db + drift_il
    effective_er = config.extinction_ratio_db + drift_er

    on_power = 10 ** (-effective_il / 10)
    off_power = on_power * 10 ** (-effective_er / 10)

    return on_power if weight_bit == 1 else off_power
