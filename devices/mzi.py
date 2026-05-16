"""Mach-Zehnder Interferometer (MZI) model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MZI:
    """Model of a Mach-Zehnder Interferometer.

    insertion_loss_db: Power loss when fully transmissive.
    extinction_ratio_db: Ratio of maximum to minimum transmission.
    """

    insertion_loss_db: float = 1.0
    extinction_ratio_db: float = 30.0

    def transmission(self, phase_shift_radians: float) -> float:
        """Calculate optical transmission for a given phase shift."""
        # Ideal MZI transmission: cos^2(phase_shift / 2)
        ideal_trans = math.cos(phase_shift_radians / 2) ** 2

        on_power = 10 ** (-self.insertion_loss_db / 10)
        off_power = on_power * 10 ** (-self.extinction_ratio_db / 10)

        # Scale ideal transmission between on_power and off_power
        trans = off_power + ideal_trans * (on_power - off_power)
        return trans
