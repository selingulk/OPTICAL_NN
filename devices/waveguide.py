"""Optical waveguide models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Waveguide:
    """Model of an optical waveguide.

    loss_db_per_cm: Propagation loss in dB/cm.
    """

    loss_db_per_cm: float = 2.0

    def get_transmission(self, length_cm: float) -> float:
        """Calculate optical transmission through a length of waveguide."""
        total_loss_db = length_cm * self.loss_db_per_cm
        return 10 ** (-total_loss_db / 10)
