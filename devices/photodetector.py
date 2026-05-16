"""Photodetector models for optical-to-electrical conversion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Photodetector:
    """Model of a Photodetector.

    responsivity_a_w: Responsivity in Amperes per Watt.
    dark_current_na: Dark current in nanoamperes.
    bandwidth_ghz: Bandwidth of the detector in GHz.
    """

    responsivity_a_w: float = 0.8
    dark_current_na: float = 10.0
    bandwidth_ghz: float = 50.0

    def get_photocurrent_ua(self, optical_power_mw: float) -> float:
        """Calculate photocurrent in microamperes for a given optical power."""
        return (optical_power_mw * self.responsivity_a_w * 1000) + (self.dark_current_na / 1000)
