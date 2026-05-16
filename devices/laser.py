"""Optical laser source models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Laser:
    """Model of an optical laser source.

    power_mw: Optical output power in milliwatts.
    wall_plug_efficiency: Efficiency of converting electrical to optical power (0 to 1).
    """

    power_mw: float = 10.0
    wall_plug_efficiency: float = 0.2

    @property
    def electrical_power_mw(self) -> float:
        """Electrical power required to drive the laser."""
        if self.wall_plug_efficiency <= 0:
            raise ValueError("Wall-plug efficiency must be greater than 0.")
        return self.power_mw / self.wall_plug_efficiency
