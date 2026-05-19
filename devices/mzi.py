"""Mach-Zehnder Interferometer (MZI) model.

An MZI can be used as a programmable beam splitter.  A mesh of MZIs implements
unitary transformations following the Reck/Clements decomposition, enabling
analogue matrix-vector multiplication in the optical domain.
"""

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


@dataclass(frozen=True)
class MZIMeshConfig:
    """Configuration for an MZI-mesh based ONN architecture.

    An N×N unitary matrix requires N*(N-1)/2 MZIs (Reck decomposition).
    Each MZI has a programmable phase shifter.

    mzi: The MZI device model to use.
    phase_shifter_power_mw: Static power consumed by each thermo-optic phase shifter.
    """

    mzi: MZI = MZI()
    phase_shifter_power_mw: float = 5.0  # per phase shifter

    def num_mzis(self, matrix_size: int) -> int:
        """Number of MZIs needed for an N×N unitary (Reck decomposition)."""
        return matrix_size * (matrix_size - 1) // 2

    def total_phase_shifter_power_mw(self, matrix_size: int) -> float:
        """Total static power from all phase shifters in the mesh."""
        # Each MZI has 2 phase shifters (internal + external)
        return self.num_mzis(matrix_size) * 2 * self.phase_shifter_power_mw

    def mesh_transmission(self, matrix_size: int) -> float:
        """Worst-case optical signal transmission through the mesh.

        In a Reck/Clements mesh, a signal passes through at most (N-1) MZIs.
        """
        stages = matrix_size - 1
        single_mzi_transmission = self.mzi.transmission(0.0)  # best-case per MZI
        return single_mzi_transmission ** stages
