"""Top-level Optical Neural Network (ONN) Architecture models."""

from dataclasses import dataclass

from .omac import OMAC


@dataclass(frozen=True)
class ONNTile:
    """A tile containing multiple OMACs and shared resources.

    num_omacs: Number of OMAC units in the tile.
    clock_frequency_ghz: Clock frequency of the accelerator in GHz.
    """

    num_omacs: int = 4
    clock_frequency_ghz: int = 1
    omac_template: OMAC = OMAC()

    def get_peak_tops(self) -> float:
        """Peak Tera-Operations Per Second (TOPS).

        An operation is a multiply-accumulate (MAC), which counts as 2 operations.
        Total operations = 2 * inputs * outputs * OMACs
        """
        macs_per_cycle_per_omac = self.omac_template.num_inputs * self.omac_template.num_outputs
        macs_per_cycle = macs_per_cycle_per_omac * self.num_omacs
        ops_per_cycle = 2 * macs_per_cycle

        # Tops = (ops_per_cycle * clock_frequency_ghz * 10^9) / 10^12 = ops_per_cycle * clock_frequency_ghz / 1000
        return (ops_per_cycle * self.clock_frequency_ghz) / 1000.0
