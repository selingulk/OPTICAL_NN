"""Top-level Optical Neural Network (ONN) Architecture models.

Provides both MRR-based and MZI-mesh-based tile configurations, plus a
latency model for mapping workloads onto the tile.
"""

from dataclasses import dataclass

from .omac import OMAC, MZI_OMAC


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

    # ---- latency helpers -------------------------------------------------------

    def cycles_for_mvm(self, matrix_rows: int, matrix_cols: int, bitwidth: int = 4) -> int:
        """Number of clock cycles to complete one MVM on this tile.

        The bit-serial architecture processes one bit-column per cycle.
        A single OMAC can handle (num_inputs) columns and (num_outputs) rows
        simultaneously, so larger matrices must be tiled.

        cycles = tile_passes_rows × tile_passes_cols × bitwidth_input × bitwidth_weight
        """
        import math
        tile_passes_rows = math.ceil(matrix_rows / self.omac_template.num_outputs)
        tile_passes_cols = math.ceil(matrix_cols / self.omac_template.num_inputs)

        # Bit-serial: iterate over all bit positions of both operands
        bit_cycles = bitwidth * bitwidth

        return tile_passes_rows * tile_passes_cols * bit_cycles

    def latency_us(self, matrix_rows: int, matrix_cols: int, bitwidth: int = 4) -> float:
        """Wall-clock latency in microseconds for one MVM."""
        cycles = self.cycles_for_mvm(matrix_rows, matrix_cols, bitwidth)
        period_ns = 1.0 / self.clock_frequency_ghz  # ns per cycle
        return (cycles * period_ns) / 1000.0  # convert ns → µs


@dataclass(frozen=True)
class MZITile:
    """A tile using MZI-mesh based OMAC units.

    This represents a coherent ONN architecture where unitary transformations
    are implemented via Clements/Reck MZI meshes.
    """

    num_omacs: int = 4
    clock_frequency_ghz: int = 1
    omac_template: MZI_OMAC = MZI_OMAC()

    def get_peak_tops(self) -> float:
        """Peak TOPS for the MZI-mesh tile.

        For an N×N mesh, one MVM produces N outputs from N inputs per cycle.
        """
        n = self.omac_template.matrix_size
        macs_per_cycle = n * n * self.num_omacs
        ops_per_cycle = 2 * macs_per_cycle
        return (ops_per_cycle * self.clock_frequency_ghz) / 1000.0

    def cycles_for_mvm(self, matrix_rows: int, matrix_cols: int, bitwidth: int = 4) -> int:
        """Cycles for MVM on MZI tile (analogue — no bit-serial overhead)."""
        import math
        n = self.omac_template.matrix_size
        tile_passes_rows = math.ceil(matrix_rows / n)
        tile_passes_cols = math.ceil(matrix_cols / n)
        # MZI mesh is analogue: one pass per tile, but DAC/ADC conversion
        # takes ~bitwidth cycles for the required precision
        return tile_passes_rows * tile_passes_cols * bitwidth

    def latency_us(self, matrix_rows: int, matrix_cols: int, bitwidth: int = 4) -> float:
        cycles = self.cycles_for_mvm(matrix_rows, matrix_cols, bitwidth)
        period_ns = 1.0 / self.clock_frequency_ghz
        return (cycles * period_ns) / 1000.0
