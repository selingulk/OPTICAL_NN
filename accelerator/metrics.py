"""Power, Performance, and Area (PPA) evaluation metrics.

Supports MRR-based (ONNTile) and MZI-mesh-based (MZITile) architectures,
plus an electronic digital baseline for comparison.
"""

from dataclasses import dataclass

from .architecture import ONNTile, MZITile


@dataclass(frozen=True)
class PPAMetrics:
    """Container for Power, Performance, and Area metrics."""

    arch_name: str
    power_mw: float
    energy_per_mac_pj: float
    throughput_tops: float
    area_mm2: float

    def __str__(self) -> str:
        return (
            f"Architecture: {self.arch_name}\n"
            f"  Power:       {self.power_mw:.2f} mW\n"
            f"  Energy/MAC:  {self.energy_per_mac_pj:.4f} pJ\n"
            f"  Throughput:  {self.throughput_tops:.4f} TOPS\n"
            f"  Area:        {self.area_mm2:.4f} mm^2"
        )


class MetricsEngine:
    """Evaluates PPA metrics for a given ONNTile (MRR-based)."""

    def __init__(self, tile: ONNTile):
        self.tile = tile

    def evaluate(self) -> PPAMetrics:
        num_omacs = self.tile.num_omacs
        omac = self.tile.omac_template

        # Area estimation
        mrr_area_um2 = 100.0
        laser_area_um2 = 1000.0
        pd_area_um2 = 200.0
        dac_area_um2 = 500.0
        adc_area_um2 = 800.0

        total_mrrs = omac.num_inputs * omac.num_outputs * num_omacs
        total_lasers = omac.num_inputs * num_omacs
        total_pds = omac.num_outputs * num_omacs

        total_area_um2 = (
            total_mrrs * mrr_area_um2
            + total_lasers * laser_area_um2
            + total_pds * pd_area_um2
            + total_lasers * dac_area_um2
            + total_pds * adc_area_um2
        )
        area_mm2 = total_area_um2 / 1_000_000.0

        # Power estimation
        laser_power_mw = total_lasers * omac.laser.electrical_power_mw
        dac_power_mw = total_lasers * (omac.dac.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        adc_power_mw = total_pds * (omac.adc.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        tia_power_mw = total_pds * (omac.tia.energy_per_bit_pj * self.tile.clock_frequency_ghz)
        total_power_mw = laser_power_mw + dac_power_mw + adc_power_mw + tia_power_mw

        # Performance estimation
        throughput_tops = self.tile.get_peak_tops()

        # Energy per MAC
        total_macs_per_cycle = omac.num_inputs * omac.num_outputs * num_omacs
        total_macs_per_second = total_macs_per_cycle * self.tile.clock_frequency_ghz * 1e9

        if total_macs_per_second > 0:
            energy_per_mac_pj = (total_power_mw * 1e-3) / total_macs_per_second * 1e12
        else:
            energy_per_mac_pj = float("inf")

        return PPAMetrics(
            arch_name="MRR-Based ONN",
            power_mw=total_power_mw,
            energy_per_mac_pj=energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )


class MZIMetricsEngine:
    """Evaluates PPA metrics for an MZI-mesh based tile."""

    def __init__(self, tile: MZITile):
        self.tile = tile

    def evaluate(self) -> PPAMetrics:
        num_omacs = self.tile.num_omacs
        omac = self.tile.omac_template
        n = omac.matrix_size

        # Area estimation
        mzi_area_um2 = 10_000.0  # MZIs are larger than MRRs
        laser_area_um2 = 1000.0
        pd_area_um2 = 200.0
        dac_area_um2 = 500.0
        adc_area_um2 = 800.0

        total_mzis = omac.num_mzis() * num_omacs
        total_lasers = n * num_omacs
        total_pds = n * num_omacs

        total_area_um2 = (
            total_mzis * mzi_area_um2
            + total_lasers * laser_area_um2
            + total_pds * pd_area_um2
            + total_lasers * dac_area_um2
            + total_pds * adc_area_um2
        )
        area_mm2 = total_area_um2 / 1_000_000.0

        # Power estimation
        laser_power_mw = total_lasers * omac.laser.electrical_power_mw
        phase_shifter_power_mw = omac.phase_shifter_power_mw() * num_omacs
        dac_power_mw = total_lasers * (omac.dac.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        adc_power_mw = total_pds * (omac.adc.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        total_power_mw = laser_power_mw + phase_shifter_power_mw + dac_power_mw + adc_power_mw

        # Performance
        throughput_tops = self.tile.get_peak_tops()

        # Energy per MAC
        macs_per_cycle = n * n * num_omacs
        macs_per_second = macs_per_cycle * self.tile.clock_frequency_ghz * 1e9
        if macs_per_second > 0:
            energy_per_mac_pj = (total_power_mw * 1e-3) / macs_per_second * 1e12
        else:
            energy_per_mac_pj = float("inf")

        return PPAMetrics(
            arch_name="MZI-Mesh ONN",
            power_mw=total_power_mw,
            energy_per_mac_pj=energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )


class ElectronicBaselineEngine:
    """Simple electronic (digital ASIC) MAC array baseline for comparison.

    Based on published data for 7nm digital MAC arrays (e.g., TPU-like).
    """

    def __init__(
        self,
        num_mac_units: int = 256,
        clock_frequency_ghz: float = 1.0,
        energy_per_mac_pj: float = 0.5,
        mac_unit_area_um2: float = 2000.0,
        arch_name: str = "Digital ASIC Baseline (7nm)",
    ):
        self.num_mac_units = num_mac_units
        self.clock_frequency_ghz = clock_frequency_ghz
        self.energy_per_mac_pj = energy_per_mac_pj
        self.mac_unit_area_um2 = mac_unit_area_um2
        self.arch_name = arch_name

    def evaluate(self) -> PPAMetrics:
        macs_per_second = self.num_mac_units * self.clock_frequency_ghz * 1e9
        throughput_tops = (2 * self.num_mac_units * self.clock_frequency_ghz) / 1000.0

        power_w = self.energy_per_mac_pj * 1e-12 * macs_per_second
        power_mw = power_w * 1e3

        area_mm2 = (self.num_mac_units * self.mac_unit_area_um2) / 1_000_000.0

        return PPAMetrics(
            arch_name=self.arch_name,
            power_mw=power_mw,
            energy_per_mac_pj=self.energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )
