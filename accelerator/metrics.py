"""Power, Performance, and Area (PPA) evaluation metrics."""

from dataclasses import dataclass

from .architecture import ONNTile


@dataclass(frozen=True)
class PPAMetrics:
    """Container for Power, Performance, and Area metrics."""

    power_mw: float
    energy_per_mac_pj: float
    throughput_tops: float
    area_mm2: float

    def __str__(self) -> str:
        return (
            f"Power: {self.power_mw:.2f} mW\n"
            f"Energy/MAC: {self.energy_per_mac_pj:.2f} pJ\n"
            f"Throughput: {self.throughput_tops:.2f} TOPS\n"
            f"Area: {self.area_mm2:.4f} mm^2"
        )


class MetricsEngine:
    """Evaluates PPA metrics for a given ONNTile."""

    def __init__(self, tile: ONNTile):
        self.tile = tile

    def evaluate(self) -> PPAMetrics:
        # Simple placeholder evaluation logic
        num_omacs = self.tile.num_omacs
        omac = self.tile.omac_template

        # Area estimation (placeholder values)
        mrr_area_um2 = 100.0
        laser_area_um2 = 1000.0
        pd_area_um2 = 200.0

        total_mrrs = omac.num_inputs * omac.num_outputs * num_omacs
        total_lasers = omac.num_inputs * num_omacs
        total_pds = omac.num_outputs * num_omacs

        total_area_um2 = (total_mrrs * mrr_area_um2) + (total_lasers * laser_area_um2) + (total_pds * pd_area_um2)
        area_mm2 = total_area_um2 / 1_000_000.0

        # Power estimation
        laser_power_mw = total_lasers * omac.laser.electrical_power_mw
        dac_power_mw = total_lasers * (omac.dac.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        adc_power_mw = total_pds * (omac.adc.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        total_power_mw = laser_power_mw + dac_power_mw + adc_power_mw

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
            power_mw=total_power_mw,
            energy_per_mac_pj=energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )
