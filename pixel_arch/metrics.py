"""PPA Metrics engines for PIXEL Architectures."""

from accelerator.metrics import PPAMetrics
from accelerator.architecture import ONNTile

class PixelOEMetricsEngine:
    """Evaluates PPA metrics for a PIXEL Optoelectronic (OE) architecture."""

    def __init__(self, tile: ONNTile):
        self.tile = tile

    def evaluate(self) -> PPAMetrics:
        num_omacs = self.tile.num_omacs
        omac = self.tile.omac_template
        n = omac.num_inputs
        m = omac.num_outputs

        # Area estimation
        mrr_area_um2 = 100.0
        laser_area_um2 = 1000.0
        pd_area_um2 = 200.0
        dac_area_um2 = 500.0
        # In OE, we use comparators (TIAs) and digital adders instead of full ADCs
        comparator_area_um2 = 100.0
        adder_area_um2 = 50.0 # CLA area

        total_mrrs = n * m * num_omacs
        total_lasers = n * num_omacs
        # Every MRR has its own PD and comparator in OE! (since it's individual conversion)
        total_pds = n * m * num_omacs
        total_comparators = n * m * num_omacs
        total_adders = m * num_omacs # One accumulator tree per output

        total_area_um2 = (
            total_mrrs * mrr_area_um2
            + total_lasers * laser_area_um2
            + total_pds * pd_area_um2
            + total_comparators * comparator_area_um2
            + total_adders * adder_area_um2
            + total_lasers * dac_area_um2
        )
        area_mm2 = total_area_um2 / 1_000_000.0

        # Power estimation
        laser_power_mw = total_lasers * omac.laser.electrical_power_mw
        dac_power_mw = total_lasers * (omac.dac.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        # Comparators use less power than high-res ADCs
        comparator_power_mw = total_comparators * (0.1 * self.tile.clock_frequency_ghz)
        adder_power_mw = total_adders * (0.05 * self.tile.clock_frequency_ghz)
        
        total_power_mw = laser_power_mw + dac_power_mw + comparator_power_mw + adder_power_mw

        throughput_tops = self.tile.get_peak_tops()

        total_macs_per_cycle = n * m * num_omacs
        total_macs_per_second = total_macs_per_cycle * self.tile.clock_frequency_ghz * 1e9

        if total_macs_per_second > 0:
            energy_per_mac_pj = (total_power_mw * 1e-3) / total_macs_per_second * 1e12
        else:
            energy_per_mac_pj = float("inf")

        return PPAMetrics(
            arch_name="PIXEL OE (Optoelectronic)",
            power_mw=total_power_mw,
            energy_per_mac_pj=energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )


class PixelOOMetricsEngine:
    """Evaluates PPA metrics for a PIXEL All-Optical (OO) architecture."""

    def __init__(self, tile: ONNTile):
        self.tile = tile

    def evaluate(self) -> PPAMetrics:
        num_omacs = self.tile.num_omacs
        omac = self.tile.omac_template
        n = omac.num_inputs
        m = omac.num_outputs

        # Area estimation
        mrr_area_um2 = 100.0
        laser_area_um2 = 1000.0
        pd_area_um2 = 200.0
        dac_area_um2 = 500.0
        adc_area_um2 = 800.0
        mzi_area_um2 = 10_000.0 # Delay line MZIs are large

        total_mrrs = n * m * num_omacs
        total_lasers = n * num_omacs
        # Only one PD/ADC per output in OO
        total_pds = m * num_omacs 
        # One delay line (MZI) per MRR along the accumulation path
        total_mzis = n * m * num_omacs

        total_area_um2 = (
            total_mrrs * mrr_area_um2
            + total_lasers * laser_area_um2
            + total_pds * pd_area_um2
            + total_pds * adc_area_um2
            + total_lasers * dac_area_um2
            + total_mzis * mzi_area_um2
        )
        area_mm2 = total_area_um2 / 1_000_000.0

        # Power estimation
        laser_power_mw = total_lasers * omac.laser.electrical_power_mw
        dac_power_mw = total_lasers * (omac.dac.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        adc_power_mw = total_pds * (omac.adc.energy_per_conversion_pj * self.tile.clock_frequency_ghz)
        
        total_power_mw = laser_power_mw + dac_power_mw + adc_power_mw

        throughput_tops = self.tile.get_peak_tops()

        total_macs_per_cycle = n * m * num_omacs
        total_macs_per_second = total_macs_per_cycle * self.tile.clock_frequency_ghz * 1e9

        if total_macs_per_second > 0:
            energy_per_mac_pj = (total_power_mw * 1e-3) / total_macs_per_second * 1e12
        else:
            energy_per_mac_pj = float("inf")

        return PPAMetrics(
            arch_name="PIXEL OO (All-Optical)",
            power_mw=total_power_mw,
            energy_per_mac_pj=energy_per_mac_pj,
            throughput_tops=throughput_tops,
            area_mm2=area_mm2,
        )
