"""Optical Multiply-Accumulate (OMAC) core models.

This module provides a full signal-path simulation:
  Laser → DAC → MRR weight bank → Waveguide → Photodetector → TIA → ADC

It also provides an MZI-mesh based alternative for analogue MVM.
"""

from dataclasses import dataclass, field

from devices.electrical import ADC, DAC, SERDES, TIA
from devices.laser import Laser
from devices.mrr import MRRConfig
from devices.mzi import MZIMeshConfig
from devices.noise import NoiseConfig, add_shot_noise, add_thermal_noise, apply_crosstalk
from devices.photodetector import Photodetector
from devices.waveguide import Waveguide


@dataclass(frozen=True)
class OMAC:
    """Model of an Optical Multiply-Accumulate block.

    Assumes a weight-stationary design with an array of MRRs acting as weights.
    Inputs are provided by lasers/modulators, and outputs are read by PDs.
    """

    num_inputs: int = 16
    num_outputs: int = 16

    laser: Laser = Laser()
    mrr_config: MRRConfig = MRRConfig()
    pd: Photodetector = Photodetector()
    wg: Waveguide = Waveguide()
    dac: DAC = DAC()
    adc: ADC = ADC()
    tia: TIA = TIA()
    noise_config: NoiseConfig = NoiseConfig()

    # Waveguide length in cm between components
    wg_length_laser_to_mrr_cm: float = 0.1
    wg_length_mrr_to_pd_cm: float = 0.1

    # --- signal-path helpers ---------------------------------------------------

    def _laser_output_power(self) -> float:
        """Optical power delivered to the MRR array by one laser."""
        raw_power = self.laser.power_mw
        # Attenuate through waveguide from laser to MRR bank
        wg_trans = self.wg.get_transmission(self.wg_length_laser_to_mrr_cm)
        return raw_power * wg_trans

    def signal_path_single(self, input_power_mw: float, weight_transmission: float) -> float:
        """Simulate full signal path for one input-weight pair.

        Returns the estimated optical power (mW) arriving at the photodetector.
        """
        # Modulated power after weight MRR
        modulated = input_power_mw * weight_transmission

        # Waveguide loss from MRR to PD
        wg_trans = self.wg.get_transmission(self.wg_length_mrr_to_pd_cm)
        at_pd = modulated * wg_trans

        return at_pd

    def compute_dot_product_analogue(
        self,
        inputs: list[int],
        weights: list[int],
        bitwidth: int = 4,
    ) -> float:
        """Compute one analogue dot product through the full optical signal path.

        Accepts integer inputs and weights (as used by the rest of the simulator),
        normalises them to continuous MRR transmission levels, and models the
        full path: laser → modulator → MRR weight → waveguide → photodetector.

        Returns the photocurrent (uA) at the output.
        """
        if len(inputs) != len(weights):
            raise ValueError("inputs and weights must have the same length.")

        laser_power = self._laser_output_power()
        rng = self.noise_config.get_rng()
        max_val = (1 << bitwidth) - 1
        if max_val == 0:
            max_val = 1

        on_power = 10 ** (-self.mrr_config.insertion_loss_db / 10)
        off_power = on_power * 10 ** (-self.mrr_config.extinction_ratio_db / 10)

        # Accumulate optical power arriving at the PD (incoherent sum)
        total_optical_power = 0.0
        for inp, wt in zip(inputs, weights):
            # Normalize integer values to [0, 1] range
            inp_norm = inp / max_val
            wt_norm = wt / max_val

            # Input modulates the laser intensity; weight sets MRR transmission
            input_power = laser_power * inp_norm
            mrr_trans = off_power + wt_norm * (on_power - off_power)
            pwr = self.signal_path_single(input_power, mrr_trans)

            # Noise injection
            if self.noise_config.shot_noise_enabled:
                pwr = add_shot_noise(pwr, rng)
            if self.noise_config.thermal_noise_std > 0:
                pwr = add_thermal_noise(pwr, self.noise_config.thermal_noise_std, rng)

            total_optical_power += pwr

        # Photodetector conversion
        photocurrent_ua = self.pd.get_photocurrent_ua(total_optical_power)
        return photocurrent_ua

    def compute_mvm_analogue(
        self,
        weight_matrix: list[list[int]],
        input_vector: list[int],
        bitwidth: int = 4,
    ) -> list[float]:
        """Compute analogue MVM through the optical signal path.

        Accepts integer weight matrix and input vector.  Returns a list of
        photocurrents (one per output row), with optional nearest-neighbour
        crosstalk applied across output channels.
        """
        results: list[float] = []
        for row in weight_matrix:
            results.append(self.compute_dot_product_analogue(input_vector, row, bitwidth))

        # Apply crosstalk across output channels
        if self.noise_config.crosstalk_fraction > 0:
            results = apply_crosstalk(results, self.noise_config.crosstalk_fraction)

        return results


@dataclass(frozen=True)
class MZI_OMAC:
    """MZI-mesh based Optical MAC unit.

    Uses a Clements/Reck MZI mesh for unitary transformation instead of
    an MRR weight bank.  Suitable for coherent ONN architectures.
    """

    matrix_size: int = 16
    mesh_config: MZIMeshConfig = MZIMeshConfig()
    laser: Laser = Laser()
    pd: Photodetector = Photodetector()
    wg: Waveguide = Waveguide()
    dac: DAC = DAC()
    adc: ADC = ADC()
    tia: TIA = TIA()

    def num_mzis(self) -> int:
        return self.mesh_config.num_mzis(self.matrix_size)

    def phase_shifter_power_mw(self) -> float:
        return self.mesh_config.total_phase_shifter_power_mw(self.matrix_size)

    def worst_case_transmission(self) -> float:
        return self.mesh_config.mesh_transmission(self.matrix_size)
