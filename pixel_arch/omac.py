"""Optical Multiply-Accumulate (OMAC) core models specifically for PIXEL architectures.

This includes:
- OE_OMAC: Optoelectronic (optical multiply, electrical shift-accumulate)
- OO_OMAC: All-Optical (optical multiply, MZI delay line accumulate)
"""

from dataclasses import dataclass, field
from collections import deque

from accelerator.omac import OMAC
from devices.electrical import ADC, DAC, TIA
from devices.laser import Laser
from devices.noise import NoiseConfig, add_shot_noise, add_thermal_noise
from devices.photodetector import Photodetector
from devices.waveguide import Waveguide
from pixel_arch.pixel_devices import PixelMRRConfig, ThresholdStrategy, pixel_optical_transmission


@dataclass(frozen=True)
class Pixel_OE_OMAC(OMAC):
    """PIXEL Optoelectronic Architecture.
    
    Multiplication (AND) is done via MRRs.
    Accumulation is done electrically using a simulated Carry Lookahead Adder (CLA).
    """
    pixel_mrr_config: PixelMRRConfig = PixelMRRConfig()

    def compute_dot_product_analogue(
        self,
        inputs: list[int],
        weights: list[int],
        bitwidth: int = 4,
    ) -> float:
        """Compute OE dot product. Light is converted to electricity immediately, then summed."""
        if len(inputs) != len(weights):
            raise ValueError("inputs and weights must have the same length.")

        laser_power = self._laser_output_power()
        rng = self.noise_config.get_rng()
        max_val = (1 << bitwidth) - 1
        if max_val == 0:
            max_val = 1

        dynamic_threshold_mw = self.pixel_mrr_config.get_dynamic_threshold()
        # Scale expected threshold by laser power and waveguide loss to be physically consistent
        threshold_power_mw = self.signal_path_single(laser_power, dynamic_threshold_mw)
        # Convert the optical power threshold to an expected photocurrent threshold
        threshold_ua = self.pd.get_photocurrent_ua(threshold_power_mw)

        total_electrical_sum = 0.0

        for inp, wt in zip(inputs, weights):
            inp_norm = inp / max_val
            wt_norm = wt / max_val

            # Validate that inputs and weights are strictly binary (0 or 1 at the physical level)
            if inp_norm not in (0.0, 1.0) or wt_norm not in (0.0, 1.0):
                raise ValueError(
                    f"Pixel_OE_OMAC only supports binary (0 or 1) inputs and weights "
                    f"at the normalized level. Got input={inp} (normalized {inp_norm}) "
                    f"and weight={wt} (normalized {wt_norm}) with bitwidth={bitwidth}."
                )

            input_bit = 1 if inp_norm == 1.0 else 0
            weight_bit = 1 if wt_norm == 1.0 else 0

            input_power = laser_power * inp_norm
            
            # Incorporate physical transmission with drift
            mrr_trans = pixel_optical_transmission(input_bit, weight_bit, self.pixel_mrr_config, rng)
            
            pwr = self.signal_path_single(input_power, mrr_trans)

            # Noise injection at the individual MRR/PD level
            if self.noise_config.shot_noise_enabled:
                pwr = add_shot_noise(pwr, rng)
            if self.noise_config.thermal_noise_std > 0:
                pwr = add_thermal_noise(pwr, self.noise_config.thermal_noise_std, rng)

            # Convert to electricity IMMEDIATELY
            photocurrent_ua = self.pd.get_photocurrent_ua(pwr)

            # TIA Decision (Digitization)
            bit_value = 1 if photocurrent_ua >= threshold_ua else 0

            # Electrical Accumulation
            total_electrical_sum += bit_value

        return total_electrical_sum


@dataclass(frozen=True)
class Pixel_OO_OMAC(OMAC):
    """PIXEL All-Optical Architecture.
    
    Multiplication is done via MRRs.
    Accumulation is done purely optically using MZI delay lines.
    """
    pixel_mrr_config: PixelMRRConfig = PixelMRRConfig()

    def compute_dot_product_analogue(
        self,
        inputs: list[int],
        weights: list[int],
        bitwidth: int = 4,
    ) -> float:
        """Compute OO dot product using an optical delay line."""
        if len(inputs) != len(weights):
            raise ValueError("inputs and weights must have the same length.")

        laser_power = self._laser_output_power()
        rng = self.noise_config.get_rng()
        max_val = (1 << bitwidth) - 1
        if max_val == 0:
            max_val = 1

        # MZI Delay Line Simulation (Queue)
        # In the PIXEL paper, light pulses are delayed by mz_delay = c/(n*f) - d_mzi
        # We abstract this by having a cycle-accurate delay queue.
        # All pulses from the array eventually superimpose in the same time bin.
        optical_delay_line = deque([0.0] * len(inputs))
        
        # Superimpose optically
        for i, (inp, wt) in enumerate(zip(inputs, weights)):
            inp_norm = inp / max_val
            wt_norm = wt / max_val

            # Validate that inputs and weights are strictly binary (0 or 1 at the physical level)
            if inp_norm not in (0.0, 1.0) or wt_norm not in (0.0, 1.0):
                raise ValueError(
                    f"Pixel_OO_OMAC only supports binary (0 or 1) inputs and weights "
                    f"at the normalized level. Got input={inp} (normalized {inp_norm}) "
                    f"and weight={wt} (normalized {wt_norm}) with bitwidth={bitwidth}."
                )

            input_bit = 1 if inp_norm == 1.0 else 0
            weight_bit = 1 if wt_norm == 1.0 else 0

            input_power = laser_power * inp_norm
            
            # Incorporate physical transmission with drift
            mrr_trans = pixel_optical_transmission(input_bit, weight_bit, self.pixel_mrr_config, rng)
            
            pwr = self.signal_path_single(input_power, mrr_trans)
            
            # The pulse is injected into the delay line.
            # In a perfectly aligned delay line, they all accumulate into the final pulse.
            # We add it to the corresponding 'time bin' in our delay line.
            optical_delay_line[0] += pwr

        # The accumulated optical pulse finally exits the delay line
        total_optical_power = optical_delay_line.popleft()

        # Noise injection happens ONCE at the single photodetector
        if self.noise_config.shot_noise_enabled:
            total_optical_power = add_shot_noise(total_optical_power, rng)
        if self.noise_config.thermal_noise_std > 0:
            total_optical_power = add_thermal_noise(total_optical_power, self.noise_config.thermal_noise_std, rng)

        # Photodetector conversion
        photocurrent_ua = self.pd.get_photocurrent_ua(total_optical_power)
        return photocurrent_ua
