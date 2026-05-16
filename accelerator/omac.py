"""Optical Multiply-Accumulate (OMAC) core models."""

from dataclasses import dataclass
from devices.electrical import ADC, DAC, SERDES, TIA
from devices.laser import Laser
from devices.mrr import MRRConfig
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
