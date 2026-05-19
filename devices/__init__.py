"""Device-level models for optical and electrical components."""

from .electrical import ADC, DAC, SERDES, TIA
from .laser import Laser
from .mrr import MRRConfig, mrr_multiply_bit, optical_transmission
from .mzi import MZI, MZIMeshConfig
from .noise import NoiseConfig, add_shot_noise, add_thermal_noise, apply_crosstalk
from .photodetector import Photodetector
from .waveguide import Waveguide
