"""Electrical periphery models for ONN components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DAC:
    """Digital-to-Analog Converter model."""
    resolution_bits: int = 4
    energy_per_conversion_pj: float = 0.5


@dataclass(frozen=True)
class ADC:
    """Analog-to-Digital Converter model."""
    resolution_bits: int = 4
    energy_per_conversion_pj: float = 1.0


@dataclass(frozen=True)
class TIA:
    """Transimpedance Amplifier model."""
    energy_per_bit_pj: float = 0.2


@dataclass(frozen=True)
class SERDES:
    """Serializer/Deserializer model."""
    energy_per_bit_pj: float = 0.1
