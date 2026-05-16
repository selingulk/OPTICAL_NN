"""Bit encoding utilities for the optical neural network simulator.

The simulator currently uses unsigned fixed-width integers. Bits are stored
least-significant bit first because bit-serial arithmetic starts from bit 0.
"""


def validate_unsigned_integer(value: int, bitwidth: int) -> None:
    """Validate that value can be represented by the selected bitwidth."""
    if not isinstance(value, int):
        raise TypeError("value must be an integer.")
    if not isinstance(bitwidth, int):
        raise TypeError("bitwidth must be an integer.")
    if bitwidth <= 0:
        raise ValueError("bitwidth must be positive.")
    if value < 0:
        raise ValueError("Only non-negative integers are supported.")
    if value >= (1 << bitwidth):
        raise ValueError(f"{value} does not fit in {bitwidth} bits.")


def to_bits(value: int, bitwidth: int = 4) -> list[int]:
    """Convert an unsigned integer to a fixed-width bit list, LSB first."""
    validate_unsigned_integer(value, bitwidth)
    return [(value >> bit_index) & 1 for bit_index in range(bitwidth)]


def from_bits(bits: list[int]) -> int:
    """Convert an LSB-first bit list back to an integer."""
    value = 0
    for bit_index, bit in enumerate(bits):
        if bit not in (0, 1):
            raise ValueError("bits must contain only 0 or 1.")
        value += bit << bit_index
    return value