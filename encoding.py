def to_bits(n: int, bitwidth: int = 4) -> list[int]:
    """Convert a non-negative integer to a bit list (LSB first)."""
    if n < 0:
        raise ValueError("Only non-negative integers are supported in this first version.")
    return [(n >> i) & 1 for i in range(bitwidth)]


def from_bits(bits: list[int]) -> int:
    """Convert a bit list (LSB first) back to an integer."""
    value = 0
    for i, bit in enumerate(bits):
        value += (bit & 1) << i
    return value