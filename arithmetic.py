from encoding import to_bits
from mrr import mrr_multiply_bit


def shift_accumulate(partials: list[tuple[int, int]]) -> int:
    """
    Sum partial products with their shift values.

    Each item in partials is:
    (partial_bit, shift_amount)
    """
    total = 0
    for value, shift in partials:
        total += value << shift
    return total


def bit_serial_multiply(a: int, b: int, bitwidth: int = 4) -> int:
    """
    Multiply two non-negative integers using bit-serial logic.

    a = sum(a_i * 2^i)
    b = sum(b_j * 2^j)

    product = sum((a_i AND b_j) * 2^(i+j))
    """
    if a < 0 or b < 0:
        raise ValueError("Only non-negative integers are supported in this first version.")

    a_bits = to_bits(a, bitwidth)
    b_bits = to_bits(b, bitwidth)

    partials = []

    for i, a_bit in enumerate(a_bits):
        for j, b_bit in enumerate(b_bits):
            partial_bit = mrr_multiply_bit(a_bit, b_bit)
            partials.append((partial_bit, i + j))

    return shift_accumulate(partials)