"""Bit-serial arithmetic blocks for the optical neural network simulator."""

from dataclasses import dataclass

from encoding import to_bits
from devices.mrr import MRRConfig, mrr_multiply_bit


@dataclass(frozen=True)
class PartialProduct:
    """One bit-level multiplication result before shift accumulation."""

    a_index: int
    b_index: int
    a_bit: int
    b_bit: int
    value: int
    shift: int


def build_partial_products(
    a: int,
    b: int,
    bitwidth: int = 4,
    mrr_config: MRRConfig | None = None,
) -> list[PartialProduct]:
    """Generate all bit-level partial products for a x b.

    For unsigned integers:
        a = sum(a_i * 2^i)
        b = sum(b_j * 2^j)
        a*b = sum((a_i AND b_j) * 2^(i+j))
    """

    a_bits = to_bits(a, bitwidth)
    b_bits = to_bits(b, bitwidth)

    partials: list[PartialProduct] = []
    for i, a_bit in enumerate(a_bits):
        for j, b_bit in enumerate(b_bits):
            partial_bit = mrr_multiply_bit(a_bit, b_bit, mrr_config)
            partials.append(
                PartialProduct(
                    a_index=i,
                    b_index=j,
                    a_bit=a_bit,
                    b_bit=b_bit,
                    value=partial_bit,
                    shift=i + j,
                )
            )
    return partials


def shift_accumulate(partials: list[PartialProduct]) -> int:
    """Sum shifted partial products."""
    total = 0
    for partial in partials:
        total += partial.value << partial.shift
    return total


def bit_serial_multiply(
    a: int,
    b: int,
    bitwidth: int = 4,
    mrr_config: MRRConfig | None = None,
) -> int:
    """Multiply two unsigned integers using bit-serial MRR partial products."""
    partials = build_partial_products(a, b, bitwidth, mrr_config)
    return shift_accumulate(partials)