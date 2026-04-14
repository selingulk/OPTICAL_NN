
from arithmetic import bit_serial_multiply


def dot_product_bit_serial(weights: list[int], inputs: list[int], bitwidth: int = 4) -> int:
    """Compute one dot product using bit-serial multiplication."""
    if len(weights) != len(inputs):
        raise ValueError("weights and inputs must have the same length.")

    total = 0
    for w, x in zip(weights, inputs):
        total += bit_serial_multiply(w, x, bitwidth)

    return total


def mvm_bit_serial(matrix: list[list[int]], vector: list[int], bitwidth: int = 4) -> list[int]:
    """Compute matrix-vector multiplication using bit-serial logic."""
    outputs = []
    for row in matrix:
        outputs.append(dot_product_bit_serial(row, vector, bitwidth))
    return outputs


def dot_product_reference(weights: list[int], inputs: list[int]) -> int:
    """Reference dot product using normal arithmetic."""
    if len(weights) != len(inputs):
        raise ValueError("weights and inputs must have the same length.")

    return sum(w * x for w, x in zip(weights, inputs))


def mvm_reference(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """Reference matrix-vector multiplication using normal arithmetic."""
    return [dot_product_reference(row, vector) for row in matrix]