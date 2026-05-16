"""Matrix-vector multiplication core for the optical NN simulator."""

from accelerator.arithmetic import bit_serial_multiply
from devices.mrr import MRRConfig


def _validate_vector(vector: list[int], name: str) -> None:
    if not vector:
        raise ValueError(f"{name} must not be empty.")
    if not all(isinstance(value, int) for value in vector):
        raise TypeError(f"{name} must contain only integers.")


def _validate_matrix(matrix: list[list[int]]) -> None:
    if not matrix:
        raise ValueError("matrix must not be empty.")
    row_length = len(matrix[0])
    if row_length == 0:
        raise ValueError("matrix rows must not be empty.")
    for row in matrix:
        if len(row) != row_length:
            raise ValueError("all matrix rows must have the same length.")
        _validate_vector(row, "matrix row")


def dot_product_bit_serial(
    weights: list[int],
    inputs: list[int],
    bitwidth: int = 4,
    mrr_config: MRRConfig | None = None,
) -> int:
    """Compute one dot product using bit-serial multiplication."""
    _validate_vector(weights, "weights")
    _validate_vector(inputs, "inputs")
    if len(weights) != len(inputs):
        raise ValueError("weights and inputs must have the same length.")

    total = 0
    for weight, input_value in zip(weights, inputs):
        total += bit_serial_multiply(weight, input_value, bitwidth, mrr_config)
    return total


def mvm_bit_serial(
    matrix: list[list[int]],
    vector: list[int],
    bitwidth: int = 4,
    mrr_config: MRRConfig | None = None,
) -> list[int]:
    """Compute matrix-vector multiplication with the bit-serial ONN core."""
    _validate_matrix(matrix)
    _validate_vector(vector, "vector")
    if len(matrix[0]) != len(vector):
        raise ValueError("matrix column count must match vector length.")

    return [dot_product_bit_serial(row, vector, bitwidth, mrr_config) for row in matrix]


def dot_product_reference(weights: list[int], inputs: list[int]) -> int:
    """Reference dot product using normal Python arithmetic."""
    _validate_vector(weights, "weights")
    _validate_vector(inputs, "inputs")
    if len(weights) != len(inputs):
        raise ValueError("weights and inputs must have the same length.")
    return sum(weight * input_value for weight, input_value in zip(weights, inputs))


def mvm_reference(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """Reference matrix-vector multiplication using normal arithmetic."""
    _validate_matrix(matrix)
    _validate_vector(vector, "vector")
    if len(matrix[0]) != len(vector):
        raise ValueError("matrix column count must match vector length.")
    return [dot_product_reference(row, vector) for row in matrix]