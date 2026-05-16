"""Simple test suite for the optical NN simulator.

Run with:
    python tests.py

The file intentionally avoids external dependencies so that the repository can
be cloned and tested immediately.
"""

from accelerator.arithmetic import bit_serial_multiply, build_partial_products
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_MRR_CONFIG, DEFAULT_VECTOR
from accelerator.core import dot_product_bit_serial, mvm_bit_serial, mvm_reference
from encoding import from_bits, to_bits
from devices.mrr import mrr_multiply_bit, optical_transmission


def assert_raises(expected_error: type[Exception], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except expected_error:
        return
    raise AssertionError(f"Expected {expected_error.__name__} was not raised.")


def test_encoding_round_trip() -> None:
    for value in range(16):
        bits = to_bits(value, BITWIDTH)
        assert from_bits(bits) == value


def test_encoding_validation() -> None:
    assert_raises(ValueError, to_bits, -1, BITWIDTH)
    assert_raises(ValueError, to_bits, 16, BITWIDTH)
    assert_raises(ValueError, from_bits, [1, 0, 2])


def test_mrr_bit_behavior() -> None:
    assert mrr_multiply_bit(0, 0) == 0
    assert mrr_multiply_bit(0, 1) == 0
    assert mrr_multiply_bit(1, 0) == 0
    assert mrr_multiply_bit(1, 1) == 1
    assert optical_transmission(1, 1, DEFAULT_MRR_CONFIG) > optical_transmission(1, 0, DEFAULT_MRR_CONFIG)


def test_bit_serial_multiply_matches_python() -> None:
    for a in range(16):
        for b in range(16):
            assert bit_serial_multiply(a, b, BITWIDTH, DEFAULT_MRR_CONFIG) == a * b


def test_partial_products_for_3_times_2() -> None:
    partials = build_partial_products(3, 2, BITWIDTH, DEFAULT_MRR_CONFIG)
    active_terms = [(p.a_index, p.b_index, p.shift) for p in partials if p.value == 1]
    assert active_terms == [(0, 1, 1), (1, 1, 2)]
    assert bit_serial_multiply(3, 2, BITWIDTH, DEFAULT_MRR_CONFIG) == 6


def test_dot_product_and_mvm() -> None:
    assert dot_product_bit_serial([1, 2, 3], [2, 1, 3], BITWIDTH, DEFAULT_MRR_CONFIG) == 13
    assert mvm_bit_serial(DEFAULT_MATRIX, DEFAULT_VECTOR, BITWIDTH, DEFAULT_MRR_CONFIG) == mvm_reference(
        DEFAULT_MATRIX,
        DEFAULT_VECTOR,
    )


def test_dimension_validation() -> None:
    assert_raises(ValueError, dot_product_bit_serial, [1, 2], [1], BITWIDTH)
    assert_raises(ValueError, mvm_bit_serial, [[1, 2], [3]], [1, 2], BITWIDTH)
    assert_raises(ValueError, mvm_bit_serial, [[1, 2]], [1, 2, 3], BITWIDTH)


def run_all_tests() -> None:
    test_encoding_round_trip()
    test_encoding_validation()
    test_mrr_bit_behavior()
    test_bit_serial_multiply_matches_python()
    test_partial_products_for_3_times_2()
    test_dot_product_and_mvm()
    test_dimension_validation()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()