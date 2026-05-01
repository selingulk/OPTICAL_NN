"""

Right now, we have the first skeleton of the system:
numbers are encoded into bits, bits are processed through the ideal MRR logic,
multiplication is built step by step, and then the MVM result is produced.

Before moving on to realistic optical behavior, I wanted a simple way to verify
that these core parts are working properly. This file gives us that control.

So if we later add new features like loss, voltage-dependent MRR states,
or Lumerical-based modeling, we can test whether the original logic still works.

In short, this file is here to protect the foundation of the project.
"""

from encoding import to_bits, from_bits
from mrr import mrr_multiply_bit
from arithmetic import bit_serial_multiply
from core import mvm_bit_serial, mvm_reference
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_VECTOR

def run_tests() -> None:
    # Inform the user that the test sequence has started
    print("Running tests...")

    # -----------------------------
    # encoding.py tests
    # -----------------------------
    # Check whether integer-to-bit conversion works correctly
    assert to_bits(5, 4) == [1, 0, 1, 0], "to_bits failed for 5"

    # Check whether bit-to-integer conversion works correctly
    assert from_bits([1, 0, 1, 0]) == 5, "from_bits failed for [1,0,1,0]"

    # -----------------------------
    # mrr.py tests
    # -----------------------------
    # Verify the ideal MRR logic behaves like bitwise AND
    assert mrr_multiply_bit(1, 1) == 1, "mrr_multiply_bit failed for 1,1"
    assert mrr_multiply_bit(1, 0) == 0, "mrr_multiply_bit failed for 1,0"
    assert mrr_multiply_bit(0, 1) == 0, "mrr_multiply_bit failed for 0,1"
    assert mrr_multiply_bit(0, 0) == 0, "mrr_multiply_bit failed for 0,0"

    # -----------------------------
    # arithmetic.py tests
    # -----------------------------
    # Check whether bit-serial multiplication produces the correct values
    assert bit_serial_multiply(3, 2, 4) == 6, "bit_serial_multiply failed for 3*2"
    assert bit_serial_multiply(3, 3, 4) == 9, "bit_serial_multiply failed for 3*3"
    assert bit_serial_multiply(2, 1, 4) == 2, "bit_serial_multiply failed for 2*1"

    # -----------------------------
    # core.py tests
    # -----------------------------
    # Define a small example matrix and input vector
    

    # Compute the simulated output using our custom bit-serial MVM
    sim_out = mvm_bit_serial(DEFAULT_MATRIX, DEFAULT_VECTOR, BITWIDTH)

    # Compute the reference output using normal arithmetic
    ref_out = mvm_reference(DEFAULT_MATRIX, DEFAULT_VECTOR)

    # Check whether the simulated result matches the expected output
    assert sim_out == [13, 7, 7], "mvm_bit_serial wrong output"

    # Check whether the simulated output matches the mathematical reference
    assert sim_out == ref_out, "mvm_bit_serial and reference do not match"

    # If all assertions pass, print success message
    print("All tests passed successfully.")


# Run the tests only when this file is executed directly
if __name__ == "__main__":
    run_tests()