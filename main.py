from core import mvm_bit_serial, mvm_reference
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_VECTOR


def main() -> None:
    simulated_output = mvm_bit_serial(DEFAULT_MATRIX, DEFAULT_VECTOR, BITWIDTH)
    reference_output = mvm_reference(DEFAULT_MATRIX, DEFAULT_VECTOR)

    print("Matrix:", DEFAULT_MATRIX)
    print("Vector:", DEFAULT_VECTOR)
    print("Bitwidth:", BITWIDTH)
    print()
    print("Simulated output :", simulated_output)
    print("Reference output :", reference_output)
    print("Match            :", simulated_output == reference_output)


if __name__ == "__main__":
    main()