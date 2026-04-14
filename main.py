
from core import mvm_bit_serial, mvm_reference


def main() -> None:
    # Small unsigned integer test case
    matrix = [
        [1, 2, 3],
        [0, 1, 2],
        [3, 1, 0],
    ]

    vector = [2, 1, 3]
    bitwidth = 4

    simulated_output = mvm_bit_serial(matrix, vector, bitwidth)
    reference_output = mvm_reference(matrix, vector)

    print("Matrix:", matrix)
    print("Vector:", vector)
    print("Bitwidth:", bitwidth)
    print()
    print("Simulated output :", simulated_output)
    print("Reference output :", reference_output)
    print("Match            :", simulated_output == reference_output)


if __name__ == "__main__":
    main()