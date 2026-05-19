"""Run a small demonstration of the optical NN matrix-vector core."""

from accelerator.arithmetic import build_partial_products, bit_serial_multiply
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_MRR_CONFIG, DEFAULT_VECTOR
from accelerator.core import mvm_bit_serial, mvm_reference


def print_partial_product_trace(a: int, b: int, bitwidth: int) -> None:
    """Print a compact trace for one multiplication example."""
    print(f"\nTrace for bit-serial multiplication: {a} x {b}")
    print("a_i b_j -> partial << shift")
    for partial in build_partial_products(a, b, bitwidth, DEFAULT_MRR_CONFIG):
        if partial.value == 1:
            print(
                f"a{partial.a_index}={partial.a_bit} "
                f"b{partial.b_index}={partial.b_bit} -> "
                f"{partial.value} << {partial.shift}"
            )
    print("Result:", bit_serial_multiply(a, b, bitwidth, DEFAULT_MRR_CONFIG))


def main() -> None:
    simulated_output = mvm_bit_serial(
        DEFAULT_MATRIX,
        DEFAULT_VECTOR,
        BITWIDTH,
        DEFAULT_MRR_CONFIG,
    )
    reference_output = mvm_reference(DEFAULT_MATRIX, DEFAULT_VECTOR)

    print("Optical Neural Network MVM Simulator")
    print("Matrix:", DEFAULT_MATRIX)
    print("Vector:", DEFAULT_VECTOR)
    print("Bitwidth:", BITWIDTH)
    print()
    print("Simulated output :", simulated_output)
    print("Reference output :", reference_output)
    print("Match            :", simulated_output == reference_output)

    print_partial_product_trace(a=3, b=2, bitwidth=BITWIDTH)

    print("\n" + "=" * 80)
    print("Running full benchmark suite (PIXEL + Baselines)...")
    from experiments.run_all_experiments import main as run_benchmarks
    run_benchmarks()


if __name__ == "__main__":
    # Support multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()