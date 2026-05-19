"""Non-ideal MRR sweep: How insertion loss and extinction ratio degrade accuracy.

Sweeps MRR parameters and checks when bit-serial MVM starts producing errors.
Generates a 2D heatmap of error rate vs IL and ER.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from accelerator.core import mvm_bit_serial, mvm_reference
from devices.mrr import MRRConfig


def compute_error_rate(
    matrix: list[list[int]],
    vector: list[int],
    bitwidth: int,
    mrr_config: MRRConfig,
) -> float:
    """Compare simulated MVM output to reference and return relative error."""
    ref = mvm_reference(matrix, vector)
    sim = mvm_bit_serial(matrix, vector, bitwidth, mrr_config)

    total_error = sum(abs(r - s) for r, s in zip(ref, sim))
    total_ref = sum(abs(r) for r in ref)
    if total_ref == 0:
        return 0.0
    return total_error / total_ref


def main():
    # Test matrices of different sizes
    matrix_3x3 = [
        [1, 2, 3],
        [0, 1, 2],
        [3, 1, 0],
    ]
    vector_3 = [2, 1, 3]

    matrix_4x4 = [
        [3, 7, 2, 5],
        [1, 0, 4, 6],
        [5, 3, 1, 2],
        [0, 2, 7, 1],
    ]
    vector_4 = [1, 3, 2, 4]

    bitwidth = 4

    # Sweep parameters
    insertion_losses = np.linspace(0.0, 5.0, 20)
    extinction_ratios = np.linspace(5.0, 30.0, 20)

    error_map = np.zeros((len(extinction_ratios), len(insertion_losses)))

    print("--- Non-Ideal MRR Sweep ---")
    print(f"Sweeping IL: {insertion_losses[0]:.1f} -- {insertion_losses[-1]:.1f} dB")
    print(f"Sweeping ER: {extinction_ratios[0]:.1f} -- {extinction_ratios[-1]:.1f} dB")
    print()

    for i, er in enumerate(extinction_ratios):
        for j, il in enumerate(insertion_losses):
            config = MRRConfig(
                insertion_loss_db=float(il),
                extinction_ratio_db=float(er),
                decision_threshold=0.5,
            )
            err1 = compute_error_rate(matrix_3x3, vector_3, bitwidth, config)
            err2 = compute_error_rate(matrix_4x4, vector_4, bitwidth, config)
            error_map[i, j] = (err1 + err2) / 2

    # Find the boundary where errors start
    print("Sample error rates:")
    for il in [0.0, 1.0, 2.0, 3.0, 5.0]:
        for er in [5.0, 10.0, 15.0, 20.0, 30.0]:
            config = MRRConfig(insertion_loss_db=il, extinction_ratio_db=er, decision_threshold=0.5)
            err = compute_error_rate(matrix_3x3, vector_3, bitwidth, config)
            if err == 0:
                status = "OK"
            else:
                status = f"ERROR ({err:.2%})"
            print(f"  IL={il:.1f}dB, ER={er:.1f}dB -> {status}")

    # ---- heatmap plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        error_map,
        aspect="auto",
        origin="lower",
        extent=[insertion_losses[0], insertion_losses[-1],
                extinction_ratios[0], extinction_ratios[-1]],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(0.01, error_map.max()),
    )
    ax.set_xlabel("Insertion Loss (dB)", fontsize=12)
    ax.set_ylabel("Extinction Ratio (dB)", fontsize=12)
    ax.set_title("MVM Error Rate vs MRR Non-Idealities", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Error", fontsize=11)

    # Mark the "safe zone" boundary
    ax.contour(
        insertion_losses,
        extinction_ratios,
        error_map,
        levels=[0.001],
        colors=["white"],
        linewidths=[2],
        linestyles=["--"],
    )

    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "nonideal_sweep.png"), dpi=150)
    print("\nPlot saved to results/nonideal_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
