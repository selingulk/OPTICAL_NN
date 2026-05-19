"""Layer sweep: How energy and latency scale with layer size.

Produces scaling plots for different layer dimensions on the ONN tile.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from accelerator.architecture import ONNTile
from accelerator.metrics import MetricsEngine
from workloads.cnn_layers import LinearLayer


def main():
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=2)
    metrics = MetricsEngine(tile).evaluate()
    bitwidth = 4

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    macs_list = []
    energy_list = []
    latency_list = []
    cycles_list = []

    print("--- Layer Sweep Experiment ---")
    print(f"Architecture: {tile.num_omacs} OMACs @ {tile.clock_frequency_ghz} GHz")
    print(f"Energy per MAC: {metrics.energy_per_mac_pj:.4f} pJ")
    print()

    for sz in sizes:
        layer = LinearLayer(sz, sz)
        macs = layer.num_macs()
        energy_nj = (macs * metrics.energy_per_mac_pj) / 1000.0
        cycles = tile.cycles_for_mvm(sz, sz, bitwidth)
        latency = tile.latency_us(sz, sz, bitwidth)

        macs_list.append(macs)
        energy_list.append(energy_nj)
        latency_list.append(latency)
        cycles_list.append(cycles)

        print(f"Layer {sz:>4}×{sz:<4}  MACs: {macs:>10,}  "
              f"Cycles: {cycles:>8,}  Latency: {latency:>10.4f} µs  "
              f"Energy: {energy_nj:>10.4f} nJ")

    # ---- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Layer Size Scaling Analysis", fontsize=14, fontweight="bold")

    # Energy vs Layer Size
    axes[0].plot(sizes, energy_list, "o-", color="#4FC3F7", linewidth=2, markersize=6)
    axes[0].set_xlabel("Layer Size (N×N)")
    axes[0].set_ylabel("Energy (nJ)")
    axes[0].set_title("Energy vs Layer Size")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Latency vs Layer Size
    axes[1].plot(sizes, latency_list, "s-", color="#81C784", linewidth=2, markersize=6)
    axes[1].set_xlabel("Layer Size (N×N)")
    axes[1].set_ylabel("Latency (µs)")
    axes[1].set_title("Latency vs Layer Size")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    # Cycles vs Layer Size
    axes[2].plot(sizes, cycles_list, "^-", color="#FFB74D", linewidth=2, markersize=6)
    axes[2].set_xlabel("Layer Size (N×N)")
    axes[2].set_ylabel("Cycles")
    axes[2].set_title("Cycle Count vs Layer Size")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "layer_sweep.png"), dpi=150)
    print("\nPlot saved to results/layer_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
