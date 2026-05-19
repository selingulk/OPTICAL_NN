"""End-to-end neural network inference simulation.

Maps MLP and CNN workloads onto MRR-based and MZI-based ONN tiles,
producing per-layer latency/energy breakdowns and comparison plots.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from accelerator.architecture import ONNTile, MZITile
from accelerator.inference import InferenceEngine
from workloads.networks import SimpleMLP, SimpleCNN


def main():
    # ---- define architectures --------------------------------------------------
    mrr_tile = ONNTile(num_omacs=4, clock_frequency_ghz=2)
    mzi_tile = MZITile(num_omacs=4, clock_frequency_ghz=2)

    # ---- define workloads ------------------------------------------------------
    mlp = SimpleMLP.get_mnist_mlp()
    cnn = SimpleCNN.get_simple_cnn()
    medmnist_cnn = SimpleCNN.get_medmnist_cnn(in_channels=3, num_classes=10)

    # ---- run inference ---------------------------------------------------------
    print("=" * 80)
    print("END-TO-END NEURAL NETWORK INFERENCE SIMULATION (incl. MedMNIST)")
    print("=" * 80)

    results = []

    for tile, name in [(mrr_tile, "MRR"), (mzi_tile, "MZI")]:
        engine = InferenceEngine(tile, bitwidth=4)

        mlp_result = engine.run_mlp(mlp)
        cnn_result = engine.run_cnn(cnn)
        medmnist_result = engine.run_cnn(medmnist_cnn)
        results.append((name, mlp_result, cnn_result, medmnist_result))

        print(f"\n{'-' * 80}")
        print(f"Architecture: {name}")
        print(mlp_result)
        print()
        print(cnn_result)
        print()
        print(medmnist_result)

    # ---- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("End-to-End Neural Network Inference Analysis", fontsize=15, fontweight="bold")

    # Unpack results
    mrr_name, mrr_mlp, mrr_cnn, mrr_medmnist = results[0]
    mzi_name, mzi_mlp, mzi_cnn, mzi_medmnist = results[1]

    # --- Plot 1: MLP per-layer energy breakdown (stacked bar) ---
    ax = axes[0, 0]
    mlp_layers_mrr = [lr.layer_name for lr in mrr_mlp.layer_results]
    mlp_energy_mrr = [lr.energy_nj for lr in mrr_mlp.layer_results]
    mlp_energy_mzi = [lr.energy_nj for lr in mzi_mlp.layer_results]
    x = np.arange(len(mlp_layers_mrr))
    w = 0.35
    ax.bar(x - w / 2, mlp_energy_mrr, w, label="MRR", color="#4FC3F7", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, mlp_energy_mzi, w, label="MZI", color="#FFB74D", edgecolor="black", linewidth=0.5)
    ax.set_title("MLP: Energy per Layer", fontweight="bold")
    ax.set_ylabel("Energy (nJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(mlp_layers_mrr, rotation=15, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Plot 2: MLP per-layer latency ---
    ax = axes[0, 1]
    mlp_lat_mrr = [lr.latency_us for lr in mrr_mlp.layer_results]
    mlp_lat_mzi = [lr.latency_us for lr in mzi_mlp.layer_results]
    ax.bar(x - w / 2, mlp_lat_mrr, w, label="MRR", color="#4FC3F7", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, mlp_lat_mzi, w, label="MZI", color="#FFB74D", edgecolor="black", linewidth=0.5)
    ax.set_title("MLP: Latency per Layer", fontweight="bold")
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(mlp_layers_mrr, rotation=15, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Plot 3: CNN per-layer energy ---
    ax = axes[1, 0]
    cnn_layers_mrr = [lr.layer_name for lr in mrr_cnn.layer_results]
    cnn_energy_mrr = [lr.energy_nj for lr in mrr_cnn.layer_results]
    cnn_energy_mzi = [lr.energy_nj for lr in mzi_cnn.layer_results]
    x2 = np.arange(len(cnn_layers_mrr))
    ax.bar(x2 - w / 2, cnn_energy_mrr, w, label="MRR", color="#4FC3F7", edgecolor="black", linewidth=0.5)
    ax.bar(x2 + w / 2, cnn_energy_mzi, w, label="MZI", color="#FFB74D", edgecolor="black", linewidth=0.5)
    ax.set_title("CNN: Energy per Layer", fontweight="bold")
    ax.set_ylabel("Energy (nJ)")
    ax.set_xticks(x2)
    ax.set_xticklabels(cnn_layers_mrr, rotation=15, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- Plot 4: Total comparison ---
    ax = axes[1, 1]
    categories = ["MLP\nEnergy", "CNN\nEnergy", "MedMNIST\nEnergy", "MedMNIST\nLatency (us)"]
    mrr_vals = [mrr_mlp.total_energy_nj, mrr_cnn.total_energy_nj,
                mrr_medmnist.total_energy_nj, mrr_medmnist.total_latency_us]
    mzi_vals = [mzi_mlp.total_energy_nj, mzi_cnn.total_energy_nj,
                mzi_medmnist.total_energy_nj, mzi_medmnist.total_latency_us]
    x3 = np.arange(len(categories))
    ax.bar(x3 - w / 2, mrr_vals, w, label="MRR", color="#4FC3F7", edgecolor="black", linewidth=0.5)
    ax.bar(x3 + w / 2, mzi_vals, w, label="MZI", color="#FFB74D", edgecolor="black", linewidth=0.5)
    ax.set_title("Total Inference: MRR vs MZI", fontweight="bold")
    ax.set_xticks(x3)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "inference_analysis.png"), dpi=150)
    print("\nPlot saved to results/inference_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
