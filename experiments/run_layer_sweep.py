"""Script to sweep layer sizes and calculate PPA."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.architecture import ONNTile
from accelerator.metrics import MetricsEngine
from workloads.cnn_layers import LinearLayer


def main():
    print("--- Layer Sweep Experiment ---")
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=2)
    engine = MetricsEngine(tile)
    metrics = engine.evaluate()

    layers = [
        LinearLayer(16, 16),
        LinearLayer(64, 64),
        LinearLayer(256, 256),
        LinearLayer(1024, 1024),
    ]

    print(f"Architecture: {tile.num_omacs} OMACs @ {tile.clock_frequency_ghz} GHz")
    print(f"Energy per MAC: {metrics.energy_per_mac_pj:.4f} pJ")

    print("\nResults:")
    for layer in layers:
        macs = layer.num_macs()
        total_energy_nj = (macs * metrics.energy_per_mac_pj) / 1000.0
        print(f"Layer {layer.in_features}x{layer.out_features} ({macs} MACs) -> Energy: {total_energy_nj:.4f} nJ")


if __name__ == "__main__":
    main()
