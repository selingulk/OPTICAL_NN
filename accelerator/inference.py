"""End-to-end neural network inference engine.

Maps a full network (MLP or CNN) onto an ONN tile and computes per-layer
and total latency, energy, and cycle counts.
"""

from dataclasses import dataclass

from .architecture import ONNTile, MZITile
from .metrics import MetricsEngine, MZIMetricsEngine


@dataclass
class LayerResult:
    """Inference result for a single layer."""

    layer_name: str
    macs: int
    cycles: int
    latency_us: float
    energy_nj: float


@dataclass
class InferenceResult:
    """Full inference result for a network."""

    network_name: str
    arch_name: str
    layer_results: list[LayerResult]

    @property
    def total_macs(self) -> int:
        return sum(lr.macs for lr in self.layer_results)

    @property
    def total_cycles(self) -> int:
        return sum(lr.cycles for lr in self.layer_results)

    @property
    def total_latency_us(self) -> float:
        return sum(lr.latency_us for lr in self.layer_results)

    @property
    def total_energy_nj(self) -> float:
        return sum(lr.energy_nj for lr in self.layer_results)

    def __str__(self) -> str:
        lines = [
            f"Inference Report: {self.network_name} on {self.arch_name}",
            f"{'Layer':<30} {'MACs':>12} {'Cycles':>10} {'Latency(us)':>12} {'Energy(nJ)':>12}",
            "-" * 78,
        ]
        for lr in self.layer_results:
            lines.append(
                f"{lr.layer_name:<30} {lr.macs:>12,} {lr.cycles:>10,} "
                f"{lr.latency_us:>12.4f} {lr.energy_nj:>12.4f}"
            )
        lines.append("-" * 78)
        lines.append(
            f"{'TOTAL':<30} {self.total_macs:>12,} {self.total_cycles:>10,} "
            f"{self.total_latency_us:>12.4f} {self.total_energy_nj:>12.4f}"
        )
        return "\n".join(lines)


class InferenceEngine:
    """Maps neural network workloads onto an ONN tile for inference analysis."""

    def __init__(self, tile: ONNTile | MZITile, bitwidth: int = 4):
        self.tile = tile
        self.bitwidth = bitwidth

        # Evaluate PPA to get energy/MAC
        if isinstance(tile, MZITile):
            self._metrics = MZIMetricsEngine(tile).evaluate()
        else:
            self._metrics = MetricsEngine(tile).evaluate()

    def run_mlp(self, mlp) -> InferenceResult:
        """Run inference for a SimpleMLP workload."""
        layer_results = []
        for i, layer in enumerate(mlp.layers):
            name = f"Linear({layer.in_features}->{layer.out_features})"
            macs = layer.num_macs()
            cycles = self.tile.cycles_for_mvm(layer.out_features, layer.in_features, self.bitwidth)
            latency = self.tile.latency_us(layer.out_features, layer.in_features, self.bitwidth)
            energy = (macs * self._metrics.energy_per_mac_pj) / 1000.0  # pJ -> nJ

            layer_results.append(LayerResult(name, macs, cycles, latency, energy))

        return InferenceResult(
            network_name="SimpleMLP",
            arch_name=self._metrics.arch_name,
            layer_results=layer_results,
        )

    def run_cnn(self, cnn) -> InferenceResult:
        """Run inference for a SimpleCNN workload."""
        layer_results = []

        for i, layer in enumerate(cnn.conv_layers):
            # Map conv layer as im2col MVM: rows = output_pixels * out_channels,
            # cols = kernel_size^2 * in_channels
            out_spatial = max(layer.input_spatial_size - layer.kernel_size + 1, 1)
            out_pixels = out_spatial * out_spatial
            matrix_rows = layer.out_channels
            matrix_cols = layer.in_channels * layer.kernel_size * layer.kernel_size

            name = f"Conv2D({layer.in_channels}->{layer.out_channels}, k={layer.kernel_size})"
            macs = layer.num_macs()
            cycles = self.tile.cycles_for_mvm(matrix_rows, matrix_cols, self.bitwidth) * out_pixels
            latency = self.tile.latency_us(matrix_rows, matrix_cols, self.bitwidth) * out_pixels
            energy = (macs * self._metrics.energy_per_mac_pj) / 1000.0

            layer_results.append(LayerResult(name, macs, cycles, latency, energy))

        for i, layer in enumerate(cnn.linear_layers):
            name = f"Linear({layer.in_features}->{layer.out_features})"
            macs = layer.num_macs()
            cycles = self.tile.cycles_for_mvm(layer.out_features, layer.in_features, self.bitwidth)
            latency = self.tile.latency_us(layer.out_features, layer.in_features, self.bitwidth)
            energy = (macs * self._metrics.energy_per_mac_pj) / 1000.0

            layer_results.append(LayerResult(name, macs, cycles, latency, energy))

        return InferenceResult(
            network_name="SimpleCNN",
            arch_name=self._metrics.arch_name,
            layer_results=layer_results,
        )
