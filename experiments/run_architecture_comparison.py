"""Architecture comparison: MRR-based vs MZI-mesh vs Electronic baseline.

Produces a multi-panel bar chart comparing Power, Throughput, Energy/MAC, and Area.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from accelerator.architecture import ONNTile, MZITile
from accelerator.metrics import MetricsEngine, MZIMetricsEngine, ElectronicBaselineEngine
from accelerator.omac import OMAC, MZI_OMAC
from devices.laser import Laser
from devices.mzi import MZIMeshConfig


def main():
    # ---- define architectures --------------------------------------------------
    # 1) Baseline MRR
    baseline_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    baseline = MetricsEngine(baseline_tile).evaluate()

    # 2) Advanced MRR (better laser, more OMACs)
    adv_laser = Laser(power_mw=5.0, wall_plug_efficiency=0.4)
    adv_omac = OMAC(laser=adv_laser)
    adv_tile = ONNTile(num_omacs=16, clock_frequency_ghz=2, omac_template=adv_omac)
    advanced = MetricsEngine(adv_tile).evaluate()

    # 3) MZI-mesh
    mzi_tile = MZITile(num_omacs=4, clock_frequency_ghz=1)
    mzi = MZIMetricsEngine(mzi_tile).evaluate()

    # 4) Electronic baseline
    electronic = ElectronicBaselineEngine(
        num_mac_units=256, clock_frequency_ghz=1.0
    ).evaluate()

    # ---- print results ---------------------------------------------------------
    configs = [baseline, advanced, mzi, electronic]
    for c in configs:
        print(c)
        print()

    # ---- plot ------------------------------------------------------------------
    names = [c.arch_name for c in configs]
    x = np.arange(len(names))
    colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ONN Architecture Comparison", fontsize=16, fontweight="bold")

    metrics_list = [
        ("Power (mW)", [c.power_mw for c in configs]),
        ("Throughput (TOPS)", [c.throughput_tops for c in configs]),
        ("Energy/MAC (pJ)", [c.energy_per_mac_pj for c in configs]),
        ("Area (mm²)", [c.area_mm2 for c in configs]),
    ]

    for ax, (title, values) in zip(axes.flat, metrics_list):
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "architecture_comparison.png"), dpi=150)
    print("Plot saved to results/architecture_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
