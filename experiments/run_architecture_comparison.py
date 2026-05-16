"""Script to compare different architecture configurations."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.architecture import ONNTile
from accelerator.metrics import MetricsEngine
from accelerator.omac import OMAC
from devices.laser import Laser


def main():
    print("--- Architecture Comparison ---")

    # Arch 1: Baseline
    baseline_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    baseline_metrics = MetricsEngine(baseline_tile).evaluate()

    # Arch 2: High Efficiency Laser + More OMACs
    high_eff_laser = Laser(power_mw=5.0, wall_plug_efficiency=0.4)
    advanced_omac = OMAC(laser=high_eff_laser)
    advanced_tile = ONNTile(num_omacs=16, clock_frequency_ghz=2, omac_template=advanced_omac)
    advanced_metrics = MetricsEngine(advanced_tile).evaluate()

    print("Baseline Architecture (4 OMACs, 1 GHz, 20% WP Eff Laser):")
    print(baseline_metrics)
    print()
    print("Advanced Architecture (16 OMACs, 2 GHz, 40% WP Eff Laser):")
    print(advanced_metrics)


if __name__ == "__main__":
    main()
