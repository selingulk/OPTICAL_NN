"""Script to run an MVM demonstration with the complete metrics engine."""

import sys
import os

# Add parent dir to sys.path to run directly from experiments/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.architecture import ONNTile
from accelerator.metrics import MetricsEngine


def main():
    print("--- ONN MVM Demo ---")
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    engine = MetricsEngine(tile)

    print("Evaluating baseline 4-OMAC tile architecture...")
    metrics = engine.evaluate()
    print(metrics)
    print("--------------------")

if __name__ == "__main__":
    main()
