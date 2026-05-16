"""Central configuration for the optical neural network simulator."""

from devices.mrr import MRRConfig

BITWIDTH = 4

DEFAULT_MATRIX = [
    [1, 2, 3],
    [0, 1, 2],
    [3, 1, 0],
]

DEFAULT_VECTOR = [2, 1, 3]

DEFAULT_MRR_CONFIG = MRRConfig(
    insertion_loss_db=0.0,
    extinction_ratio_db=20.0,
    decision_threshold=0.5,
)