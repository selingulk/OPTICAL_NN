"""Microring resonator model used by the bit-serial ONN simulator.

This file intentionally starts with an ideal digital abstraction and also keeps
a small non-ideal optical transmission model. The digital result is used by the
current matrix-vector multiplication pipeline, while the transmission model
makes the project easier to extend with Lumerical data later.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MRRConfig:
    """Parameters for a simplified MRR transmission model.

    insertion_loss_db: optical power loss in the ON state.
    extinction_ratio_db: ON/OFF separation in dB.
    decision_threshold: threshold used to convert optical power back to a bit.
    """

    insertion_loss_db: float = 0.0
    extinction_ratio_db: float = 20.0
    decision_threshold: float = 0.5


def _validate_bit(bit: int, name: str) -> None:
    if bit not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1.")


def optical_transmission(input_bit: int, weight_bit: int, config: MRRConfig | None = None) -> float:
    """Return normalized optical output power for one MRR cell.

    If there is no optical input, output power is zero. If input_bit is one,
    weight_bit selects the ON or OFF state of the resonator.
    """

    _validate_bit(input_bit, "input_bit")
    _validate_bit(weight_bit, "weight_bit")

    if config is None:
        config = MRRConfig()

    if input_bit == 0:
        return 0.0

    on_power = 10 ** (-config.insertion_loss_db / 10)
    off_power = on_power * 10 ** (-config.extinction_ratio_db / 10)
    return on_power if weight_bit == 1 else off_power


def mrr_multiply_bit(input_bit: int, weight_bit: int, config: MRRConfig | None = None) -> int:
    """Multiply two bits using the current MRR abstraction.

    In the ideal case this is equivalent to logical AND. The implementation is
    written through optical_transmission so that non-ideal MRR behavior can be
    introduced without changing the arithmetic pipeline.
    """

    if config is None:
        config = MRRConfig()

    power = optical_transmission(input_bit, weight_bit, config)
    return int(power >= config.decision_threshold)