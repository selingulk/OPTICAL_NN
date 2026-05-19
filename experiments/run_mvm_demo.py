"""MVM demonstration with the full optical signal path.

Runs the bit-serial MVM core and also demonstrates the analogue signal-path
simulation through the OMAC (laser -> MRR -> waveguide -> PD).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.arithmetic import build_partial_products, bit_serial_multiply
from accelerator.core import mvm_bit_serial, mvm_reference
from accelerator.omac import OMAC
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_MRR_CONFIG, DEFAULT_VECTOR
from devices.noise import NoiseConfig


def main():
    print("=" * 60)
    print("  Optical Neural Network -- MVM Demonstration")
    print("=" * 60)

    # ---- Bit-serial MVM --------------------------------------------------------
    simulated = mvm_bit_serial(DEFAULT_MATRIX, DEFAULT_VECTOR, BITWIDTH, DEFAULT_MRR_CONFIG)
    reference = mvm_reference(DEFAULT_MATRIX, DEFAULT_VECTOR)

    print("\n1) Bit-Serial MVM Core")
    print(f"   Matrix:  {DEFAULT_MATRIX}")
    print(f"   Vector:  {DEFAULT_VECTOR}")
    print(f"   Bitwidth: {BITWIDTH}")
    print(f"   Simulated: {simulated}")
    print(f"   Reference: {reference}")
    print(f"   Match:     {simulated == reference}")

    # ---- Partial product trace -------------------------------------------------
    a, b = 3, 2
    print(f"\n2) Partial Product Trace: {a} x {b}")
    print("   a_i  b_j -> partial << shift")
    for p in build_partial_products(a, b, BITWIDTH, DEFAULT_MRR_CONFIG):
        if p.value == 1:
            print(f"   a{p.a_index}={p.a_bit}  b{p.b_index}={p.b_bit} ->     {p.value}  << {p.shift}")
    print(f"   Result: {bit_serial_multiply(a, b, BITWIDTH, DEFAULT_MRR_CONFIG)}")

    # ---- Analogue signal-path demo ---------------------------------------------
    print("\n3) Analogue Signal-Path Simulation (OMAC)")
    omac = OMAC(num_inputs=3, num_outputs=3, mrr_config=DEFAULT_MRR_CONFIG)

    print(f"   Laser power: {omac.laser.power_mw} mW  (WPE: {omac.laser.wall_plug_efficiency})")
    print(f"   Waveguide loss: {omac.wg.loss_db_per_cm} dB/cm")
    print(f"   PD responsivity: {omac.pd.responsivity_a_w} A/W")

    # Ideal (no noise)
    results_ideal = omac.compute_mvm_analogue(DEFAULT_MATRIX, DEFAULT_VECTOR)
    print(f"\n   Ideal photocurrents (uA): {[f'{v:.2f}' for v in results_ideal]}")

    # With noise
    noisy_omac = OMAC(
        num_inputs=3, num_outputs=3,
        mrr_config=DEFAULT_MRR_CONFIG,
        noise_config=NoiseConfig(
            shot_noise_enabled=True,
            thermal_noise_std=0.001,
            crosstalk_fraction=0.02,
        ),
    )
    results_noisy = noisy_omac.compute_mvm_analogue(DEFAULT_MATRIX, DEFAULT_VECTOR)
    print(f"   Noisy photocurrents (uA):  {[f'{v:.2f}' for v in results_noisy]}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
