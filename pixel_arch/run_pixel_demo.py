"""Demonstration of the PIXEL architectures (OE and OO).

This script compares the standard incoherent OMAC against the 
Pixel OE (electrical accumulation) and Pixel OO (optical delay line) architectures.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.omac import OMAC
from pixel_arch.omac import Pixel_OE_OMAC, Pixel_OO_OMAC
from pixel_arch.pixel_devices import PixelMRRConfig, ThresholdStrategy


def main():
    # Define a simple vector dot product
    inputs = [15, 15, 0, 15]  # e.g., 4-bit max values
    weights = [15, 0, 15, 15] # Expect 2 active matches (index 0 and 3)

    print("=" * 80)
    print("PIXEL Architecture Simulation Demo")
    print("=" * 80)
    print(f"Inputs:  {inputs}")
    print(f"Weights: {weights}")
    print("-" * 80)

    # 1. Standard OMAC (Incoherent optical sum, no threshold per MRR)
    standard_omac = OMAC()
    std_result = standard_omac.compute_dot_product_analogue(inputs, weights, bitwidth=4)
    print(f"Standard OMAC Output (Photocurrent uA): {std_result:.2f}")

    # 2. Pixel OE OMAC (Optical multiply, electrical sum)
    # We use MEAN_POWER threshold strategy for the internal TIA decision
    oe_config = PixelMRRConfig(threshold_strategy=ThresholdStrategy.MEAN_POWER)
    oe_omac = Pixel_OE_OMAC(pixel_mrr_config=oe_config)
    
    # Notice that OE OMAC returns an electrical sum (bits added together), not photocurrent!
    oe_result = oe_omac.compute_dot_product_analogue(inputs, weights, bitwidth=4)
    print(f"Pixel OE OMAC Output (Electrical Sum):  {oe_result:.2f}")

    # 3. Pixel OO OMAC (Optical multiply, optical delay line sum)
    oo_config = PixelMRRConfig()
    oo_omac = Pixel_OO_OMAC(pixel_mrr_config=oo_config)
    oo_result = oo_omac.compute_dot_product_analogue(inputs, weights, bitwidth=4)
    print(f"Pixel OO OMAC Output (Photocurrent uA): {oo_result:.2f}")


if __name__ == "__main__":
    main()
