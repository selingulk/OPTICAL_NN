"""Comprehensive test suite for the optical NN simulator.

Run with:
    python tests.py

The file intentionally avoids external dependencies so that the repository can
be cloned and tested immediately.
"""

from accelerator.arithmetic import bit_serial_multiply, build_partial_products
from accelerator.architecture import ONNTile, MZITile
from accelerator.core import dot_product_bit_serial, mvm_bit_serial, mvm_reference
from accelerator.inference import InferenceEngine
from accelerator.metrics import MetricsEngine, MZIMetricsEngine, ElectronicBaselineEngine
from accelerator.omac import OMAC, MZI_OMAC
from config import BITWIDTH, DEFAULT_MATRIX, DEFAULT_MRR_CONFIG, DEFAULT_VECTOR
from devices.laser import Laser
from devices.mrr import mrr_multiply_bit, optical_transmission, MRRConfig
from devices.mzi import MZI, MZIMeshConfig
from devices.noise import NoiseConfig, add_shot_noise, add_thermal_noise, apply_crosstalk
from devices.photodetector import Photodetector
from devices.waveguide import Waveguide
from encoding import from_bits, to_bits
from workloads.networks import SimpleMLP, SimpleCNN
from pixel_arch.pixel_devices import PixelMRRConfig, ThresholdStrategy, pixel_optical_transmission
from pixel_arch.omac import Pixel_OE_OMAC, Pixel_OO_OMAC
from pixel_arch.metrics import PixelOEMetricsEngine, PixelOOMetricsEngine

import random


def assert_raises(expected_error: type[Exception], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except expected_error:
        return
    raise AssertionError(f"Expected {expected_error.__name__} was not raised.")


# ============================================================================
# Encoding Tests
# ============================================================================

def test_encoding_round_trip() -> None:
    for value in range(16):
        bits = to_bits(value, BITWIDTH)
        assert from_bits(bits) == value


def test_encoding_validation() -> None:
    assert_raises(ValueError, to_bits, -1, BITWIDTH)
    assert_raises(ValueError, to_bits, 16, BITWIDTH)
    assert_raises(ValueError, from_bits, [1, 0, 2])


# ============================================================================
# Device Tests
# ============================================================================

def test_mrr_bit_behavior() -> None:
    assert mrr_multiply_bit(0, 0) == 0
    assert mrr_multiply_bit(0, 1) == 0
    assert mrr_multiply_bit(1, 0) == 0
    assert mrr_multiply_bit(1, 1) == 1
    assert optical_transmission(1, 1, DEFAULT_MRR_CONFIG) > optical_transmission(1, 0, DEFAULT_MRR_CONFIG)


def test_mzi_transmission() -> None:
    import math
    mzi = MZI()
    # At phase_shift=0, should be near maximum transmission
    t_max = mzi.transmission(0.0)
    # At phase_shift=pi, should be near minimum
    t_min = mzi.transmission(math.pi)
    assert t_max > t_min, f"MZI max {t_max} should be > min {t_min}"
    assert t_max > 0.5, f"MZI max transmission should be significant, got {t_max}"


def test_mzi_mesh_config() -> None:
    mesh = MZIMeshConfig()
    assert mesh.num_mzis(4) == 6  # 4*(4-1)/2 = 6
    assert mesh.num_mzis(16) == 120  # 16*15/2 = 120
    power = mesh.total_phase_shifter_power_mw(4)
    assert power > 0


def test_laser() -> None:
    laser = Laser(power_mw=10.0, wall_plug_efficiency=0.2)
    assert laser.electrical_power_mw == 50.0  # 10 / 0.2


def test_photodetector() -> None:
    pd = Photodetector()
    current = pd.get_photocurrent_ua(1.0)
    assert current > 0


def test_waveguide() -> None:
    wg = Waveguide(loss_db_per_cm=2.0)
    t = wg.get_transmission(1.0)
    assert 0 < t < 1, f"Waveguide transmission should be in (0,1), got {t}"
    # Zero length = no loss
    assert wg.get_transmission(0.0) == 1.0


def test_noise_shot() -> None:
    rng = random.Random(42)
    # Shot noise on a positive signal should still be positive (statistically)
    vals = [add_shot_noise(1.0, rng) for _ in range(100)]
    assert all(v >= 0 for v in vals)
    # Mean should be close to 1.0
    mean = sum(vals) / len(vals)
    assert 0.9 < mean < 1.1, f"Shot noise mean {mean} too far from 1.0"


def test_noise_thermal() -> None:
    rng = random.Random(42)
    noisy = add_thermal_noise(5.0, 0.01, rng)
    assert abs(noisy - 5.0) < 0.1, "Thermal noise should be small"


def test_noise_crosstalk() -> None:
    signals = [0.0, 1.0, 0.0]
    result = apply_crosstalk(signals, 0.1)
    # Middle channel should lose some power, neighbours should gain
    assert result[0] > 0, "Crosstalk should leak into neighbour"
    assert result[2] > 0, "Crosstalk should leak into neighbour"
    assert result[1] < 1.0, "Middle channel should lose power"


# ============================================================================
# Arithmetic Tests
# ============================================================================

def test_bit_serial_multiply_matches_python() -> None:
    for a in range(16):
        for b in range(16):
            assert bit_serial_multiply(a, b, BITWIDTH, DEFAULT_MRR_CONFIG) == a * b


def test_partial_products_for_3_times_2() -> None:
    partials = build_partial_products(3, 2, BITWIDTH, DEFAULT_MRR_CONFIG)
    active_terms = [(p.a_index, p.b_index, p.shift) for p in partials if p.value == 1]
    assert active_terms == [(0, 1, 1), (1, 1, 2)]
    assert bit_serial_multiply(3, 2, BITWIDTH, DEFAULT_MRR_CONFIG) == 6


# ============================================================================
# Core MVM Tests
# ============================================================================

def test_dot_product_and_mvm() -> None:
    assert dot_product_bit_serial([1, 2, 3], [2, 1, 3], BITWIDTH, DEFAULT_MRR_CONFIG) == 13
    assert mvm_bit_serial(DEFAULT_MATRIX, DEFAULT_VECTOR, BITWIDTH, DEFAULT_MRR_CONFIG) == mvm_reference(
        DEFAULT_MATRIX,
        DEFAULT_VECTOR,
    )


def test_dimension_validation() -> None:
    assert_raises(ValueError, dot_product_bit_serial, [1, 2], [1], BITWIDTH)
    assert_raises(ValueError, mvm_bit_serial, [[1, 2], [3]], [1, 2], BITWIDTH)
    assert_raises(ValueError, mvm_bit_serial, [[1, 2]], [1, 2, 3], BITWIDTH)


# ============================================================================
# OMAC Signal-Path Tests
# ============================================================================

def test_omac_signal_path() -> None:
    omac = OMAC(num_inputs=3, num_outputs=3, mrr_config=DEFAULT_MRR_CONFIG)
    # Simple signal path test: input=1, weight=1 should produce photocurrent
    current = omac.compute_dot_product_analogue([1, 1, 1], [1, 1, 1])
    assert current > 0, f"OMAC dot product should produce positive photocurrent, got {current}"


def test_omac_mvm_analogue() -> None:
    omac = OMAC(num_inputs=3, num_outputs=3, mrr_config=DEFAULT_MRR_CONFIG)
    results = omac.compute_mvm_analogue(DEFAULT_MATRIX, DEFAULT_VECTOR)
    assert len(results) == 3
    # All values should be non-negative
    assert all(r >= 0 for r in results), f"All outputs should be non-negative: {results}"


def test_omac_with_noise() -> None:
    noisy_omac = OMAC(
        num_inputs=3, num_outputs=3,
        mrr_config=DEFAULT_MRR_CONFIG,
        noise_config=NoiseConfig(shot_noise_enabled=True, thermal_noise_std=0.001, crosstalk_fraction=0.01),
    )
    results = noisy_omac.compute_mvm_analogue(DEFAULT_MATRIX, DEFAULT_VECTOR)
    assert len(results) == 3


# ============================================================================
# Architecture & Latency Tests
# ============================================================================

def test_onn_tile_tops() -> None:
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    tops = tile.get_peak_tops()
    assert tops > 0


def test_onn_tile_latency() -> None:
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    cycles = tile.cycles_for_mvm(16, 16, bitwidth=4)
    assert cycles > 0
    latency = tile.latency_us(16, 16, bitwidth=4)
    assert latency > 0
    # Larger matrix should take more cycles
    cycles_big = tile.cycles_for_mvm(64, 64, bitwidth=4)
    assert cycles_big >= cycles


def test_mzi_tile() -> None:
    tile = MZITile(num_omacs=4, clock_frequency_ghz=1)
    tops = tile.get_peak_tops()
    assert tops > 0
    cycles = tile.cycles_for_mvm(16, 16, bitwidth=4)
    assert cycles > 0


# ============================================================================
# Metrics Tests
# ============================================================================

def test_mrr_metrics() -> None:
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    m = MetricsEngine(tile).evaluate()
    assert m.power_mw > 0
    assert m.throughput_tops > 0
    assert m.area_mm2 > 0
    assert m.energy_per_mac_pj > 0
    assert m.arch_name == "MRR-Based ONN"


def test_mzi_metrics() -> None:
    tile = MZITile(num_omacs=4, clock_frequency_ghz=1)
    m = MZIMetricsEngine(tile).evaluate()
    assert m.power_mw > 0
    assert m.throughput_tops > 0
    assert m.area_mm2 > 0
    assert m.arch_name == "MZI-Mesh ONN"


def test_electronic_baseline() -> None:
    m = ElectronicBaselineEngine(num_mac_units=256).evaluate()
    assert m.power_mw > 0
    assert m.throughput_tops > 0
    assert m.arch_name == "Digital ASIC Baseline (7nm)"


# ============================================================================
# Inference Tests
# ============================================================================

def test_inference_mlp() -> None:
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=2)
    engine = InferenceEngine(tile, bitwidth=4)
    mlp = SimpleMLP.get_mnist_mlp()
    result = engine.run_mlp(mlp)
    assert result.total_macs > 0
    assert result.total_cycles > 0
    assert result.total_latency_us > 0
    assert result.total_energy_nj > 0
    assert len(result.layer_results) == 3  # 3 layers in MNIST MLP


def test_inference_cnn() -> None:
    tile = ONNTile(num_omacs=4, clock_frequency_ghz=2)
    engine = InferenceEngine(tile, bitwidth=4)
    cnn = SimpleCNN.get_simple_cnn()
    result = engine.run_cnn(cnn)
    assert result.total_macs > 0
    assert len(result.layer_results) == 4  # 2 conv + 2 linear


# ============================================================================
# Workload Tests
# ============================================================================

def test_workloads() -> None:
    mlp = SimpleMLP.get_mnist_mlp()
    assert mlp.total_macs() > 0

    cnn = SimpleCNN.get_simple_cnn()
    assert cnn.total_macs() > 0
    assert cnn.total_macs() > mlp.total_macs()  # CNN should have more MACs


# ============================================================================
# PIXEL Architecture Tests
# ============================================================================

def test_pixel_mrr_config() -> None:
    from pixel_arch.pixel_devices import PixelMRRConfig, ThresholdStrategy
    
    # 1. FIXED strategy
    cfg_fixed = PixelMRRConfig(threshold_strategy=ThresholdStrategy.FIXED, fixed_threshold=0.3)
    assert cfg_fixed.get_dynamic_threshold() == 0.3
    
    # 2. MEAN_POWER strategy
    cfg_mean = PixelMRRConfig(insertion_loss_db=0.0, extinction_ratio_db=20.0, threshold_strategy=ThresholdStrategy.MEAN_POWER)
    expected_on = 1.0
    expected_off = 0.01
    assert abs(cfg_mean.get_dynamic_threshold() - (expected_on + expected_off) / 2.0) < 1e-7
    
    # 3. SCALED_ON strategy
    cfg_scaled = PixelMRRConfig(insertion_loss_db=0.0, threshold_strategy=ThresholdStrategy.SCALED_ON, threshold_scale_factor=0.6)
    assert abs(cfg_scaled.get_dynamic_threshold() - 0.6) < 1e-7


def test_pixel_oe_omac() -> None:
    from pixel_arch.omac import Pixel_OE_OMAC
    from pixel_arch.pixel_devices import PixelMRRConfig
    from devices.waveguide import Waveguide
    
    # Test valid binary inputs / weights
    oe_omac = Pixel_OE_OMAC(pixel_mrr_config=PixelMRRConfig(thermal_drift_std_db=0.0))
    res = oe_omac.compute_dot_product_analogue([15, 0], [15, 15], bitwidth=4)
    assert res == 1.0
    
    # Test non-binary inputs/weights validation rejection
    assert_raises(ValueError, oe_omac.compute_dot_product_analogue, [7], [15], bitwidth=4)
    assert_raises(ValueError, oe_omac.compute_dot_product_analogue, [15], [7], bitwidth=4)
    
    # Test Loss vs No-Loss Simulation support
    oe_ideal = Pixel_OE_OMAC(
        pixel_mrr_config=PixelMRRConfig(insertion_loss_db=0.0, thermal_drift_std_db=0.0),
        wg=Waveguide(loss_db_per_cm=0.0)
    )
    oe_lossy = Pixel_OE_OMAC(
        pixel_mrr_config=PixelMRRConfig(insertion_loss_db=2.0, thermal_drift_std_db=0.5),
        wg=Waveguide(loss_db_per_cm=2.0)
    )
    
    res_ideal = oe_ideal.compute_dot_product_analogue([15], [15], bitwidth=4)
    res_lossy = oe_lossy.compute_dot_product_analogue([15], [15], bitwidth=4)
    assert res_ideal == 1.0
    assert res_lossy in (0.0, 1.0)


def test_pixel_oo_omac() -> None:
    from pixel_arch.omac import Pixel_OO_OMAC
    from pixel_arch.pixel_devices import PixelMRRConfig
    from devices.waveguide import Waveguide
    
    oo_omac = Pixel_OO_OMAC(pixel_mrr_config=PixelMRRConfig(thermal_drift_std_db=0.0))
    res = oo_omac.compute_dot_product_analogue([15, 0], [15, 15], bitwidth=4)
    assert res > 0.0
    
    # Test validation rejection
    assert_raises(ValueError, oo_omac.compute_dot_product_analogue, [7], [15], bitwidth=4)
    
    # Test Loss vs No-Loss Simulation support
    oo_ideal = Pixel_OO_OMAC(
        pixel_mrr_config=PixelMRRConfig(insertion_loss_db=0.0, thermal_drift_std_db=0.0),
        wg=Waveguide(loss_db_per_cm=0.0)
    )
    oo_lossy = Pixel_OO_OMAC(
        pixel_mrr_config=PixelMRRConfig(insertion_loss_db=2.0, thermal_drift_std_db=0.5),
        wg=Waveguide(loss_db_per_cm=2.0)
    )
    res_ideal = oo_ideal.compute_dot_product_analogue([15], [15], bitwidth=4)
    res_lossy = oo_lossy.compute_dot_product_analogue([15], [15], bitwidth=4)
    assert res_ideal > 0.0
    assert res_lossy >= 0.0


def test_pixel_metrics() -> None:
    from pixel_arch.omac import Pixel_OE_OMAC, Pixel_OO_OMAC
    
    oe_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1, omac_template=Pixel_OE_OMAC())
    oo_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1, omac_template=Pixel_OO_OMAC())
    
    # Check InferenceEngine correctly selects the appropriate metrics engine
    engine_oe = InferenceEngine(oe_tile, bitwidth=4)
    engine_oo = InferenceEngine(oo_tile, bitwidth=4)
    
    assert engine_oe._metrics.arch_name == "PIXEL OE (Optoelectronic)"
    assert engine_oo._metrics.arch_name == "PIXEL OO (All-Optical)"
    
    assert engine_oe._metrics.power_mw > 0
    assert engine_oe._metrics.area_mm2 > 0
    assert engine_oo._metrics.power_mw > 0
    assert engine_oo._metrics.area_mm2 > 0


# ============================================================================
# Runner
# ============================================================================

def run_all_tests() -> None:
    tests = [
        # Encoding
        test_encoding_round_trip,
        test_encoding_validation,
        # Devices
        test_mrr_bit_behavior,
        test_mzi_transmission,
        test_mzi_mesh_config,
        test_laser,
        test_photodetector,
        test_waveguide,
        test_noise_shot,
        test_noise_thermal,
        test_noise_crosstalk,
        # Arithmetic
        test_bit_serial_multiply_matches_python,
        test_partial_products_for_3_times_2,
        # Core MVM
        test_dot_product_and_mvm,
        test_dimension_validation,
        # OMAC Signal Path
        test_omac_signal_path,
        test_omac_mvm_analogue,
        test_omac_with_noise,
        # Architecture & Latency
        test_onn_tile_tops,
        test_onn_tile_latency,
        test_mzi_tile,
        # Metrics
        test_mrr_metrics,
        test_mzi_metrics,
        test_electronic_baseline,
        # Inference
        test_inference_mlp,
        test_inference_cnn,
        # Workloads
        test_workloads,
        # PIXEL sub-package
        test_pixel_mrr_config,
        test_pixel_oe_omac,
        test_pixel_oo_omac,
        test_pixel_metrics,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test.__name__}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed > 0:
        print(f"{failed} test(s) FAILED.")
        raise SystemExit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()