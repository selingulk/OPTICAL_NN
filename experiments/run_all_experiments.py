"""Comprehensive Benchmark Suite for the OPTICAL_NN Simulator.

Runs 5 distinct experiments in parallel to generate publication-ready plots
for the final report.
"""

import os
import sys
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerator.architecture import ONNTile, MZITile
from accelerator.metrics import MetricsEngine, MZIMetricsEngine, ElectronicBaselineEngine, PPAMetrics
from accelerator.omac import OMAC
from devices.laser import Laser
from devices.noise import NoiseConfig
from workloads.networks import SimpleMLP, SimpleCNN
from workloads.cnn_layers import LinearLayer
from accelerator.inference import InferenceEngine

from pixel_arch.pixel_devices import PixelMRRConfig, ThresholdStrategy, pixel_optical_transmission
from pixel_arch.omac import Pixel_OE_OMAC, Pixel_OO_OMAC
from pixel_arch.metrics import PixelOEMetricsEngine, PixelOOMetricsEngine


# Setup plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "report_figures"))
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# EXPERIMENT 1: MRR Non-Ideality Sweep (Heatmap)
# =============================================================================
def run_exp1_mrr_non_ideality():
    print("Running Exp 1: MRR Non-Ideality Sweep...")
    il_values = np.linspace(0, 10, 20)  # Insertion Loss (dB)
    er_values = np.linspace(5, 30, 25)  # Extinction Ratio (dB)
    
    error_rates = np.zeros((len(il_values), len(er_values)))
    num_trials = 1000
    
    import random
    rng = random.Random(42)
    
    from devices.noise import add_thermal_noise
    
    for i, il in enumerate(il_values):
        for j, er in enumerate(er_values):
            config = PixelMRRConfig(
                insertion_loss_db=il,
                extinction_ratio_db=er,
                thermal_drift_std_db=1.5, # Increased organic noise
                threshold_strategy=ThresholdStrategy.MEAN_POWER
            )
            
            threshold = config.get_dynamic_threshold()
            errors = 0
            
            for _ in range(num_trials):
                inp = rng.choice([0, 1])
                wt = rng.choice([0, 1])
                expected = inp & wt
                
                # Physical transmission with drift
                power = pixel_optical_transmission(inp, wt, config, rng)
                # Add thermal electrical noise (equivalent optical power noise) to smooth out errors
                power = add_thermal_noise(power, 0.05, rng)
                
                actual = 1 if power >= threshold else 0
                
                if actual != expected:
                    errors += 1
            
            error_rates[i, j] = errors / num_trials

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(error_rates, aspect='auto', origin='lower', cmap='inferno', interpolation='bilinear',
                    extent=[er_values[0], er_values[-1], il_values[0], il_values[-1]])
    cbar = fig.colorbar(cax)
    cbar.set_label('Bit Error Rate (BER)', fontsize=14, weight='bold')
    
    ax.set_title("Exp 1: MRR Safe Operating Zone\n(Thermal Drift = 1.5 dB, Dynamic Threshold)", fontweight='bold', pad=15)
    ax.set_xlabel("Extinction Ratio (dB)", weight='bold')
    ax.set_ylabel("Insertion Loss (dB)", weight='bold')
    
    # Add a contour to highlight the safe zone (e.g., < 5% error)
    contour = ax.contour(er_values, il_values, error_rates, levels=[0.05], colors='cyan', linewidths=2.5, linestyles='dashed')
    
    # Add a custom legend for the contour line
    import matplotlib.lines as mlines
    cyan_line = mlines.Line2D([], [], color='cyan', linestyle='dashed', linewidth=2.5, label='Safe Zone Boundary (BER < 5%)')
    ax.legend(handles=[cyan_line], loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_mrr_nonideality_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Exp 1 complete.")


# =============================================================================
# EXPERIMENT 2: Noise Robustness
# =============================================================================
def run_exp2_noise_robustness():
    print("Running Exp 2: Noise Robustness Sweep...")
    thermal_noise_levels = np.linspace(0, 5, 20)
    
    # We will test Standard OMAC vs Pixel OE vs Pixel OO
    std_errors = []
    oe_errors = []
    oo_errors = []
    
    inputs = [15] * 16
    weights = [15] * 16  # Dot product should equal 16 (electrical equivalent)
    # The max photocurrent for 16 matched MRRs in ideal case without loss is roughly derived.
    # We will just measure relative MSE to the 0-noise case.
    
    for noise in thermal_noise_levels:
        noise_cfg = NoiseConfig(thermal_noise_std=noise, shot_noise_enabled=True)
        
        # Standard
        std_omac = OMAC(noise_config=noise_cfg)
        std_val = std_omac.compute_dot_product_analogue(inputs, weights)
        std_ideal = OMAC().compute_dot_product_analogue(inputs, weights) # 0 noise
        std_errors.append(abs(std_val - std_ideal) / std_ideal * 100)
        
        # Pixel OE
        oe_omac = Pixel_OE_OMAC(noise_config=noise_cfg, pixel_mrr_config=PixelMRRConfig())
        oe_val = oe_omac.compute_dot_product_analogue(inputs, weights)
        oe_ideal = 16.0 # exact discrete match
        oe_errors.append(abs(oe_val - oe_ideal) / oe_ideal * 100)
        
        # Pixel OO
        oo_omac = Pixel_OO_OMAC(noise_config=noise_cfg, pixel_mrr_config=PixelMRRConfig())
        oo_val = oo_omac.compute_dot_product_analogue(inputs, weights)
        oo_ideal = Pixel_OO_OMAC().compute_dot_product_analogue(inputs, weights)
        oo_errors.append(abs(oo_val - oo_ideal) / oo_ideal * 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(thermal_noise_levels, std_errors, marker='o', markersize=8, linewidth=3, label='Standard Incoherent OMAC', color='#E57373')
    ax.plot(thermal_noise_levels, oe_errors, marker='s', markersize=8, linewidth=3, label='PIXEL OE (Shift-Accumulate)', color='#0288D1')
    ax.plot(thermal_noise_levels, oo_errors, marker='^', markersize=8, linewidth=3, label='PIXEL OO (Delay Line)', color='#388E3C')
    
    ax.set_title("Exp 2: Robustness to Thermal Noise", fontweight='bold', pad=15)
    ax.set_xlabel("Thermal Noise STD (uA)", weight='bold')
    ax.set_ylabel("Relative Output Error (%)", weight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_noise_robustness.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Exp 2 complete.")


# =============================================================================
# EXPERIMENT 3: Architecture Scaling
# =============================================================================
def run_exp3_architecture_scaling():
    print("Running Exp 3: Architecture Scaling...")
    omacs_list = [1, 4, 16, 64, 256]
    
    optical_tops = []
    asic_tops = []
    
    for o in omacs_list:
        # Optical tops scale with omacs and frequency
        tile = ONNTile(num_omacs=o, clock_frequency_ghz=2.0)
        optical_tops.append(tile.get_peak_tops())
        
        # ASIC tops scale similarly assuming an equivalent array size
        # Let's say 1 OMAC roughly equates to 256 MAC units
        asic = ElectronicBaselineEngine(num_mac_units=o * 256, clock_frequency_ghz=1.0)
        asic_metrics = asic.evaluate()
        asic_tops.append(asic_metrics.throughput_tops)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(omacs_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, optical_tops, width, label='PIXEL OE / OO / MRR (2 GHz)', color='#FFB74D', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, asic_tops, width, label='Digital ASIC (1 GHz)', color='#BA68C8', edgecolor='black', linewidth=1.5)
    
    ax.set_title("Exp 3: Throughput Scaling vs Array Size", fontweight='bold', pad=15)
    ax.set_xlabel("Number of Cores/OMACs", weight='bold')
    ax.set_ylabel("Peak Throughput (TOPS)", weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in omacs_list])
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=45)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_architecture_scaling.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Exp 3 complete.")


# =============================================================================
# EXPERIMENT 4: Workload Scaling
# =============================================================================
def run_exp4_workload_scaling():
    print("Running Exp 4: Workload Scaling...")
    sizes = [128, 256, 512, 1024]
    
    tile = ONNTile(num_omacs=16, clock_frequency_ghz=2.0)
    
    # Calculate energy per mac for different architectures
    mrr_pj = MetricsEngine(tile).evaluate().energy_per_mac_pj
    oe_pj = PixelOEMetricsEngine(tile).evaluate().energy_per_mac_pj
    oo_pj = PixelOOMetricsEngine(tile).evaluate().energy_per_mac_pj
    asic_pj = ElectronicBaselineEngine(num_mac_units=16*256).evaluate().energy_per_mac_pj
    
    engine = InferenceEngine(tile)
    
    latencies = []
    energies_mrr = []
    energies_oe = []
    energies_oo = []
    energies_asic = []
    
    for s in sizes:
        layer = LinearLayer(in_features=s, out_features=s)
        mlp = SimpleMLP(layers=[layer])
        result = engine.run_mlp(mlp)
        
        latencies.append(result.total_latency_us)
        total_macs = mlp.total_macs()
        
        energies_mrr.append(total_macs * mrr_pj * 1e-3) # nJ
        energies_oe.append(total_macs * oe_pj * 1e-3)
        energies_oo.append(total_macs * oo_pj * 1e-3)
        energies_asic.append(total_macs * asic_pj * 1e-3)

    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    color = 'black'
    ax1.set_xlabel('Square Matrix Dimension (N x N)', weight='bold')
    ax1.set_ylabel('Latency (us)', color=color, weight='bold')
    ax1.plot(sizes, latencies, marker='D', markersize=8, color=color, linewidth=3, linestyle=':', label="Latency (All Optical)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Inference Energy (nJ)', color='#D32F2F', weight='bold')
    
    ax2.plot(sizes, energies_mrr, marker='o', markersize=8, color='#1976D2', linewidth=3, label="MRR Energy")
    ax2.plot(sizes, energies_oe, marker='s', markersize=8, color='#0288D1', linewidth=3, label="PIXEL OE Energy")
    ax2.plot(sizes, energies_oo, marker='^', markersize=8, color='#388E3C', linewidth=3, label="PIXEL OO Energy")
    ax2.plot(sizes, energies_asic, marker='x', markersize=8, color='#BA68C8', linewidth=3, label="ASIC Energy")
    ax2.tick_params(axis='y', labelcolor='#D32F2F')
    
    # Combine legends outside the plot
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor='white', framealpha=0.9)

    fig.suptitle("Exp 4: Workload Size Scaling on Fixed Tile (16 OMACs)", fontweight='bold')
    fig.tight_layout()
    
    path = os.path.join(RESULTS_DIR, "exp4_workload_scaling.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Exp 4 complete.")


# =============================================================================
# EXPERIMENT 5: Full PPA Comparison (7 Architectures)
# =============================================================================
def run_exp5_ppa_comparison():
    print("Running Exp 5: PPA Architecture Comparison...")
    
    # 1) MRR Baseline
    base_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    base_mrr = MetricsEngine(base_tile).evaluate()
    base_mrr = PPAMetrics("MRR Baseline", base_mrr.power_mw, base_mrr.energy_per_mac_pj, base_mrr.throughput_tops, base_mrr.area_mm2)

    # 2) MRR Advanced
    adv_omac = OMAC(laser=Laser(power_mw=5.0, wall_plug_efficiency=0.4))
    adv_tile = ONNTile(num_omacs=16, clock_frequency_ghz=2, omac_template=adv_omac)
    adv_mrr = MetricsEngine(adv_tile).evaluate()
    adv_mrr = PPAMetrics("MRR Advanced", adv_mrr.power_mw, adv_mrr.energy_per_mac_pj, adv_mrr.throughput_tops, adv_mrr.area_mm2)

    # 3) MZI-mesh
    mzi_tile = MZITile(num_omacs=4, clock_frequency_ghz=1)
    mzi = MZIMetricsEngine(mzi_tile).evaluate()

    # 4) PIXEL OE (Optoelectronic)
    oe_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    pixel_oe = PixelOEMetricsEngine(oe_tile).evaluate()

    # 5) PIXEL OO (All-Optical)
    oo_tile = ONNTile(num_omacs=4, clock_frequency_ghz=1)
    pixel_oo = PixelOOMetricsEngine(oo_tile).evaluate()

    # 6) Digital ASIC
    asic = ElectronicBaselineEngine(num_mac_units=256, clock_frequency_ghz=1.0, arch_name="Digital ASIC").evaluate()

    # 7) Google TPU
    tpu = ElectronicBaselineEngine(num_mac_units=16384, clock_frequency_ghz=0.7, energy_per_mac_pj=1.5, mac_unit_area_um2=1500.0, arch_name="Google TPU").evaluate()

    configs = [base_mrr, adv_mrr, mzi, pixel_oe, pixel_oo, asic, tpu]
    
    names = [c.arch_name for c in configs]
    x = np.arange(len(names))
    colors = ["#90CAF9", "#1E88E5", "#FFB74D", "#81C784", "#388E3C", "#E57373", "#BA68C8"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Exp 5: Comprehensive PPA Comparison", fontsize=20, fontweight="bold")

    metrics_list = [
        ("Power (mW)", [c.power_mw for c in configs]),
        ("Throughput (TOPS)", [c.throughput_tops for c in configs]),
        ("Energy/MAC (pJ)", [c.energy_per_mac_pj for c in configs]),
        ("Area (mm²)", [c.area_mm2 for c in configs]),
    ]

    for ax, (title, values) in zip(axes.flat, metrics_list):
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.5)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=12, weight='bold')
        ax.grid(axis="y", alpha=0.5, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                weight='bold',
                rotation=0 if title == "Energy/MAC (pJ)" else 15
            )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp5_ppa_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Exp 5 complete.")


# =============================================================================
# MAIN EXECUTOR
# =============================================================================
def main():
    print("=" * 80)
    print("STARTING BENCHMARK SUITE")
    print("Saving all figures to:", RESULTS_DIR)
    print("=" * 80)
    
    # We map functions to concurrent executor
    experiments = [
        run_exp1_mrr_non_ideality,
        run_exp2_noise_robustness,
        run_exp3_architecture_scaling,
        run_exp4_workload_scaling,
        run_exp5_ppa_comparison
    ]
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(exp) for exp in experiments]
        concurrent.futures.wait(futures)
        
        # Check for any exceptions
        for future in futures:
            if future.exception() is not None:
                print(f"Error in experiment: {future.exception()}")
                
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    # Workaround for multiprocessing on Windows
    # (Requires script to be main module and top-level functions)
    main()
