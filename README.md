# Optical Neural Network Accelerator Simulator

A cycle-level simulator for photonic neural network accelerators, developed for the
**Introduction to AI Processor Design** course at Eskisehir Technical University.

The simulator models the full signal path of an optical matrix-vector multiplication (MVM)
engine---from laser source through Mach-Zehnder interferometers (MZI) or microring resonators
(MRR), waveguides, and photodetectors---and evaluates Power, Performance, and Area (PPA)
metrics for different architectural configurations.

---

## Architecture Overview

```
                    +--------------------------------------------------+
                    |              ONN Accelerator Tile                 |
                    |                                                  |
                    |  +----------+  +----------+  +----------+       |
                    |  |  OMAC 0  |  |  OMAC 1  |  |  OMAC N  |       |
                    |  |          |  |          |  |          |       |
  Input Vector ---->|  | Laser    |  | Laser    |  | Laser    |       |
  (from DAC)        |  | MRR/MZI  |  | MRR/MZI  |  | MRR/MZI  |       |
                    |  | Waveguide|  | Waveguide|  | Waveguide|       |
                    |  | Photo-   |  | Photo-   |  | Photo-   |       |
                    |  | detector |  | detector |  | detector |       |
                    |  +----+-----+  +----+-----+  +----+-----+       |
                    |       |             |             |              |
                    |       v             v             v              |
                    |  +----+-------------+-------------+----+        |
                    |  |   TIA  ->  ADC  ->  Shift-Accumulate |        |
                    |  +------------------------------------------+   |
                    +--------------------------------------------------+
                                          |
                                          v
                                   Output Vector
```

Four compute architectures are supported:

| Architecture | Weight Encoding | Compute Style | Key Component |
|-------------|----------------|---------------|---------------|
| **MRR-based** | Binary weight bits on microring resonators | Bit-serial multiply-accumulate | MRR array |
| **MZI-mesh** | Phase-encoded weights in MZI mesh | Analogue unitary transform | Clements/Reck mesh |
| **PIXEL OE** | Binary weight bits on MRRs | Optoelectronic: Optical multiply, electrical accumulate | MRR + CLA Adders |
| **PIXEL OO** | Binary weight bits on MRRs | All-Optical: Optical multiply, optical delay line accumulate| MRR + MZI Delay Lines |

---

## Project Structure

```
OPTICAL_NN/
|-- devices/                # Physical device models
|   |-- laser.py            # Laser source (power, wall-plug efficiency)
|   |-- mrr.py              # Microring resonator (transmission, bit-multiply)
|   |-- mzi.py              # Mach-Zehnder interferometer + mesh config
|   |-- photodetector.py    # Photodetector (responsivity, dark current)
|   |-- waveguide.py        # Waveguide propagation loss
|   |-- electrical.py       # DAC, ADC, TIA, SERDES models
|   |-- noise.py            # Shot noise, thermal noise, crosstalk
|
|-- accelerator/            # System-level architecture
|   |-- omac.py             # Optical MAC unit (MRR-based + MZI-based)
|   |-- architecture.py     # Tile configs (ONNTile, MZITile) + latency model
|   |-- arithmetic.py       # Bit-serial partial product generation
|   |-- core.py             # Matrix-vector multiplication engine
|   |-- metrics.py          # PPA evaluation (MRR, MZI, electronic baseline)
|   |-- inference.py        # End-to-end network inference mapper
|
|-- pixel_arch/             # PIXEL architecture implementations
|   |-- omac.py             # Optoelectronic (OE) and All-Optical (OO) OMAC variants
|   |-- pixel_devices.py    # MRR thermal drift and dynamic TIA threshold models
|   |-- metrics.py          # PPA evaluation for OE and OO designs
|   |-- run_pixel_demo.py   # PIXEL architecture demonstration script
|
|-- workloads/              # Neural network workload definitions
|   |-- cnn_layers.py       # Conv2D and Linear layer models
|   |-- networks.py         # SimpleMLP (MNIST) and SimpleCNN definitions
|
|-- experiments/            # Runnable experiment scripts (produce plots)
|   |-- run_all_experiments.py     # Run all available experiments sequentially
|   |-- run_mvm_demo.py            # MVM correctness + signal-path demo
|   |-- run_architecture_comparison.py  # MRR vs MZI vs electronic PPA
|   |-- run_layer_sweep.py         # Energy/latency scaling with layer size
|   |-- run_nonideal_sweep.py      # MRR non-ideality error analysis
|   |-- run_inference.py           # End-to-end network inference
|
|-- results/                # Generated plots (PNG)
|   |-- report_figures/     # Pre-generated figures for reports
|-- encoding.py             # Bit encoding/decoding utilities
|-- config.py               # Default simulation parameters
|-- main.py                 # Quick demo entry point
|-- tests.py                # 27-test comprehensive test suite
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- matplotlib (for experiment plots)
- numpy

```bash
pip install matplotlib numpy
```

### Run the Quick Demo

```bash
python main.py
```

### Run All Tests (27 tests)

```bash
python tests.py
```

### Run Experiments

```bash
python experiments/run_all_experiments.py         # Run all available experiments
python experiments/run_mvm_demo.py                # MVM + signal path demo
python experiments/run_architecture_comparison.py # Architecture PPA comparison
python experiments/run_layer_sweep.py             # Layer size scaling
python experiments/run_nonideal_sweep.py          # Non-ideal MRR analysis
python experiments/run_inference.py               # End-to-end network inference
python pixel_arch/run_pixel_demo.py               # PIXEL OE/OO architecture demo
```

All experiments save plots to the `results/` directory.

---

## Key Features

### 1. Device-Level Physical Models
- **Laser**: Configurable optical power and wall-plug efficiency
- **MRR**: Insertion loss, extinction ratio, decision threshold; bit-level AND operation
- **MZI**: Phase-dependent transmission with insertion loss and extinction ratio
- **Photodetector**: Responsivity, dark current, bandwidth
- **Waveguide**: Propagation loss (dB/cm)
- **Electrical**: DAC, ADC, TIA, SERDES energy models
- **PIXEL Devices**: MRR thermal drift modeling (Gaussian perturbations to IL/ER) and dynamic TIA decision thresholds (Fixed, Mean Power, Scaled On)

### 2. Noise and Non-Ideal Effects
- Shot noise (Poisson-approximated Gaussian)
- Thermal noise (additive Gaussian)
- Nearest-neighbour crosstalk coupling

### 3. Bit-Serial Compute Pipeline
- Unsigned integer to fixed-width bit encoding (LSB-first)
- Partial product generation via MRR optical AND
- Shift-and-accumulate for full multiplication
- Dot product and matrix-vector multiplication

### 4. Full Signal-Path Simulation (OMAC)
- Laser -> DAC -> MRR weight bank -> waveguide -> photodetector -> TIA -> ADC
- Analogue MVM with continuous-valued input/weight normalization
- Noise injection at each stage

### 5. Architecture Comparison
- **MRR-based ONN**: Bit-serial, MRR weight bank, compact area
- **MZI-mesh ONN**: Analogue, Clements/Reck decomposition, higher power
- **PIXEL OE (Optoelectronic)**: MRR multiply, electrical accumulate via CLA, individual comparators per MRR
- **PIXEL OO (All-Optical)**: MRR multiply, passive optical accumulate via MZI delay lines, single ADC per output
- **Digital ASIC baseline**: 7nm electronic MAC array for reference

### 6. PPA Metrics Engine
- Power: Laser + DAC + ADC + TIA + phase shifter breakdown
- Performance: Peak TOPS, cycle-accurate latency, tiling for large matrices
- Area: Component-level area estimation (MRR, MZI, laser, PD, DAC, ADC)
- Energy efficiency: pJ/MAC

### 7. End-to-End Inference Mapping
- Maps MLP and CNN layers onto the accelerator tile
- Per-layer and total cycle count, latency, energy
- Tiling for matrices larger than the OMAC array

---

## Sample Results

### Architecture Comparison
| Metric | MRR Baseline | MRR Advanced | MZI-Mesh | Digital ASIC | TPU Baseline |
|--------|-------------|-------------|----------|-------------|--------------|
| Power (mW) | 3,309 | 4,070 | 8,096 | 128 | 17,203 |
| Throughput (TOPS) | 2.05 | 16.38 | 2.05 | 0.51 | 22.94 |
| Energy/MAC (pJ) | 3.23 | 0.50 | 7.91 | 0.50 | 1.50 |
| Area (mm^2) | 0.26 | 1.05 | 4.96 | 0.51 | 24.58 |

### Inference: SimpleMLP (MNIST) on MRR Tile
| Layer | MACs | Cycles | Latency (us) | Energy (nJ) |
|-------|------|--------|-------------|-------------|
| Linear(784->128) | 100,352 | 6,272 | 3.14 | 167.46 |
| Linear(128->64) | 8,192 | 512 | 0.26 | 13.67 |
| Linear(64->10) | 640 | 64 | 0.03 | 1.07 |
| **TOTAL** | **109,184** | **6,848** | **3.42** | **182.20** |

---

## References

1. Shen, Y. et al., "Deep learning with coherent nanophotonic circuits," *Nature Photonics*, 2017.
2. Tait, A.N. et al., "Neuromorphic photonic networks using silicon photonic weight banks," *Scientific Reports*, 2017.
3. Clements, W.R. et al., "Optimal design for universal multiport interferometers," *Optica*, 2016.
4. Reck, M. et al., "Experimental realization of any discrete unitary operator," *Physical Review Letters*, 1994.

---

## License

This project was developed as coursework for **Introduction to AI Processor Design**
at Eskisehir Technical University, Spring 2026.