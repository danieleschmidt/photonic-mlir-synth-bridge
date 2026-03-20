# photonic-mlir-synth-bridge

**MLIR dialect + HLS generator for silicon photonic AI accelerators**

Generates Verilog-style hardware description code for photonic neural network
accelerators based on Mach-Zehnder Interferometer (MZI) mesh architectures.

## Overview

Silicon photonic accelerators perform matrix-vector multiplication at the speed
of light using optical interference in MZI meshes. This library provides:

- **HLS code generation** — Verilog-style pseudocode for MZI elements, meshes, and layers
- **Mesh scheduling** — Clements decomposition pipeline staging for N-port unitaries
- **WDM multiplexing** — Wavelength-division multiplexing channel assignment
- **Transformer demo** — Full attention head HLS generation as a worked example

## Installation

```bash
pip install -e .
# or
pip install numpy
```

## Quick Start

```python
from photonic_mlir_synth import (
    PhotonicHLSGen,
    MeshScheduler,
    WavelengthMultiplexer,
    TransformerHLSDemo,
)
import numpy as np

# Generate a single MZI module
gen = PhotonicHLSGen()
hls = gen.generate_mzi_module(mzi_id=0, theta=np.pi/4, phi=np.pi/3)
print(hls)

# Schedule a 4x4 MZI mesh (Clements decomposition)
scheduler = MeshScheduler(mesh_size=4)
scheduler.schedule_clements(4)
print(f"Pipeline depth: {scheduler.get_depth()} stages")
print(f"MZI count: {scheduler.get_mzi_count()}")

# Assign WDM channels to parallel data paths
wdm = WavelengthMultiplexer(num_wavelengths=4)
assignments = wdm.assign_wavelengths([0, 1, 2, 3])
print(f"Path 0 wavelength: {wdm.get_channel(0):.2f} nm")

# Full transformer attention head demo
demo = TransformerHLSDemo()
result = demo.full_demo(seq_len=8, d_model=32, d_head=8)
print(result["hls"][:500])
```

## Architecture

### PhotonicHLSGen

Generates Verilog-style module definitions for:

- **MZI elements**: `generate_mzi_module(mzi_id, theta, phi)` — single MZI with configurable phase shifts
- **MZI meshes**: `generate_mesh_module(name, size)` — N×N unitary mesh (Clements layout)
- **Photonic layers**: `generate_layer(name, in_size, out_size)` — full NN layer with optical nonlinearity

### MeshScheduler

Computes Clements decomposition pipeline stages for N-port unitary MZI meshes:

- **N-1 pipeline stages** for an N-port mesh
- **N(N-1)/2 total MZIs** (complete unitary parameterization)
- Alternating even/odd column layout for maximum parallelism

### WavelengthMultiplexer

Manages C-band WDM channel allocation (default: 1550 nm base, 0.8 nm spacing):

- Round-robin channel assignment with wavelength reuse on spatial modes
- Demultiplexing for routing analysis
- Frequency conversion utilities

### TransformerHLSDemo

Maps transformer self-attention to photonic hardware:

1. **QKV projection** — three parallel MZI meshes implementing Q, K, V linear transforms
2. **Scaled dot-product attention** — optical homodyne detection for inner products, gain-normalized softmax
3. **WDM parallelism** — parallel projection paths on different wavelengths

## Background

### MZI Transfer Matrix

A single MZI implements the 2×2 unitary:

```
U(θ, φ) = [[cos(θ/2),  j·sin(θ/2)],  ·  [[1,  0     ],
            [j·sin(θ/2), cos(θ/2) ]]      [0,  e^(jφ) ]]
```

### Clements Decomposition

Any N×N unitary can be decomposed into a rectangular mesh of N(N-1)/2 MZIs
arranged in N-1 columns. This library uses the Clements (2016) layout which
achieves optimal depth.

### WDM Parallelism

Wavelength-division multiplexing allows multiple independent matrix operations
on the same waveguide chip using different optical frequencies (channels).

## Running the Demo

```bash
cd examples
python demo.py
```

## Tests

```bash
pytest tests/ -v
```

## References

- Clements et al., "Optimal design for universal multiport interferometers," Optica 2016
- Shen et al., "Deep learning with coherent nanophotonic circuits," Nature Photonics 2017
- Hamerly et al., "Large-Scale Optical Neural Networks Based on Photoelectric Multiplication," PRX 2019
