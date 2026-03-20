#!/usr/bin/env python3
"""
Photonic MLIR Synthesis Bridge - Demo

Demonstrates HLS generation for a transformer attention head
on silicon photonic hardware.
"""

import sys
import os

# Allow running from repo root or examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photonic_mlir_synth import TransformerHLSDemo, MeshScheduler, WavelengthMultiplexer


def main():
    print("=" * 70)
    print("Photonic MLIR Synthesis Bridge - Demo")
    print("Silicon Photonic AI Accelerator HLS Generator")
    print("=" * 70)

    # Run full demo
    demo = TransformerHLSDemo()
    result = demo.full_demo(seq_len=8, d_model=32, d_head=8)

    # Show HLS output (truncated for display)
    hls = result["hls"]
    lines = hls.split("\n")
    print("\n--- Generated Verilog-style HLS (first 60 lines) ---")
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"... ({len(lines) - 60} more lines)")

    # WDM channel assignments
    print("\n--- WDM Channel Assignments ---")
    wdm_assign = result["wdm_assignments"]
    for path_id, wavelength_nm in sorted(wdm_assign.items()):
        print(f"  Path {path_id:2d}: λ = {wavelength_nm:.2f} nm")

    # Demultiplex to show wavelength sharing
    wdm = WavelengthMultiplexer(num_wavelengths=4)
    wdm.assign_wavelengths(list(wdm_assign.keys()))
    demux = wdm.demultiplex(wdm_assign)
    print("\n--- WDM Demultiplexed Channels ---")
    for wavelength, paths in sorted(demux.items()):
        print(f"  λ = {wavelength:.2f} nm → paths: {paths}")

    # Pipeline schedule
    print(f"\n--- Pipeline Schedule (d_model=32 MZI mesh) ---")
    print(f"  Pipeline depth: {result['pipeline_depth']} stages")
    print(f"  Total MZI count: {result['mzi_count']}")
    stages = result["pipeline_stages"]
    print(f"  Stages:")
    for i, stage in enumerate(stages[:6]):
        print(f"    Stage {i}: {len(stage)} parallel MZIs → {stage[:3]}{'...' if len(stage)>3 else ''}")
    if len(stages) > 6:
        print(f"    ... ({len(stages) - 6} more stages)")

    # Standalone MZI module
    print("\n--- Single MZI Module Example ---")
    from photonic_mlir_synth import PhotonicHLSGen
    import numpy as np
    gen = PhotonicHLSGen()
    mzi_hls = gen.generate_mzi_module(mzi_id=0, theta=np.pi/4, phi=np.pi/3)
    print(mzi_hls)

    print("\n--- Critical Path ---")
    scheduler = MeshScheduler(mesh_size=8)
    scheduler.schedule_clements(8)
    cpath = scheduler.get_critical_path()
    print(f"  Critical path ({len(cpath)} MZIs): {cpath}")
    print(f"  Pipeline depth: {scheduler.get_depth()} stages")
    print(f"  Parallelism per stage: {scheduler.get_parallelism()}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
