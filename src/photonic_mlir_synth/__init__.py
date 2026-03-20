"""
photonic_mlir_synth - MLIR dialect + HLS generator for silicon photonic AI accelerators.

Provides tools for generating Verilog-style hardware description code for
photonic neural network accelerators based on MZI mesh architectures.
"""

from .hlsgen import PhotonicHLSGen
from .scheduler import MeshScheduler
from .wavelength import WavelengthMultiplexer
from .transformer_demo import TransformerHLSDemo

__version__ = "0.1.0"
__all__ = [
    "PhotonicHLSGen",
    "MeshScheduler",
    "WavelengthMultiplexer",
    "TransformerHLSDemo",
]
