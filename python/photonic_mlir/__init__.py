"""
Photonic MLIR: Compiler Infrastructure for Silicon Photonic AI Accelerators

This package provides high-level Python interfaces for compiling PyTorch models
to photonic circuits using MLIR infrastructure.
"""

from .compiler import PhotonicCompiler, PhotonicBackend
from .optimization import OptimizationPipeline, PhotonicPasses
from .simulation import PhotonicSimulator, HardwareInterface

try:
    from .pytorch_frontend import PhotonicLayer, PhotonicMLP, PhotonicConv2d
except ImportError:
    # PyTorch not available, provide mocks
    PhotonicLayer = None
    PhotonicMLP = None
    PhotonicConv2d = None

__version__ = "0.1.0"
__author__ = "Photonic MLIR Team"

__all__ = [
    "PhotonicCompiler",
    "PhotonicBackend", 
    "OptimizationPipeline",
    "PhotonicPasses",
    "PhotonicSimulator",
    "HardwareInterface",
    "PhotonicLayer",
    "PhotonicMLP",
    "PhotonicConv2d",
]