"""
Photonic MLIR: Compiler Infrastructure for Silicon Photonic AI Accelerators

This package provides high-level Python interfaces for compiling PyTorch models
to photonic circuits using MLIR infrastructure.
"""

from .compiler import PhotonicCompiler, PhotonicBackend
from .optimization import OptimizationPipeline, PhotonicPasses
from .simulation import PhotonicSimulator, HardwareInterface
from .benchmarking import BenchmarkSuite, ModelBenchmark
from .research import ResearchSuite, PhotonicVsElectronicComparison
from .breakthrough_research import (
    PhotonicNeuralArchitectureSearch, QuantumEnhancedPhotonicLearning,
    create_breakthrough_research_suite, run_comparative_photonic_study
)
from .adaptive_ml import MultiModalPhotonicElectronicFusion, create_multimodal_fusion_system
from .cache import get_cache_manager, cached_compilation, cached_simulation
from .load_balancer import LoadBalancer, DistributedCompiler, create_local_cluster
from .monitoring import get_metrics_collector, get_health_checker, performance_monitor
from .quantum_enhanced_compiler import (
    QuantumEnhancedPhotonicCompiler, QuantumPhotonicFusionMode,
    create_quantum_enhanced_research_suite, run_breakthrough_quantum_study
)
from .adaptive_realtime_compiler import (
    RealTimeAdaptiveCompiler, AdaptiveMode, CompilationRequest, CompilationPriority,
    create_real_time_adaptive_compiler, start_adaptive_compilation_service
)

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
    "BenchmarkSuite",
    "ModelBenchmark",
    "ResearchSuite", 
    "PhotonicVsElectronicComparison",
    "PhotonicNeuralArchitectureSearch",
    "QuantumEnhancedPhotonicLearning", 
    "create_breakthrough_research_suite",
    "run_comparative_photonic_study",
    "MultiModalPhotonicElectronicFusion",
    "create_multimodal_fusion_system",
    "get_cache_manager",
    "cached_compilation",
    "cached_simulation",
    "LoadBalancer",
    "DistributedCompiler",
    "create_local_cluster",
    "get_metrics_collector",
    "get_health_checker",
    "performance_monitor",
    "PhotonicLayer",
    "PhotonicMLP",
    "PhotonicConv2d",
    # Breakthrough Quantum-Enhanced Capabilities
    "QuantumEnhancedPhotonicCompiler",
    "QuantumPhotonicFusionMode",
    "create_quantum_enhanced_research_suite",
    "run_breakthrough_quantum_study",
    # Next-Generation Adaptive Compilation
    "RealTimeAdaptiveCompiler",
    "AdaptiveMode",
    "CompilationRequest", 
    "CompilationPriority",
    "create_real_time_adaptive_compiler",
    "start_adaptive_compilation_service",
]