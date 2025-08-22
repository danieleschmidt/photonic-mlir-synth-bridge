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
from .breakthrough_evolution import (
    BreakthroughEvolutionEngine, EvolutionStrategy, run_breakthrough_evolution,
    create_autonomous_evolution_system
)
from .quantum_photonic_fusion import (
    QuantumPhotonicFusionEngine, QuantumPhotonicMode, PhotonicQuantumGate,
    create_quantum_photonic_fusion_system, run_quantum_photonic_breakthrough_experiment
)
from .neural_photonic_synthesis import (
    NeuralPhotonicSynthesisEngine, SynthesisMode, PhotonicCircuitDesign,
    create_neural_photonic_synthesis_system, run_autonomous_photonic_synthesis
)
from .autonomous_scaling_optimizer import (
    AutonomousScalingOptimizer, ScalingStrategy, ResourceMetrics,
    create_autonomous_scaling_system, run_scaling_optimization_demo
)
from .neuromorphic_photonic import (
    NeuromorphicPhotonicProcessor, SynapticPlasticityType, PhotonicNeuronType,
    create_neuromorphic_photonic_system, run_neuromorphic_demo
)
from .self_evolving_nas import (
    SelfEvolvingPhotonicNAS, PhotonicPrimitive, SearchStrategy, OptimizationObjective,
    create_self_evolving_nas_system, run_nas_demo
)
from .holographic_computing import (
    HolographicProcessor, HologramType, PhotorefractiveType, ProcessingMode,
    create_holographic_computing_system, run_holographic_demo
)
from .continuous_variable_quantum import (
    CVQuantumCircuit, CVQuantumNeuralNetwork, CVQuantumOptimizer, CVQuantumState,
    create_cv_quantum_system, run_cv_quantum_demo
)

# Generation 4+ Breakthrough Enhancements
from .breakthrough_quantum_coherence import (
    CoherentMatrixMultiplier, QuantumCoherenceResearchSuite, CoherenceMode,
    create_breakthrough_coherence_system, run_quantum_coherence_demo
)
from .breakthrough_photonic_nas import (
    SelfEvolvingPhotonicNAS, PhotonicLayerType, EvolutionStrategy, PhotonicArchitectureGenome,
    create_self_evolving_photonic_nas, run_breakthrough_nas_experiment
)
from .breakthrough_holographic_fusion import (
    HolographicProcessor, HolographicFusionEngine, HologramType, PhotorefractiveType,
    create_holographic_computing_system, run_holographic_fusion_demo
)
from .breakthrough_cv_quantum import (
    CVQuantumState, CVQuantumCircuit, CVQuantumNeuralNetwork, CVQuantumOptimizer,
    CVQuantumGate, CVQuantumMode, create_cv_quantum_system, run_cv_quantum_breakthrough_demo
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
    # Breakthrough Evolution & Autonomous Systems
    "BreakthroughEvolutionEngine",
    "EvolutionStrategy",
    "run_breakthrough_evolution",
    "create_autonomous_evolution_system",
    "QuantumPhotonicFusionEngine",
    "QuantumPhotonicMode",
    "PhotonicQuantumGate",
    "create_quantum_photonic_fusion_system",
    "run_quantum_photonic_breakthrough_experiment",
    "NeuralPhotonicSynthesisEngine",
    "SynthesisMode",
    "PhotonicCircuitDesign",
    "create_neural_photonic_synthesis_system",
    "run_autonomous_photonic_synthesis",
    "AutonomousScalingOptimizer",
    "ScalingStrategy",
    "ResourceMetrics",
    "create_autonomous_scaling_system",
    "run_scaling_optimization_demo",
    # Generation 4+ Breakthrough Capabilities
    "NeuromorphicPhotonicProcessor",
    "SynapticPlasticityType",
    "PhotonicNeuronType", 
    "create_neuromorphic_photonic_system",
    "run_neuromorphic_demo",
    "SelfEvolvingPhotonicNAS",
    "PhotonicPrimitive",
    "SearchStrategy",
    "OptimizationObjective",
    "create_self_evolving_nas_system",
    "run_nas_demo",
    "HolographicProcessor",
    "HologramType",
    "PhotorefractiveType",
    "ProcessingMode",
    "create_holographic_computing_system",
    "run_holographic_demo",
    "CVQuantumCircuit",
    "CVQuantumNeuralNetwork",
    "CVQuantumOptimizer",
    "CVQuantumState",
    "create_cv_quantum_system",
    "run_cv_quantum_demo",
    # Generation 4+ Breakthrough Enhancements
    "CoherentMatrixMultiplier",
    "QuantumCoherenceResearchSuite", 
    "CoherenceMode",
    "create_breakthrough_coherence_system",
    "run_quantum_coherence_demo",
    "PhotonicLayerType",
    "EvolutionStrategy",
    "PhotonicArchitectureGenome",
    "create_self_evolving_photonic_nas",
    "run_breakthrough_nas_experiment",
    "HolographicFusionEngine",
    "run_holographic_fusion_demo",
    "CVQuantumGate",
    "CVQuantumMode",
    "run_cv_quantum_breakthrough_demo",
]