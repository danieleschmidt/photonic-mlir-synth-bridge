# Photonic MLIR API Reference

Complete API reference for the Photonic MLIR compiler infrastructure.

## Core Components

### photonic_mlir.compiler

#### PhotonicCompiler

Main compiler class for converting PyTorch models to photonic circuits.

```python
class PhotonicCompiler:
    def __init__(self, 
                 backend: PhotonicBackend = PhotonicBackend.SIMULATION_ONLY,
                 wavelengths: List[float] = None,
                 power_budget: float = 100.0)
```

**Parameters:**
- `backend`: Target photonic hardware backend
- `wavelengths`: List of wavelength channels in nanometers (default: [1550.0])
- `power_budget`: Maximum power consumption in milliwatts

**Methods:**

##### compile()

```python
def compile(self, 
           model: Any,
           example_input: Any,
           optimization_level: int = 2) -> PhotonicCircuit
```

Compile PyTorch model to photonic circuit.

**Parameters:**
- `model`: PyTorch model to compile
- `example_input`: Example input tensor for tracing
- `optimization_level`: Optimization level (0-3)

**Returns:**
- `PhotonicCircuit`: Compiled photonic circuit

**Raises:**
- `ValidationError`: If inputs are invalid
- `CompilationError`: If compilation fails

**Example:**
```python
import torch
from photonic_mlir import PhotonicCompiler, PhotonicBackend

model = torch.nn.Linear(784, 10)
compiler = PhotonicCompiler(
    backend=PhotonicBackend.LIGHTMATTER,
    wavelengths=[1550.0, 1551.0, 1552.0, 1553.0],
    power_budget=100.0
)

circuit = compiler.compile(
    model, 
    torch.randn(1, 784),
    optimization_level=3
)
```

#### PhotonicBackend

Enumeration of supported photonic hardware backends.

```python
class PhotonicBackend(Enum):
    LIGHTMATTER = "lightmatter"
    ANALOG_PHOTONICS = "analog_photonics"
    AIM_PHOTONICS = "aim_photonics"
    SIMULATION_ONLY = "simulation"
```

#### PhotonicCircuit

Represents a compiled photonic circuit.

**Attributes:**
- `mlir_module: str`: MLIR representation of the circuit
- `config: Dict[str, Any]`: Compilation configuration

**Methods:**

##### generate_hls()

```python
def generate_hls(self, target: str = "AIM_Photonics_PDK", 
                process_node: str = "45nm_SOI") -> str
```

Generate HLS code for the photonic circuit.

##### save_netlist()

```python
def save_netlist(self, path: str) -> None
```

Save SPICE netlist for the circuit.

##### save_power_report()

```python
def save_power_report(self, path: str) -> None
```

Save power analysis report.

##### save_layout()

```python
def save_layout(self, path: str) -> None
```

Save GDS layout file.

### photonic_mlir.optimization

#### OptimizationPipeline

Pipeline for applying multiple optimization passes.

```python
class OptimizationPipeline:
    def __init__(self)
```

**Methods:**

##### add_pass()

```python
def add_pass(self, pass_instance: PhotonicPass) -> None
```

Add an optimization pass to the pipeline.

##### run()

```python
def run(self, photonic_circuit: PhotonicCircuit) -> PhotonicCircuit
```

Run all optimization passes on the circuit.

##### generate_report()

```python
def generate_report(self) -> OptimizationReport
```

Generate optimization report.

#### PhotonicPasses

Factory class for creating photonic optimization passes.

**Static Methods:**

##### WavelengthAllocation()

```python
@staticmethod
def WavelengthAllocation(channels: int = 4) -> WavelengthAllocationPass
```

Create wavelength allocation optimization pass.

##### ThermalAwarePlacement()

```python
@staticmethod
def ThermalAwarePlacement(max_temp: float = 350.0) -> ThermalAwarePlacementPass
```

Create thermal-aware placement pass.

##### PhaseQuantization()

```python
@staticmethod
def PhaseQuantization(bits: int = 8) -> PhaseQuantizationPass
```

Create phase quantization pass.

##### PowerGating()

```python
@staticmethod
def PowerGating(threshold: float = 0.1) -> PowerGatingPass
```

Create power gating pass.

##### CoherentNoiseReduction()

```python
@staticmethod
def CoherentNoiseReduction(method: str = "balanced_detection") -> CoherentNoiseReductionPass
```

Create coherent noise reduction pass.

**Example:**
```python
from photonic_mlir import OptimizationPipeline, PhotonicPasses

pipeline = OptimizationPipeline()
pipeline.add_pass(PhotonicPasses.WavelengthAllocation(channels=4))
pipeline.add_pass(PhotonicPasses.ThermalAwarePlacement(max_temp=350.0))
pipeline.add_pass(PhotonicPasses.PhaseQuantization(bits=8))

optimized_circuit = pipeline.run(circuit)
report = pipeline.generate_report()
```

### photonic_mlir.simulation

#### PhotonicSimulator

Cycle-accurate photonic circuit simulator.

```python
class PhotonicSimulator:
    def __init__(self,
                 pdk: str = "AIM_Photonics_45nm",
                 temperature: float = 300.0,
                 include_noise: bool = True,
                 monte_carlo_runs: int = 100)
```

**Parameters:**
- `pdk`: Process design kit for device models
- `temperature`: Operating temperature in Kelvin
- `include_noise`: Whether to include noise modeling
- `monte_carlo_runs`: Number of Monte Carlo simulation runs

**Methods:**

##### simulate()

```python
def simulate(self,
            photonic_circuit: PhotonicCircuit,
            test_inputs: torch.Tensor,
            metrics: List[str] = None) -> SimulationMetrics
```

Simulate photonic circuit with test inputs.

**Parameters:**
- `photonic_circuit`: Circuit to simulate
- `test_inputs`: Input data for simulation
- `metrics`: List of metrics to compute ["ber", "snr", "power_consumption"]

**Returns:**
- `SimulationMetrics`: Simulation results

**Example:**
```python
from photonic_mlir import PhotonicSimulator

simulator = PhotonicSimulator(
    pdk="AIM_Photonics_45nm",
    temperature=300.0,
    include_noise=True,
    monte_carlo_runs=100
)

results = simulator.simulate(
    circuit,
    test_inputs=torch.randn(100, 784),
    metrics=["ber", "snr", "power_consumption"]
)

print(f"SNR: {results.snr:.1f} dB")
print(f"Power: {results.power_consumption:.1f} mW")
```

#### HardwareInterface

Interface for connecting to real photonic hardware.

```python
class HardwareInterface:
    def __init__(self,
                 device: str = "lightmatter_envise",
                 calibration_file: Optional[str] = None)
```

**Methods:**

##### connect()

```python
def connect(self) -> bool
```

Connect to photonic hardware.

##### execute()

```python
def execute(self,
           photonic_circuit: PhotonicCircuit,
           test_inputs: torch.Tensor,
           power_limit: float = 100.0) -> SimulationMetrics
```

Execute circuit on photonic hardware.

##### calibrate()

```python
def calibrate(self) -> bool
```

Perform hardware calibration.

### photonic_mlir.validation

#### InputValidator

Comprehensive input validation for photonic compilation.

**Static Methods:**

##### validate_model()

```python
@staticmethod
def validate_model(model: torch.nn.Module) -> Dict[str, Any]
```

Validate PyTorch model for photonic compilation.

##### validate_input_tensor()

```python
@staticmethod
def validate_input_tensor(tensor: torch.Tensor, name: str = "input") -> Dict[str, Any]
```

Validate input tensor for photonic processing.

##### validate_wavelengths()

```python
@staticmethod
def validate_wavelengths(wavelengths: List[float], pdk: str = "AIM_Photonics_PDK") -> Dict[str, Any]
```

Validate wavelength specifications.

##### validate_power_budget()

```python
@staticmethod
def validate_power_budget(power_budget: float) -> Dict[str, Any]
```

Validate power budget specification.

##### validate_optimization_level()

```python
@staticmethod
def validate_optimization_level(level: int) -> Dict[str, Any]
```

Validate optimization level.

### photonic_mlir.cache

#### get_cache_manager()

```python
def get_cache_manager() -> CacheManager
```

Get global cache manager instance.

#### cached_compilation()

```python
def cached_compilation(func: Callable) -> Callable
```

Decorator for caching compilation results.

#### cached_simulation()

```python
def cached_simulation(func: Callable) -> Callable
```

Decorator for caching simulation results.

**Example:**
```python
from photonic_mlir import get_cache_manager, cached_compilation

# Get cache manager
cache = get_cache_manager()
print(f"Cache enabled: {cache.is_enabled()}")
print(f"Cache stats: {cache.get_global_stats()}")

# Use caching decorator
@cached_compilation
def my_compile_function(model, input_data):
    # Expensive compilation operation
    return compile_model(model, input_data)
```

### photonic_mlir.monitoring

#### get_metrics_collector()

```python
def get_metrics_collector() -> MetricsCollector
```

Get global metrics collector instance.

#### get_health_checker()

```python
def get_health_checker() -> HealthChecker
```

Get global health checker instance.

#### performance_monitor()

```python
@contextmanager
def performance_monitor(operation: str, 
                       metrics_collector: Optional[MetricsCollector] = None)
```

Context manager for automatic performance monitoring.

**Example:**
```python
from photonic_mlir import performance_monitor, get_metrics_collector

# Monitor performance
with performance_monitor("model_compilation"):
    circuit = compiler.compile(model, input_data)

# Get metrics
metrics = get_metrics_collector()
stats = metrics.get_performance_summary()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

### photonic_mlir.load_balancer

#### LoadBalancer

Advanced load balancer with multiple strategies and auto-scaling.

```python
class LoadBalancer:
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 health_check_interval: float = 30.0)
```

#### create_local_cluster()

```python
def create_local_cluster(num_nodes: int = 4, 
                        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE) -> DistributedCompiler
```

Create a local cluster for distributed compilation.

**Example:**
```python
from photonic_mlir import create_local_cluster

# Create distributed compiler
cluster = create_local_cluster(num_nodes=4)

# Submit compilation tasks
task_id = cluster.submit_task(compilation_task)

# Check status
stats = cluster.get_stats()
print(f"Active tasks: {stats['active_tasks']}")

# Shutdown when done
cluster.shutdown()
```

### photonic_mlir.research

#### create_comprehensive_research_suite()

```python
def create_comprehensive_research_suite() -> ResearchSuite
```

Create a comprehensive research suite with all standard experiments.

#### PhotonicVsElectronicComparison

Research experiment comparing photonic vs electronic implementations.

**Methods:**

##### run_experiment()

```python
def run_experiment(self, model: Any, test_dataset: Any, num_runs: int = 10) -> ResearchMetrics
```

##### generate_comparison_report()

```python
def generate_comparison_report(self) -> Dict[str, Any]
```

**Example:**
```python
from photonic_mlir.research import create_comprehensive_research_suite

# Create research suite
suite = create_comprehensive_research_suite()

# Run all experiments
suite.run_all_experiments(model, test_dataset, num_runs=10)

# Results saved to research_results/ directory
```

### photonic_mlir.benchmarking

#### create_standard_benchmark_suite()

```python
def create_standard_benchmark_suite() -> BenchmarkSuite
```

Create a standard benchmark suite with common models.

#### run_quick_benchmark()

```python
def run_quick_benchmark() -> BenchmarkSuite
```

Run a quick benchmark with standard configuration.

**Example:**
```python
from photonic_mlir.benchmarking import run_quick_benchmark

# Run quick benchmark
suite = run_quick_benchmark()

# Get results
summary = suite.generate_summary_report()
print(f"Average compilation time: {summary.avg_compilation_time:.2f}s")
```

### photonic_mlir.cli

Command-line interface functions.

#### compile_command()

```python
def compile_command() -> int
```

Main compilation command entry point.

#### simulate_command()

```python
def simulate_command() -> int
```

Simulation command entry point.

#### benchmark_command()

```python
def benchmark_command() -> int
```

Benchmarking command entry point.

## Data Types

### SimulationMetrics

Container for simulation results.

**Attributes:**
- `ber: float`: Bit error rate
- `snr: float`: Signal-to-noise ratio (dB)
- `power_consumption: float`: Power consumption (mW)
- `latency: float`: Latency (μs)
- `throughput: float`: Throughput (TOPS)
- `accuracy_correlation: float`: Accuracy correlation with expected results

### CompilationMetrics

Compilation-specific metrics.

**Attributes:**
- `model_name: str`: Name of the compiled model
- `parameter_count: int`: Number of model parameters
- `compilation_time_ms: float`: Compilation time in milliseconds
- `optimization_level: int`: Optimization level used
- `backend: str`: Target backend
- `wavelength_count: int`: Number of wavelength channels
- `power_budget_mw: float`: Power budget in milliwatts
- `success: bool`: Whether compilation succeeded

### OptimizationReport

Report from optimization pipeline.

**Attributes:**
- `power_savings: float`: Power savings percentage
- `area_savings: float`: Area savings percentage
- `latency_improvement: float`: Latency improvement percentage
- `pass_results: Dict[str, Dict[str, float]]`: Results from individual passes

## Exceptions

### ValidationError

```python
class ValidationError(Exception):
    def __init__(self, message: str, parameter: str = None, expected: Any = None, got: Any = None)
```

Raised when input validation fails.

### CompilationError

```python
class CompilationError(Exception):
    def __init__(self, message: str, phase: str = None, details: Dict[str, Any] = None)
```

Raised when compilation fails.

### PowerBudgetExceededError

```python
class PowerBudgetExceededError(ValidationError):
    def __init__(self, estimated_power: float, budget: float)
```

Raised when estimated power consumption exceeds budget.

### WavelengthConflictError

```python
class WavelengthConflictError(ValidationError):
    def __init__(self, conflicting_wavelengths: List[float])
```

Raised when wavelength specifications conflict.

## Constants

### Wavelength Ranges

Valid wavelength ranges for different PDKs:

```python
VALID_WAVELENGTH_RANGES = {
    "AIM_Photonics_PDK": (1500, 1600),      # nm
    "IMEC_SiPhotonics": (1520, 1580),       # nm
    "GlobalFoundries": (1530, 1570),        # nm
}
```

### Power Limits

```python
MIN_POWER_BUDGET = 1.0      # mW
MAX_POWER_BUDGET = 1000.0   # mW
```

### Tensor Limits

```python
MAX_TENSOR_DIMS = 6
MAX_TENSOR_SIZE = 1e9       # elements
```

## Environment Variables

### Configuration

- `PHOTONIC_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PHOTONIC_CACHE_ENABLED`: Enable caching (true/false)
- `PHOTONIC_CACHE_SIZE_MB`: Cache size in MB
- `PHOTONIC_MAX_WORKERS`: Maximum number of worker threads
- `PHOTONIC_DEBUG`: Enable debug mode (true/false)

### Hardware

- `PHOTONIC_ENABLE_GPU`: Enable GPU acceleration (true/false)
- `PHOTONIC_GPU_MEMORY_FRACTION`: GPU memory fraction (0.0-1.0)

### Security

- `PHOTONIC_ENABLE_SECURITY`: Enable security features (true/false)
- `PHOTONIC_JWT_SECRET`: JWT secret key

## Version Information

```python
import photonic_mlir
print(photonic_mlir.__version__)  # "0.1.0"
print(photonic_mlir.__author__)   # "Photonic MLIR Team"
```

## Complete Example

```python
import torch
import torch.nn as nn
from photonic_mlir import (
    PhotonicCompiler, PhotonicBackend,
    OptimizationPipeline, PhotonicPasses,
    PhotonicSimulator, performance_monitor
)

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Create model and input
model = SimpleModel()
model.eval()
example_input = torch.randn(1, 784)

# Compile with monitoring
with performance_monitor("full_compilation"):
    # Create compiler
    compiler = PhotonicCompiler(
        backend=PhotonicBackend.LIGHTMATTER,
        wavelengths=[1550.0, 1551.0, 1552.0, 1553.0],
        power_budget=100.0
    )
    
    # Compile model
    circuit = compiler.compile(
        model,
        example_input,
        optimization_level=3
    )
    
    # Apply additional optimizations
    pipeline = OptimizationPipeline()
    pipeline.add_pass(PhotonicPasses.WavelengthAllocation(channels=4))
    pipeline.add_pass(PhotonicPasses.ThermalAwarePlacement(max_temp=350.0))
    
    optimized_circuit = pipeline.run(circuit)
    
    # Simulate
    simulator = PhotonicSimulator(
        pdk="AIM_Photonics_45nm",
        temperature=300.0,
        include_noise=True
    )
    
    results = simulator.simulate(
        optimized_circuit,
        torch.randn(100, 784),
        metrics=["ber", "snr", "power_consumption"]
    )

# Generate outputs
hls_code = optimized_circuit.generate_hls()
optimized_circuit.save_netlist("circuit.sp")
optimized_circuit.save_power_report("power_report.txt")

print(f"Compilation successful!")
print(f"SNR: {results.snr:.1f} dB")
print(f"Power: {results.power_consumption:.1f} mW")
print(f"Optimization report: {pipeline.generate_report().power_savings:.1f}% power savings")
```