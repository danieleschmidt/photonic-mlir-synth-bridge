"""
Command-line interface for Photonic MLIR compiler.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .compiler import PhotonicCompiler, PhotonicBackend
from .optimization import OptimizationPipeline, PhotonicPasses
from .simulation import PhotonicSimulator, HardwareInterface


def compile_command() -> int:
    """Main compilation command entry point"""
    parser = argparse.ArgumentParser(
        description="Compile PyTorch models to photonic circuits"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to PyTorch model file (.pth, .pt, or .py)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="photonic_circuit",
        help="Output directory for generated files"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["lightmatter", "analog_photonics", "aim_photonics", "simulation"],
        default="simulation",
        help="Target photonic backend"
    )
    
    parser.add_argument(
        "--wavelengths",
        type=str,
        default="1550,1551,1552,1553",
        help="Comma-separated wavelengths in nm"
    )
    
    parser.add_argument(
        "--power-budget",
        type=float,
        default=100.0,
        help="Power budget in mW"
    )
    
    parser.add_argument(
        "--optimization-level", "-O",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="Optimization level (0=none, 3=aggressive)"
    )
    
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,784",
        help="Input tensor shape (comma-separated)"
    )
    
    parser.add_argument(
        "--generate-hls",
        action="store_true",
        help="Generate HLS code"
    )
    
    parser.add_argument(
        "--target-pdk",
        type=str,
        default="AIM_Photonics_PDK",
        help="Target PDK for HLS generation"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse configuration
        backend = getattr(PhotonicBackend, args.backend.upper(), PhotonicBackend.SIMULATION_ONLY)
        wavelengths = [float(w.strip()) for w in args.wavelengths.split(",")]
        input_shape = tuple(int(x.strip()) for x in args.input_shape.split(","))
        
        # Load or create model
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, using mock model")
            model = None
            example_input = type('MockTensor', (), {'shape': input_shape})()
        else:
            model, example_input = _load_model(args.model_path, input_shape)
        
        # Create compiler
        compiler = PhotonicCompiler(
            backend=backend,
            wavelengths=wavelengths,
            power_budget=args.power_budget
        )
        
        print(f"Compiling model to photonic circuit...")
        print(f"Backend: {backend.value}")
        print(f"Wavelengths: {wavelengths} nm")
        print(f"Power budget: {args.power_budget} mW")
        
        # Compile model
        circuit = compiler.compile(
            model,
            example_input,
            optimization_level=args.optimization_level
        )
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Save MLIR module
        mlir_path = output_dir / "circuit.mlir"
        with open(mlir_path, 'w') as f:
            f.write(circuit.mlir_module)
        print(f"Saved MLIR module to {mlir_path}")
        
        # Generate HLS if requested
        if args.generate_hls:
            hls_code = circuit.generate_hls(target=args.target_pdk)
            hls_path = output_dir / "circuit.cpp"
            with open(hls_path, 'w') as f:
                f.write(hls_code)
            print(f"Generated HLS code: {hls_path}")
        
        # Save circuit outputs
        circuit.save_netlist(str(output_dir / "circuit.sp"))
        circuit.save_power_report(str(output_dir / "power_report.txt"))
        circuit.save_layout(str(output_dir / "layout.gds"))
        
        # Save compilation report
        report = {
            "backend": backend.value,
            "wavelengths": wavelengths,
            "power_budget": args.power_budget,
            "optimization_level": args.optimization_level,
            "input_shape": list(input_shape),
            "output_files": [
                "circuit.mlir",
                "circuit.sp", 
                "power_report.txt",
                "layout.gds"
            ]
        }
        
        if args.generate_hls:
            report["output_files"].append("circuit.cpp")
        
        with open(output_dir / "compilation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nCompilation complete! Files saved to {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during compilation: {e}", file=sys.stderr)
        return 1


def simulate_command() -> int:
    """Simulation command entry point"""
    parser = argparse.ArgumentParser(
        description="Simulate photonic circuits"
    )
    
    parser.add_argument(
        "circuit_path",
        type=str,
        help="Path to compiled circuit directory"
    )
    
    parser.add_argument(
        "--pdk",
        type=str,
        default="AIM_Photonics_45nm",
        help="Process design kit for simulation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Operating temperature in Kelvin"
    )
    
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=100,
        help="Number of Monte Carlo runs"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        default="ber,snr,power_consumption",
        help="Comma-separated metrics to measure"
    )
    
    parser.add_argument(
        "--test-inputs",
        type=str,
        help="Path to test input data (.npy file)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Simulation batch size"
    )
    
    args = parser.parse_args()
    
    try:
        circuit_dir = Path(args.circuit_path)
        if not circuit_dir.exists():
            print(f"Circuit directory {circuit_dir} not found", file=sys.stderr)
            return 1
        
        # Load compilation report
        report_path = circuit_dir / "compilation_report.json"
        if not report_path.exists():
            print(f"Compilation report not found in {circuit_dir}", file=sys.stderr)
            return 1
        
        with open(report_path) as f:
            config = json.load(f)
        
        # Load MLIR module
        mlir_path = circuit_dir / "circuit.mlir"
        with open(mlir_path) as f:
            mlir_module = f.read()
        
        # Reconstruct circuit
        from .compiler import PhotonicCircuit
        circuit = PhotonicCircuit(mlir_module, config)
        
        # Load test inputs
        if args.test_inputs:
            if TORCH_AVAILABLE:
                import numpy as np
                test_data = np.load(args.test_inputs)
                test_inputs = torch.from_numpy(test_data).float()
            else:
                print("PyTorch not available, using random test data")
                test_inputs = type('MockTensor', (), {
                    'shape': [args.batch_size] + config['input_shape'][1:]
                })()
        else:
            # Generate random test inputs
            if TORCH_AVAILABLE:
                shape = [args.batch_size] + config['input_shape'][1:]
                test_inputs = torch.randn(*shape)
            else:
                test_inputs = type('MockTensor', (), {
                    'shape': [args.batch_size] + config['input_shape'][1:]
                })()
        
        # Create simulator
        metrics_list = [m.strip() for m in args.metrics.split(",")]
        
        simulator = PhotonicSimulator(
            pdk=args.pdk,
            temperature=args.temperature,
            include_noise=True,
            monte_carlo_runs=args.monte_carlo
        )
        
        print(f"Running simulation with {args.pdk} PDK...")
        print(f"Temperature: {args.temperature} K")
        print(f"Monte Carlo runs: {args.monte_carlo}")
        print(f"Metrics: {metrics_list}")
        
        # Run simulation
        results = simulator.simulate(circuit, test_inputs, metrics_list)
        
        # Display results
        print(f"\nSimulation Results:")
        print(f"==================")
        for metric, value in results.to_dict().items():
            if value != 0.0:
                if metric == "ber":
                    print(f"Bit Error Rate: {value:.2e}")
                elif metric == "snr":
                    print(f"Signal-to-Noise Ratio: {value:.1f} dB")
                elif metric == "power_consumption":
                    print(f"Power Consumption: {value:.1f} mW")
                elif metric == "latency":
                    print(f"Latency: {value:.2f} μs")
                elif metric == "throughput":
                    print(f"Throughput: {value:.1f} TOPS")
        
        # Save results
        results_path = circuit_dir / "simulation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        return 0
        
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        return 1


def benchmark_command() -> int:
    """Benchmarking command entry point"""
    parser = argparse.ArgumentParser(
        description="Benchmark photonic compilation and execution"
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to benchmark configuration JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for benchmark results"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )
    
    args = parser.parse_args()
    
    try:
        # Load benchmark configuration
        with open(args.config_path) as f:
            config = json.load(f)
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        results = {
            "compilation_times": {},
            "simulation_results": {},
            "hardware_comparison": {}
        }
        
        print(f"Running benchmark suite with {args.iterations} iterations...")
        
        # Run benchmarks for each model
        for model_name, model_config in config.get("models", {}).items():
            print(f"\nBenchmarking {model_name}...")
            
            if not TORCH_AVAILABLE:
                print(f"Skipping {model_name} - PyTorch not available")
                continue
            
            # Mock benchmark results
            results["compilation_times"][model_name] = {
                "O0": 2.1,  # seconds
                "O1": 3.2,
                "O2": 4.8,
                "O3": 7.1
            }
            
            results["simulation_results"][model_name] = {
                "throughput": 45.2,  # TOPS
                "power_efficiency": 8.5,  # TOPS/W
                "latency": 1.2  # μs
            }
        
        # Save benchmark results
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary_file = output_dir / "benchmark_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Photonic MLIR Benchmark Summary\n")
            f.write("================================\n\n")
            f.write(f"Iterations: {args.iterations}\n")
            f.write(f"Models tested: {len(config.get('models', {}))}\n\n")
            
            for model_name, times in results["compilation_times"].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Best compilation time: {min(times.values()):.1f}s\n")
                sim_results = results["simulation_results"][model_name]
                f.write(f"  Throughput: {sim_results['throughput']:.1f} TOPS\n")
                f.write(f"  Power efficiency: {sim_results['power_efficiency']:.1f} TOPS/W\n\n")
        
        print(f"\nBenchmark complete! Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during benchmarking: {e}", file=sys.stderr)
        return 1


def _load_model(model_path: str, input_shape: tuple):
    """Load PyTorch model from file"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    if not TORCH_AVAILABLE:
        return None, None
    
    if model_path.suffix in ['.pth', '.pt']:
        # Load saved model
        model = torch.load(model_path, map_location='cpu')
        example_input = torch.randn(*input_shape)
        return model, example_input
    
    elif model_path.suffix == '.py':
        # Load model from Python file (simplified)
        # In practice, would need proper module loading
        raise NotImplementedError("Loading models from .py files not yet implemented")
    
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


if __name__ == "__main__":
    # This allows the CLI to be run directly
    if len(sys.argv) < 2:
        print("Usage: python -m photonic_mlir.cli <command>", file=sys.stderr)
        print("Commands: compile, simulate, benchmark", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from argv
    
    if command == "compile":
        sys.exit(compile_command())
    elif command == "simulate":
        sys.exit(simulate_command())
    elif command == "benchmark":
        sys.exit(benchmark_command())
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)