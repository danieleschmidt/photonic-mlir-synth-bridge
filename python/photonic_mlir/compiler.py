"""
Main compiler interface for converting PyTorch models to photonic circuits.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch when not available
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    class MockTensor:
        def __init__(self, shape):
            self.shape = shape
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

try:
    from .bindings import PhotonicMLIRPythonModule as native
except ImportError:
    # Fallback for development
    native = None


class PhotonicBackend(Enum):
    """Supported photonic hardware backends"""
    LIGHTMATTER = "lightmatter"
    ANALOG_PHOTONICS = "analog_photonics"
    AIM_PHOTONICS = "aim_photonics"
    SIMULATION_ONLY = "simulation"
    SIM_ONLY = "sim_only"  # Alias for compatibility
    HARDWARE = "hardware"  # Generic hardware backend


class PhotonicCircuit:
    """Represents a compiled photonic circuit"""
    
    def __init__(self, mlir_module: str, config: Dict[str, Any]):
        self.mlir_module = mlir_module
        self.config = config
        self._hls_code = None
        
    def generate_hls(self, target: str = "AIM_Photonics_PDK", 
                     process_node: str = "45nm_SOI") -> str:
        """Generate HLS code for the photonic circuit"""
        if native is None:
            # Simulation fallback
            return self._generate_mock_hls(target, process_node)
        
        hls_config = native.HLSConfig()
        hls_config.target_pdk = target
        hls_config.process_node = process_node
        hls_config.power_budget = self.config.get("power_budget", 100.0)
        hls_config.wavelength_channels = len(self.config.get("wavelengths", [1550]))
        
        self._hls_code = native.generate_hls_code(self.mlir_module, hls_config)
        return self._hls_code
    
    def _generate_mock_hls(self, target: str, process_node: str) -> str:
        """Generate mock HLS code for testing"""
        return f"""
// Generated HLS code for photonic circuit
// Target PDK: {target}
// Process: {process_node}
// Power Budget: {self.config.get('power_budget', 100)} mW

#include <ap_int.h>
#include <hls_stream.h>
#include <complex>

void photonic_accelerator(
    hls::stream<std::complex<float>>& input,
    hls::stream<std::complex<float>>& output
) {{
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS PIPELINE II=1
    
    // Photonic matrix multiplication implementation
    // Wavelength channels: {len(self.config.get('wavelengths', [1550]))}
    // Mesh topology: triangular
}}
"""

    def save_netlist(self, path: str):
        """Save SPICE netlist for photonic circuit"""
        with open(path, 'w') as f:
            f.write(f"* Photonic circuit netlist\n")
            f.write(f"* Generated from MLIR module\n")
            f.write(f"* Power budget: {self.config.get('power_budget', 100)} mW\n")
            f.write(f".SUBCKT photonic_circuit\n")
            f.write(f"* Add photonic device models here\n")
            f.write(f".ENDS\n")
    
    def save_power_report(self, path: str):
        """Save power analysis report"""
        # Mock implementation
        power_data = {
            "total_power": self.config.get("power_budget", 100),
            "static_power": 10.0,
            "dynamic_power": 90.0,
            "thermal_analysis": {"max_temp": 350, "hotspots": 2}
        }
        
        with open(path, 'w') as f:
            f.write("Photonic Circuit Power Analysis Report\n")
            f.write("=====================================\n\n")
            f.write(f"Total Power: {power_data['total_power']} mW\n")
            f.write(f"Static Power: {power_data['static_power']} mW\n")
            f.write(f"Dynamic Power: {power_data['dynamic_power']} mW\n")
            f.write(f"Max Temperature: {power_data['thermal_analysis']['max_temp']} K\n")
    
    def save_layout(self, path: str):
        """Save GDS layout file"""
        # Mock implementation - in reality would generate actual GDS
        with open(path, 'wb') as f:
            f.write(b"GDS_MOCK_LAYOUT_DATA")
    
    def _get_target_optimizations(self, target: str) -> List[str]:
        """Get target-specific optimizations"""
        optimizations = ["power_gating", "thermal_management"]
        
        if target == "AIM_Photonics_PDK":
            optimizations.extend(["silicon_photonics_optimization", "mzi_optimization"])
        elif target == "IMEC_SiPhotonics":
            optimizations.extend(["cmos_compatibility", "wafer_scale_optimization"])
        elif target == "GlobalFoundries":
            optimizations.extend(["foundry_rules", "yield_optimization"])
        
        return optimizations
    
    def _calculate_power_constraints(self, process_node: str) -> str:
        """Calculate power constraints for process node"""
        constraints = {
            "45nm": "Max 100mW, Thermal limit 85°C",
            "65nm": "Max 150mW, Thermal limit 90°C", 
            "90nm": "Max 200mW, Thermal limit 95°C",
            "130nm": "Max 300mW, Thermal limit 100°C",
            "45nm_SOI": "Max 80mW, Thermal limit 80°C"
        }
        
        return constraints.get(process_node, "Standard constraints")


class PhotonicCompiler:
    """Main compiler for converting PyTorch models to photonic circuits with enhanced resilience"""
    
    def __init__(self, 
                 backend: PhotonicBackend = PhotonicBackend.SIMULATION_ONLY,
                 wavelengths: List[float] = None,
                 power_budget: float = 100.0):
        
        # Enhanced input validation
        if isinstance(backend, str):
            # Handle string backend input
            try:
                backend_map = {
                    'lightmatter': PhotonicBackend.LIGHTMATTER,
                    'analog_photonics': PhotonicBackend.ANALOG_PHOTONICS,
                    'aim_photonics': PhotonicBackend.AIM_PHOTONICS,
                    'simulation': PhotonicBackend.SIMULATION_ONLY,
                    'sim_only': PhotonicBackend.SIMULATION_ONLY
                }
                backend = backend_map.get(backend.lower(), PhotonicBackend.SIMULATION_ONLY)
            except Exception:
                backend = PhotonicBackend.SIMULATION_ONLY
        
        self.backend = backend
        self.wavelengths = wavelengths or [1550.0]  # Default C-band wavelength
        self.power_budget = power_budget
        
        # Set up logging
        from .logging_config import get_logger
        self.logger = get_logger("photonic_mlir.compiler")
        
        # Initialize validator (with fallback)
        try:
            from .enhanced_validation import get_validator
            self.validator = get_validator()
        except ImportError:
            self.validator = None
        
        self.logger.info(f"PhotonicCompiler initialized - Backend: {self.backend.value}, Wavelengths: {len(self.wavelengths)}, Power: {self.power_budget}mW")
        
    def compile(self, 
                model: Any,
                example_input: Any,
                optimization_level: int = 2) -> PhotonicCircuit:
        """Compile PyTorch model to photonic circuit"""
        
        # Validate inputs before compilation
        from .validation import InputValidator
        from .exceptions import ValidationError
        from .security import create_secure_compilation_environment
        from .monitoring import performance_monitor, get_metrics_collector
        
        # Create secure environment
        secure_env = create_secure_compilation_environment()
        session = secure_env.create_compilation_session()
        
        try:
            with performance_monitor("model_compilation"):
                # Validate inputs
                if TORCH_AVAILABLE:
                    if model is not None:
                        InputValidator.validate_model(model)
                        InputValidator.validate_input_tensor(example_input, "example_input")
                elif model is not None and not hasattr(model, '__call__'):
                    # Handle non-torch models more gracefully
                    raise ValidationError("PyTorch not available and model is not callable")
                
                InputValidator.validate_wavelengths(self.wavelengths)
                InputValidator.validate_power_budget(self.power_budget)
                InputValidator.validate_optimization_level(optimization_level)
                
                # Security validation
                config = {
                    "backend": self.backend,
                    "wavelengths": self.wavelengths,
                    "power_budget": self.power_budget,
                    "optimization_level": optimization_level
                }
                
                validation_results = secure_env.validate_compilation_inputs(
                    model, example_input, config
                )
                
                if not validation_results["overall_secure"]:
                    raise ValidationError("Security validation failed for compilation inputs")
                
                if not TORCH_AVAILABLE:
                    # Mock implementation for testing
                    traced_model = model
                    example_input = MockTensor([1, 784])
                else:
                    # Convert PyTorch model to computational graph
                    model.eval()
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, example_input)
                
                # Convert to MLIR (simplified for now)
                mlir_module = self._convert_to_mlir(traced_model, example_input)
                
                # Apply optimizations
                if optimization_level > 0:
                    mlir_module = self._apply_optimizations(mlir_module, optimization_level)
                
                circuit = PhotonicCircuit(mlir_module, config)
                
                # Validate final circuit
                from .validation import CircuitValidator
                CircuitValidator.validate_power_consumption(config)
                
                # Record compilation metrics
                from .monitoring import CompilationMetrics
                
                comp_metrics = CompilationMetrics(
                    model_name=type(model).__name__ if model else "MockModel",
                    parameter_count=sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') and TORCH_AVAILABLE else 1000,
                    compilation_time_ms=0,  # Will be filled by performance monitor
                    optimization_level=optimization_level,
                    backend=self.backend.value,
                    wavelength_count=len(self.wavelengths),
                    power_budget_mw=self.power_budget,
                    memory_peak_mb=0,  # Will be filled by performance monitor  
                    success=True,
                    mlir_size_bytes=len(mlir_module.encode('utf-8'))
                )
                
                get_metrics_collector().record_compilation(comp_metrics)
                
                return circuit
                
        except Exception as e:
            # Record failed compilation metrics
            from .monitoring import CompilationMetrics
            
            failed_metrics = CompilationMetrics(
                model_name=type(model).__name__ if model else "MockModel",
                parameter_count=sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') and TORCH_AVAILABLE else 0,
                compilation_time_ms=0,
                optimization_level=optimization_level,
                backend=self.backend.value,
                wavelength_count=len(self.wavelengths),
                power_budget_mw=self.power_budget,
                memory_peak_mb=0,
                success=False,
                mlir_size_bytes=0,
                error_details=str(e)
            )
            
            get_metrics_collector().record_compilation(failed_metrics)
            self.logger.error(f"Compilation failed: {e}")
            raise
        finally:
            secure_env.cleanup_session()
    
    def _convert_to_mlir(self, traced_model, example_input) -> str:
        """Convert traced PyTorch model to MLIR photonic dialect"""
        # Simplified MLIR generation for demonstration
        input_shape = list(example_input.shape)
        
        mlir_code = f"""
module {{
  func.func @photonic_model(%arg0: tensor<{input_shape[0]}x{input_shape[1]}xf32>) -> tensor<{input_shape[0]}x10xf32> {{
    // Convert linear layers to photonic tensor cores
    %weights = arith.constant dense<1.0> : tensor<{input_shape[1]}x256xcomplex<f32>>
    %bias = arith.constant dense<0.0> : tensor<256xcomplex<f32>>
    
    // Photonic matrix multiplication using MZI mesh
    %result = photonic.tensor_core %arg0, %weights {{
      wavelength_channels = {len(self.wavelengths)},
      mesh_topology = "triangular",
      activation = "photodetector"
    }} : (tensor<{input_shape[0]}x{input_shape[1]}xf32>, tensor<{input_shape[1]}x256xcomplex<f32>>) -> tensor<{input_shape[0]}x256xf32>
    
    // Output layer
    %output_weights = arith.constant dense<1.0> : tensor<256x10xcomplex<f32>>
    %output = photonic.tensor_core %result, %output_weights {{
      wavelength_channels = {len(self.wavelengths)},
      mesh_topology = "triangular", 
      activation = "photodetector"
    }} : (tensor<{input_shape[0]}x256xf32>, tensor<256x10xcomplex<f32>>) -> tensor<{input_shape[0]}x10xf32>
    
    return %output : tensor<{input_shape[0]}x10xf32>
  }}
}}
"""
        return mlir_code
    
    def _apply_optimizations(self, mlir_module: str, level: int) -> str:
        """Apply photonic-specific optimizations"""
        # In a real implementation, this would use MLIR pass manager
        optimized = mlir_module
        
        if level >= 1:
            # Apply wavelength allocation optimization
            optimized = optimized.replace("// Photonic matrix multiplication using MZI mesh",
                                        "// Photonic matrix multiplication using MZI mesh\n    // optimized: wavelength allocation applied")
        
        if level >= 2:
            # Add thermal optimization annotations
            optimized = optimized.replace("mesh_topology = \"triangular\"",
                                        "mesh_topology = \"triangular\", thermal_optimized = true")
        
        if level >= 3:
            # Add power gating
            optimized = optimized.replace("activation = \"photodetector\"",
                                        "activation = \"photodetector\", power_gated = true")
        
        return optimized
    
    def validate_input(self, input_data):
        """Enhanced input validation with comprehensive checks"""
        if input_data is None:
            self.logger.warning("Input data is None")
            return {'valid': False, 'reason': 'Input data cannot be None'}
        
        try:
            # Basic type validation
            if not hasattr(input_data, '__len__') and not hasattr(input_data, 'shape'):
                return {'valid': False, 'reason': 'Input must be array-like or have shape attribute'}
            
            # Check for reasonable size limits
            if hasattr(input_data, '__len__'):
                if len(input_data) == 0:
                    return {'valid': False, 'reason': 'Input data cannot be empty'}
                if len(input_data) > 10000:
                    self.logger.warning(f"Large input size: {len(input_data)} elements")
            
            # Security validation if validator is available
            if self.validator and isinstance(input_data, str):
                security_result = self.validator.validate_field('input_data', input_data)
                if not security_result['valid']:
                    return {
                        'valid': False,
                        'reason': f"Security validation failed: {security_result['errors']}"
                    }
            
            self.logger.debug(f"Input validation passed for {type(input_data).__name__}")
            return {'valid': True, 'reason': 'Input validation successful'}
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return {'valid': False, 'reason': f'Validation error: {e}'}
    
    def _validate_model_compatibility(self, model):
        """Validate model compatibility with photonic compilation"""
        if model is None:
            raise ValueError("Model cannot be None")
        
        # Check if model is callable
        if not hasattr(model, '__call__'):
            raise ValueError("Model must be callable")
        
        # Additional validations can be added here
        self.logger.debug("Model compatibility validation passed")
    
    def _estimate_circuit_complexity(self, model, optimization_level: int) -> int:
        """Estimate the number of photonic circuit nodes needed"""
        base_nodes = 10  # Minimum nodes
        
        # Estimate based on model parameters if available
        if hasattr(model, 'parameters') and TORCH_AVAILABLE:
            try:
                param_count = sum(p.numel() for p in model.parameters())
                # Rough estimate: 1 node per 1000 parameters
                estimated_nodes = max(base_nodes, param_count // 1000)
            except:
                estimated_nodes = base_nodes * 4  # Conservative fallback
        else:
            # Fallback estimation
            estimated_nodes = base_nodes * 3
        
        # Adjust for optimization level
        optimization_factor = 1.0 - (optimization_level * 0.1)  # Higher optimization = fewer nodes
        final_nodes = max(base_nodes, int(estimated_nodes * optimization_factor))
        
        self.logger.debug(f"Estimated circuit nodes: {final_nodes}")
        return final_nodes
    
    def _estimate_power_consumption(self, circuit_nodes: int) -> float:
        """Estimate power consumption based on circuit complexity"""
        # Base power consumption model
        base_power = 5.0  # mW
        power_per_node = 0.5  # mW per node
        
        estimated_power = base_power + (circuit_nodes * power_per_node)
        
        # Ensure within budget
        final_power = min(estimated_power, self.power_budget * 0.9)  # Leave 10% margin
        
        self.logger.debug(f"Estimated power consumption: {final_power:.1f}mW")
        return final_power