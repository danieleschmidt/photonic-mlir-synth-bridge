"""
Tests for the PhotonicCompiler class.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from photonic_mlir.compiler import PhotonicCompiler, PhotonicBackend, PhotonicCircuit
from photonic_mlir.exceptions import ValidationError, CompilationError


class TestPhotonicCompiler:
    """Test PhotonicCompiler functionality"""
    
    def test_compiler_initialization(self):
        """Test compiler initialization with different parameters"""
        # Default initialization
        compiler = PhotonicCompiler()
        assert compiler.backend == PhotonicBackend.SIMULATION_ONLY
        assert compiler.wavelengths == [1550.0]
        assert compiler.power_budget == 100.0
        
        # Custom initialization
        wavelengths = [1550.0, 1551.0, 1552.0]
        compiler = PhotonicCompiler(
            backend=PhotonicBackend.LIGHTMATTER,
            wavelengths=wavelengths,
            power_budget=200.0
        )
        assert compiler.backend == PhotonicBackend.LIGHTMATTER
        assert compiler.wavelengths == wavelengths
        assert compiler.power_budget == 200.0
    
    def test_simple_model_compilation(self, compiler, simple_model, example_input):
        """Test compilation of a simple PyTorch model"""
        circuit = compiler.compile(simple_model, example_input)
        
        assert isinstance(circuit, PhotonicCircuit)
        assert circuit.mlir_module is not None
        assert len(circuit.mlir_module) > 0
        assert circuit.config["backend"] == PhotonicBackend.SIMULATION_ONLY
        assert circuit.config["wavelengths"] == [1550.0, 1551.0]
        assert circuit.config["power_budget"] == 100.0
    
    def test_photonic_mlp_compilation(self, compiler, photonic_mlp, mlp_input):
        """Test compilation of photonic MLP model"""
        circuit = compiler.compile(photonic_mlp, mlp_input, optimization_level=2)
        
        assert isinstance(circuit, PhotonicCircuit)
        assert "photonic.tensor_core" in circuit.mlir_module
        assert circuit.config["optimization_level"] == 2
    
    def test_optimization_levels(self, compiler, simple_model, example_input):
        """Test different optimization levels"""
        for level in [0, 1, 2, 3]:
            circuit = compiler.compile(simple_model, example_input, optimization_level=level)
            assert circuit.config["optimization_level"] == level
            
            # Check that higher optimization levels produce different MLIR
            if level > 0:
                assert "optimized" in circuit.mlir_module or "thermal" in circuit.mlir_module
    
    def test_invalid_model_compilation(self, compiler):
        """Test compilation with invalid model"""
        with pytest.raises(ValidationError):
            # Try to compile non-PyTorch model
            compiler.compile("not a model", torch.randn(1, 10))
        
        # Test model in training mode
        model = nn.Linear(10, 5)
        model.train()  # Should be in eval mode
        
        with pytest.raises(ValidationError):
            compiler.compile(model, torch.randn(1, 10))
    
    def test_invalid_input_compilation(self, compiler, simple_model):
        """Test compilation with invalid input"""
        with pytest.raises(ValidationError):
            # Try to compile with non-tensor input
            compiler.compile(simple_model, "not a tensor")
        
        with pytest.raises(ValidationError):
            # Try to compile with tensor containing NaN
            invalid_input = torch.tensor([[float('nan')]])
            compiler.compile(simple_model, invalid_input)
    
    def test_backend_specific_compilation(self, simple_model, example_input):
        """Test compilation for different backends"""
        backends = [
            PhotonicBackend.SIMULATION_ONLY,
            PhotonicBackend.LIGHTMATTER,
            PhotonicBackend.ANALOG_PHOTONICS
        ]
        
        for backend in backends:
            compiler = PhotonicCompiler(backend=backend)
            circuit = compiler.compile(simple_model, example_input)
            assert circuit.config["backend"] == backend
    
    def test_wavelength_configuration(self, simple_model, example_input):
        """Test compilation with different wavelength configurations"""
        wavelength_configs = [
            [1550.0],
            [1550.0, 1551.0],
            [1550.0, 1551.0, 1552.0, 1553.0],
            list(range(1530, 1570, 2))  # Dense WDM
        ]
        
        for wavelengths in wavelength_configs:
            compiler = PhotonicCompiler(wavelengths=wavelengths)
            circuit = compiler.compile(simple_model, example_input)
            assert circuit.config["wavelengths"] == wavelengths
    
    def test_power_budget_constraints(self, simple_model, example_input):
        """Test power budget validation during compilation"""
        # Normal power budget
        compiler = PhotonicCompiler(power_budget=100.0)
        circuit = compiler.compile(simple_model, example_input)
        assert circuit.config["power_budget"] == 100.0
        
        # Very low power budget
        compiler = PhotonicCompiler(power_budget=5.0)
        circuit = compiler.compile(simple_model, example_input)
        # Should still compile but may have warnings
        
        # Very high power budget
        compiler = PhotonicCompiler(power_budget=500.0)
        circuit = compiler.compile(simple_model, example_input)
        assert circuit.config["power_budget"] == 500.0


class TestPhotonicCircuit:
    """Test PhotonicCircuit functionality"""
    
    def test_circuit_initialization(self):
        """Test circuit initialization"""
        mlir_code = "module { func.func @test() { return } }"
        config = {"backend": PhotonicBackend.SIMULATION_ONLY}
        
        circuit = PhotonicCircuit(mlir_code, config)
        assert circuit.mlir_module == mlir_code
        assert circuit.config == config
    
    def test_hls_generation(self, compiler, simple_model, example_input):
        """Test HLS code generation from circuit"""
        circuit = compiler.compile(simple_model, example_input)
        
        hls_code = circuit.generate_hls()
        assert isinstance(hls_code, str)
        assert len(hls_code) > 0
        assert "#include" in hls_code
        assert "photonic" in hls_code.lower()
    
    def test_hls_generation_with_different_targets(self, compiler, simple_model, example_input):
        """Test HLS generation for different target PDKs"""
        circuit = compiler.compile(simple_model, example_input)
        
        targets = [
            ("AIM_Photonics_PDK", "45nm_SOI"),
            ("IMEC_SiPhotonics", "220nm_SOI"),
            ("GlobalFoundries", "180nm_CMOS")
        ]
        
        for target, process in targets:
            hls_code = circuit.generate_hls(target=target, process_node=process)
            assert target in hls_code
            assert process in hls_code
    
    def test_netlist_saving(self, compiler, simple_model, example_input, temp_dir):
        """Test saving SPICE netlist"""
        circuit = compiler.compile(simple_model, example_input)
        
        netlist_path = f"{temp_dir}/test_circuit.sp"
        circuit.save_netlist(netlist_path)
        
        # Check file was created and has content
        import os
        assert os.path.exists(netlist_path)
        
        with open(netlist_path, 'r') as f:
            content = f.read()
            assert "photonic circuit netlist" in content.lower()
            assert ".SUBCKT" in content
    
    def test_power_report_saving(self, compiler, simple_model, example_input, temp_dir):
        """Test saving power analysis report"""
        circuit = compiler.compile(simple_model, example_input)
        
        report_path = f"{temp_dir}/power_report.txt"
        circuit.save_power_report(report_path)
        
        import os
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            content = f.read()
            assert "Power Analysis Report" in content
            assert "Total Power" in content
    
    def test_layout_saving(self, compiler, simple_model, example_input, temp_dir):
        """Test saving GDS layout"""
        circuit = compiler.compile(simple_model, example_input)
        
        layout_path = f"{temp_dir}/test_layout.gds"
        circuit.save_layout(layout_path)
        
        import os
        assert os.path.exists(layout_path)
        # For mock implementation, just check file exists
        assert os.path.getsize(layout_path) > 0


class TestCompilerIntegration:
    """Integration tests for compiler pipeline"""
    
    @pytest.mark.integration
    def test_end_to_end_compilation(self, photonic_mlp, mlp_input):
        """Test complete compilation pipeline"""
        compiler = PhotonicCompiler(
            backend=PhotonicBackend.SIMULATION_ONLY,
            wavelengths=[1550.0, 1551.0, 1552.0, 1553.0],
            power_budget=150.0
        )
        
        # Compile model
        circuit = compiler.compile(photonic_mlp, mlp_input, optimization_level=3)
        
        # Generate HLS
        hls_code = circuit.generate_hls("AIM_Photonics_PDK", "45nm_SOI")
        
        # Verify outputs
        assert isinstance(circuit, PhotonicCircuit)
        assert len(circuit.mlir_module) > 100  # Should be substantial MLIR code
        assert len(hls_code) > 200  # Should be substantial HLS code
        assert "wavelength_channels = 4" in circuit.mlir_module
        assert "power_gated = true" in circuit.mlir_module  # Level 3 optimization
    
    @pytest.mark.integration
    def test_compilation_with_optimization(self, compiler, simple_model, example_input):
        """Test compilation with optimization pipeline"""
        from photonic_mlir.optimization import OptimizationPipeline, PhotonicPasses
        
        # Compile circuit
        circuit = compiler.compile(simple_model, example_input, optimization_level=2)
        
        # Apply additional optimization
        pipeline = OptimizationPipeline()
        pipeline.add_pass(PhotonicPasses.WavelengthAllocation(channels=2))
        pipeline.add_pass(PhotonicPasses.ThermalAwarePlacement())
        
        optimized_circuit = pipeline.run(circuit)
        
        # Verify optimization applied
        assert optimized_circuit.mlir_module != circuit.mlir_module
        assert hasattr(optimized_circuit, '_optimization_report')
    
    @pytest.mark.performance
    def test_compilation_performance(self, large_model, performance_input):
        """Test compilation performance with larger models"""
        import time
        
        compiler = PhotonicCompiler()
        
        start_time = time.time()
        circuit = compiler.compile(large_model, performance_input[:1])  # Single batch
        compilation_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert compilation_time < 10.0  # seconds
        assert isinstance(circuit, PhotonicCircuit)
        
        # Test HLS generation performance
        start_time = time.time()
        hls_code = circuit.generate_hls()
        hls_time = time.time() - start_time
        
        assert hls_time < 5.0  # seconds
        assert len(hls_code) > 0
    
    @pytest.mark.integration
    def test_multi_model_compilation(self, temp_dir):
        """Test compiling multiple models in sequence"""
        models = [
            nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)),
            nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3)),
            nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1))
        ]
        
        inputs = [
            torch.randn(1, 10),
            torch.randn(1, 20),
            torch.randn(1, 5)
        ]
        
        compiler = PhotonicCompiler()
        circuits = []
        
        for model, input_tensor in zip(models, inputs):
            model.eval()
            circuit = compiler.compile(model, input_tensor)
            circuits.append(circuit)
        
        # Verify all compilations succeeded
        assert len(circuits) == 3
        for i, circuit in enumerate(circuits):
            assert isinstance(circuit, PhotonicCircuit)
            
            # Save each circuit
            circuit.save_netlist(f"{temp_dir}/model_{i}_netlist.sp")
            hls_code = circuit.generate_hls()
            assert len(hls_code) > 0


class TestCompilerErrorHandling:
    """Test error handling in compiler"""
    
    def test_compilation_error_handling(self):
        """Test handling of compilation errors"""
        compiler = PhotonicCompiler()
        
        # Test with invalid model type
        with pytest.raises(ValidationError) as exc_info:
            compiler.compile(None, torch.randn(1, 10))
        
        assert "Model must be a PyTorch nn.Module" in str(exc_info.value) or "Security validation failed" in str(exc_info.value)
    
    def test_mlir_generation_error(self, compiler, simple_model):
        """Test error handling in MLIR generation"""
        # Test with invalid input shape
        model = nn.Linear(10, 5)
        model.eval()
        
        # Input shape mismatch  
        with pytest.raises((ValidationError, CompilationError, RuntimeError)):
            compiler.compile(model, torch.randn(1, 5))  # Wrong input size
    
    def test_hls_generation_error(self, compiler, simple_model, example_input):
        """Test error handling in HLS generation"""
        circuit = compiler.compile(simple_model, example_input)
        
        # Test with invalid target PDK
        try:
            hls_code = circuit.generate_hls(target="INVALID_PDK")
            # Should either work with fallback or raise appropriate error
            assert len(hls_code) > 0
        except Exception as e:
            # If it raises an error, it should be a well-defined one
            assert isinstance(e, (ValidationError, CompilationError))
    
    @pytest.mark.integration
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion during compilation"""
        # Create extremely large model to test resource limits
        try:
            large_model = nn.Sequential(
                *[nn.Linear(1000, 1000) for _ in range(100)]  # Very large model
            )
            large_model.eval()
            
            compiler = PhotonicCompiler()
            
            # This should either succeed with warnings or fail gracefully
            with pytest.raises((ValidationError, CompilationError, MemoryError)):
                compiler.compile(large_model, torch.randn(1, 1000))
                
        except Exception as e:
            # If we can't even create the model, that's fine for this test
            pytest.skip(f"Could not create large model for testing: {e}")