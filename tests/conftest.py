"""
Pytest configuration and fixtures for Photonic MLIR tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Import photonic_mlir modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from photonic_mlir.compiler import PhotonicCompiler, PhotonicBackend
from photonic_mlir.pytorch_frontend import PhotonicMLP, PhotonicCNN
from photonic_mlir.simulation import PhotonicSimulator
from photonic_mlir.optimization import OptimizationPipeline


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="photonic_mlir_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing"""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    model.eval()
    return model


@pytest.fixture
def photonic_mlp():
    """Create a photonic MLP model for testing"""
    model = PhotonicMLP(
        input_size=784,
        hidden_sizes=[256, 128],
        num_classes=10,
        wavelengths=[1550.0, 1551.0, 1552.0, 1553.0]
    )
    model.eval()
    return model


@pytest.fixture
def photonic_cnn():
    """Create a photonic CNN model for testing"""
    model = PhotonicCNN(
        num_classes=10,
        wavelengths=[1550.0, 1551.0, 1552.0, 1553.0]
    )
    model.eval()
    return model


@pytest.fixture
def example_input():
    """Create example input tensor"""
    return torch.randn(1, 10)


@pytest.fixture
def mnist_input():
    """Create MNIST-like input tensor"""
    return torch.randn(1, 1, 28, 28)


@pytest.fixture
def mlp_input():
    """Create MLP input tensor"""
    return torch.randn(1, 784)


@pytest.fixture
def compiler():
    """Create photonic compiler instance"""
    return PhotonicCompiler(
        backend=PhotonicBackend.SIMULATION_ONLY,
        wavelengths=[1550.0, 1551.0],
        power_budget=100.0
    )


@pytest.fixture
def simulator():
    """Create photonic simulator instance"""
    return PhotonicSimulator(
        pdk="AIM_Photonics_45nm",
        temperature=300.0,
        include_noise=True,
        monte_carlo_runs=10  # Reduced for faster tests
    )


@pytest.fixture
def optimization_pipeline():
    """Create optimization pipeline for testing"""
    from photonic_mlir.optimization import PhotonicPasses
    
    pipeline = OptimizationPipeline()
    pipeline.add_pass(PhotonicPasses.WavelengthAllocation(channels=4))
    pipeline.add_pass(PhotonicPasses.ThermalAwarePlacement(max_temp=350.0))
    pipeline.add_pass(PhotonicPasses.PhaseQuantization(bits=8))
    
    return pipeline


@pytest.fixture
def test_wavelengths():
    """Standard test wavelengths"""
    return [1550.0, 1551.0, 1552.0, 1553.0]


@pytest.fixture
def test_config():
    """Standard test configuration"""
    return {
        "backend": PhotonicBackend.SIMULATION_ONLY,
        "wavelengths": [1550.0, 1551.0],
        "power_budget": 50.0,
        "optimization_level": 2
    }


@pytest.fixture
def large_model():
    """Create a larger model for performance testing"""
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.eval()
    return model


@pytest.fixture
def performance_input():
    """Create large input for performance testing"""
    return torch.randn(100, 1000)


# Test data fixtures
@pytest.fixture
def test_matrices():
    """Generate test matrices for photonic operations"""
    return {
        "small": torch.randn(4, 4, dtype=torch.complex64),
        "medium": torch.randn(16, 16, dtype=torch.complex64),
        "large": torch.randn(64, 64, dtype=torch.complex64)
    }


@pytest.fixture
def noise_parameters():
    """Parameters for noise testing"""
    return {
        "thermal_noise_power": 1e-12,  # W
        "shot_noise_current": 1e-9,    # A
        "phase_noise_std": 0.01,       # radians
        "amplitude_noise_std": 0.001   # relative
    }


# Mock hardware fixture
@pytest.fixture
def mock_hardware():
    """Mock hardware interface for testing"""
    class MockHardware:
        def __init__(self):
            self.connected = False
            self.calibrated = False
            
        def connect(self):
            self.connected = True
            return True
            
        def calibrate(self):
            self.calibrated = True
            return True
            
        def execute(self, circuit, inputs):
            # Simulate hardware execution
            return {
                "power_consumption": 45.0,
                "latency": 2.1,
                "snr": 23.5,
                "ber": 1e-6
            }
    
    return MockHardware()


# Parametrized test fixtures
@pytest.fixture(params=[
    PhotonicBackend.SIMULATION_ONLY,
    PhotonicBackend.LIGHTMATTER,
    PhotonicBackend.ANALOG_PHOTONICS
])
def backend(request):
    """Parametrized backend fixture"""
    return request.param


@pytest.fixture(params=[1, 2, 3])
def optimization_level(request):
    """Parametrized optimization level fixture"""
    return request.param


@pytest.fixture(params=[
    [1550.0],
    [1550.0, 1551.0],
    [1550.0, 1551.0, 1552.0, 1553.0]
])
def wavelength_configs(request):
    """Parametrized wavelength configurations"""
    return request.param


# Performance markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests as hardware-dependent"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'performance', 'hardware', 'slow', 'security'] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if 'performance' in item.name or 'large' in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark hardware tests
        if 'hardware' in item.name or 'device' in item.name:
            item.add_marker(pytest.mark.hardware)


# Skip conditions
def pytest_runtest_setup(item):
    """Setup conditions for test execution"""
    # Skip hardware tests if hardware not available
    if item.get_closest_marker("hardware"):
        # Check if actual hardware is available
        # For now, always skip hardware tests in CI
        if os.getenv("CI"):
            pytest.skip("Hardware tests skipped in CI environment")
    
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow"):
        if not item.config.getoption("--runslow"):
            pytest.skip("Slow tests skipped (use --runslow to run)")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runhardware", action="store_true", default=False,
        help="run hardware-dependent tests"
    )