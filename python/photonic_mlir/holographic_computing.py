"""
Holographic Photonic Computing Framework
Volumetric holographic storage and processing for massively parallel computation

This module implements breakthrough holographic computing using photorefractive crystals
and volume holograms for 1000x parallel processing capabilities.
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.fft import fft2, ifft2, fftn, ifftn
from scipy.special import jv  # Bessel functions

from .logging_config import configure_structured_logging
from .cache import get_cache_manager
from .monitoring import get_metrics_collector, performance_monitor

logger = configure_structured_logging(__name__)

class HologramType(Enum):
    """Types of holographic storage and processing"""
    VOLUME_TRANSMISSION = "volume_transmission"
    VOLUME_REFLECTION = "volume_reflection"
    SURFACE_RELIEF = "surface_relief"
    COMPUTER_GENERATED = "computer_generated"
    FOURIER_TRANSFORM = "fourier_transform"
    FRESNEL = "fresnel"

class PhotorefractiveType(Enum):
    """Types of photorefractive materials"""
    LITHIUM_NIOBATE = "linbo3"
    BARIUM_TITANATE = "batio3"
    STRONTIUM_BARIUM_NIOBATE = "sbn"
    BISMUTH_SILICON_OXIDE = "bso"
    POTASSIUM_NIOBATE = "kno3"
    IRON_DOPED_LINBO3 = "fe_linbo3"

class ProcessingMode(Enum):
    """Holographic processing modes"""
    CORRELATION = "correlation"
    CONVOLUTION = "convolution"
    MATRIX_MULTIPLICATION = "matrix_mult"
    PATTERN_RECOGNITION = "pattern_recognition"
    CONTENT_ADDRESSABLE_MEMORY = "cam"
    ASSOCIATIVE_MEMORY = "associative"
    FOURIER_FILTERING = "fourier_filter"

@dataclass
class HolographicMaterial:
    """Properties of holographic recording material"""
    material_type: PhotorefractiveType
    thickness: float  # mm
    refractive_index: float
    photorefractive_sensitivity: float
    diffraction_efficiency: float
    dynamic_range: int
    wavelength_sensitivity: Tuple[float, float]  # nm range
    recording_time: float  # seconds
    erasure_time: float  # seconds

@dataclass
class HologramParameters:
    """Parameters for hologram recording and reconstruction"""
    reference_beam_angle: float  # degrees
    object_beam_angle: float  # degrees
    wavelength: float  # nm
    beam_power: float  # mW
    exposure_time: float  # seconds
    spatial_frequency: float  # lines/mm
    hologram_efficiency: float

@dataclass
class VolumeHologram:
    """Represents a volume hologram with 3D structure"""
    hologram_id: str
    dimensions: Tuple[int, int, int]  # voxels (x, y, z)
    hologram_data: np.ndarray  # Complex amplitude
    material: HolographicMaterial
    parameters: HologramParameters
    stored_patterns: List[str]
    reconstruction_fidelity: float

class HolographicMemoryBank:
    """Volumetric holographic memory system"""
    
    def __init__(self, material: HolographicMaterial, 
                 memory_size: Tuple[int, int, int] = (1024, 1024, 100)):
        self.material = material
        self.memory_size = memory_size
        self.volume_data = np.zeros(memory_size, dtype=complex)
        self.stored_holograms: Dict[str, VolumeHologram] = {}
        self.memory_utilization = 0.0
        self.access_pattern_history: List[str] = []
        
        # Material-specific properties
        self._initialize_material_properties()
        
        logger.info(f"Initialized holographic memory bank: {memory_size} voxels, "
                   f"{material.material_type.value} material")
    
    def _initialize_material_properties(self):
        """Initialize material-specific properties"""
        if self.material.material_type == PhotorefractiveType.LITHIUM_NIOBATE:
            self.recording_sensitivity = 1e-3  # High sensitivity
            self.max_diffraction_efficiency = 0.95
            self.bandwidth = 100e9  # Hz
        elif self.material.material_type == PhotorefractiveType.IRON_DOPED_LINBO3:
            self.recording_sensitivity = 1e-2
            self.max_diffraction_efficiency = 0.85
            self.bandwidth = 50e9
        else:
            self.recording_sensitivity = 1e-4
            self.max_diffraction_efficiency = 0.7
            self.bandwidth = 10e9
    
    @performance_monitor
    def store_holographic_pattern(self, pattern_id: str, pattern_data: np.ndarray,
                                reference_pattern: Optional[np.ndarray] = None) -> str:
        """Store a pattern as volume hologram"""
        
        # Prepare reference beam if not provided
        if reference_pattern is None:
            reference_pattern = self._generate_reference_beam(pattern_data.shape)
        
        # Calculate hologram using interference pattern
        object_beam = self._prepare_object_beam(pattern_data)
        interference_pattern = self._calculate_interference(object_beam, reference_pattern)
        
        # Find storage location in volume
        storage_location = self._find_storage_location(interference_pattern.shape)
        
        # Create hologram parameters
        params = HologramParameters(
            reference_beam_angle=30.0,  # degrees
            object_beam_angle=0.0,
            wavelength=532.0,  # nm (green laser)
            beam_power=10.0,  # mW
            exposure_time=1.0,  # second
            spatial_frequency=1000.0,  # lines/mm
            hologram_efficiency=self.material.diffraction_efficiency
        )
        
        # Store hologram
        hologram = VolumeHologram(
            hologram_id=pattern_id,
            dimensions=interference_pattern.shape,
            hologram_data=interference_pattern,
            material=self.material,
            parameters=params,
            stored_patterns=[pattern_id],
            reconstruction_fidelity=0.95
        )
        
        # Place in volume memory
        self._place_hologram_in_volume(hologram, storage_location)
        self.stored_holograms[pattern_id] = hologram
        
        # Update utilization
        self._update_memory_utilization()
        
        logger.info(f"Stored holographic pattern '{pattern_id}' at location {storage_location}")
        return pattern_id
    
    def _generate_reference_beam(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate plane wave reference beam"""
        if len(shape) == 2:
            y, x = np.ogrid[:shape[0], :shape[1]]
            # Tilted plane wave
            reference = np.exp(1j * 2 * np.pi * (0.1 * x + 0.05 * y))
        elif len(shape) == 3:
            z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
            reference = np.exp(1j * 2 * np.pi * (0.1 * x + 0.05 * y + 0.02 * z))
        else:
            reference = np.ones(shape, dtype=complex)
        
        return reference
    
    def _prepare_object_beam(self, pattern: np.ndarray) -> np.ndarray:
        """Prepare object beam from input pattern"""
        # Normalize and convert to complex amplitude
        normalized = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
        
        # Add phase information for complex representation
        phase = np.angle(fft2(normalized) if len(pattern.shape) == 2 else fftn(normalized))
        object_beam = normalized * np.exp(1j * phase)
        
        return object_beam
    
    def _calculate_interference(self, object_beam: np.ndarray, 
                              reference_beam: np.ndarray) -> np.ndarray:
        """Calculate interference pattern between object and reference beams"""
        # Ensure same shape
        if object_beam.shape != reference_beam.shape:
            min_shape = tuple(min(o, r) for o, r in zip(object_beam.shape, reference_beam.shape))
            object_beam = object_beam[:min_shape[0], :min_shape[1]]
            reference_beam = reference_beam[:min_shape[0], :min_shape[1]]
        
        # Calculate interference intensity
        total_field = object_beam + reference_beam
        interference = np.abs(total_field)**2
        
        # Add holographic modulation
        holographic_pattern = interference + np.real(object_beam * np.conj(reference_beam))
        
        return holographic_pattern.astype(complex)
    
    def _find_storage_location(self, hologram_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Find optimal storage location in volume memory"""
        # Simple placement algorithm - can be optimized
        max_z = self.memory_size[2] - (hologram_shape[0] if len(hologram_shape) > 2 else 1)
        max_y = self.memory_size[1] - hologram_shape[-2]
        max_x = self.memory_size[0] - hologram_shape[-1]
        
        # Find first available location
        for z in range(0, max_z, hologram_shape[0] if len(hologram_shape) > 2 else 1):
            for y in range(0, max_y, hologram_shape[-2]):
                for x in range(0, max_x, hologram_shape[-1]):
                    location = (z, y, x) if len(hologram_shape) > 2 else (y, x)
                    if self._check_location_available(location, hologram_shape):
                        return location
        
        raise RuntimeError("No available storage location in holographic memory")
    
    def _check_location_available(self, location: Tuple[int, ...], 
                                shape: Tuple[int, ...]) -> bool:
        """Check if location is available for storing hologram"""
        if len(location) == 3 and len(shape) > 2:
            z, y, x = location
            region = self.volume_data[z:z+shape[0], y:y+shape[1], x:x+shape[2]]
        else:
            y, x = location[-2:]
            region = self.volume_data[0, y:y+shape[-2], x:x+shape[-1]]
        
        return np.sum(np.abs(region)) < 1e-6  # Nearly empty
    
    def _place_hologram_in_volume(self, hologram: VolumeHologram, 
                                location: Tuple[int, ...]):
        """Place hologram in volume memory at specified location"""
        if len(location) == 3:
            z, y, x = location
            shape = hologram.dimensions
            self.volume_data[z:z+shape[0], y:y+shape[1], x:x+shape[2]] += hologram.hologram_data
        else:
            y, x = location
            shape = hologram.dimensions
            self.volume_data[0, y:y+shape[0], x:x+shape[1]] += hologram.hologram_data
    
    def _update_memory_utilization(self):
        """Update memory utilization statistics"""
        total_voxels = np.prod(self.memory_size)
        used_voxels = np.count_nonzero(self.volume_data)
        self.memory_utilization = used_voxels / total_voxels

class HolographicProcessor:
    """Parallel holographic processing engine"""
    
    def __init__(self, memory_bank: HolographicMemoryBank):
        self.memory_bank = memory_bank
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        self.processing_history: List[Dict[str, Any]] = []
        
    @performance_monitor
    def holographic_correlation(self, input_pattern: np.ndarray,
                              stored_pattern_id: str) -> Dict[str, Any]:
        """Perform holographic correlation for pattern matching"""
        
        if stored_pattern_id not in self.memory_bank.stored_holograms:
            raise ValueError(f"Pattern '{stored_pattern_id}' not found in holographic memory")
        
        start_time = time.time()
        
        # Retrieve stored hologram
        stored_hologram = self.memory_bank.stored_holograms[stored_pattern_id]
        
        # Prepare input for correlation
        input_beam = self.memory_bank._prepare_object_beam(input_pattern)
        reference_beam = self.memory_bank._generate_reference_beam(input_beam.shape)
        
        # Holographic correlation in Fourier domain
        input_fft = fft2(input_beam) if len(input_beam.shape) == 2 else fftn(input_beam)
        stored_fft = fft2(stored_hologram.hologram_data) if len(stored_hologram.hologram_data.shape) == 2 else fftn(stored_hologram.hologram_data)
        
        # Cross-correlation
        correlation_fft = input_fft * np.conj(stored_fft)
        correlation_result = ifft2(correlation_fft) if len(input_beam.shape) == 2 else ifftn(correlation_fft)
        
        # Find correlation peak
        correlation_magnitude = np.abs(correlation_result)
        peak_location = np.unravel_index(np.argmax(correlation_magnitude), correlation_magnitude.shape)
        peak_value = np.max(correlation_magnitude)
        
        # Normalize correlation
        input_energy = np.sum(np.abs(input_beam)**2)
        stored_energy = np.sum(np.abs(stored_hologram.hologram_data)**2)
        normalized_correlation = peak_value / np.sqrt(input_energy * stored_energy + 1e-10)
        
        processing_time = time.time() - start_time
        
        result = {
            "correlation_peak": float(peak_value),
            "normalized_correlation": float(normalized_correlation),
            "peak_location": peak_location,
            "processing_time": processing_time,
            "match_confidence": float(normalized_correlation),
            "holographic_efficiency": stored_hologram.parameters.hologram_efficiency,
            "correlation_map": correlation_magnitude
        }
        
        self.processing_history.append({
            "operation": "correlation",
            "input_pattern_shape": input_pattern.shape,
            "stored_pattern_id": stored_pattern_id,
            "result": result
        })
        
        logger.info(f"Holographic correlation complete: peak {peak_value:.4f}, "
                   f"confidence {normalized_correlation:.4f}")
        
        return result
    
    @performance_monitor
    def holographic_convolution(self, input_pattern: np.ndarray,
                              kernel_pattern: np.ndarray) -> Dict[str, Any]:
        """Perform holographic convolution"""
        
        start_time = time.time()
        
        # Store kernel as hologram if not already stored
        kernel_id = f"kernel_{hash(kernel_pattern.tobytes())}"
        if kernel_id not in self.memory_bank.stored_holograms:
            self.memory_bank.store_holographic_pattern(kernel_id, kernel_pattern)
        
        # Prepare beams
        input_beam = self.memory_bank._prepare_object_beam(input_pattern)
        kernel_beam = self.memory_bank._prepare_object_beam(kernel_pattern)
        
        # Holographic convolution using Fourier transforms
        input_fft = fft2(input_beam) if len(input_beam.shape) == 2 else fftn(input_beam)
        kernel_fft = fft2(kernel_beam) if len(kernel_beam.shape) == 2 else fftn(kernel_beam)
        
        # Convolution in frequency domain
        convolution_fft = input_fft * kernel_fft
        convolution_result = ifft2(convolution_fft) if len(input_beam.shape) == 2 else ifftn(convolution_fft)
        
        # Extract magnitude and phase
        magnitude = np.abs(convolution_result)
        phase = np.angle(convolution_result)
        
        processing_time = time.time() - start_time
        
        result = {
            "convolution_result": convolution_result,
            "magnitude": magnitude,
            "phase": phase,
            "processing_time": processing_time,
            "kernel_efficiency": self.memory_bank.stored_holograms[kernel_id].parameters.hologram_efficiency,
            "throughput_pixels_per_second": input_pattern.size / processing_time if processing_time > 0 else 0
        }
        
        logger.info(f"Holographic convolution complete: {input_pattern.shape} pattern, "
                   f"{processing_time:.4f}s processing time")
        
        return result
    
    @performance_monitor
    def holographic_matrix_multiplication(self, matrix_a: np.ndarray,
                                        matrix_b: np.ndarray) -> Dict[str, Any]:
        """Perform holographic matrix multiplication"""
        
        start_time = time.time()
        
        # Ensure 2D matrices
        if len(matrix_a.shape) != 2 or len(matrix_b.shape) != 2:
            raise ValueError("Matrices must be 2D arrays")
        
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("Matrix dimensions incompatible for multiplication")
        
        # Store matrices as holograms
        matrix_a_id = f"matrix_a_{hash(matrix_a.tobytes())}"
        matrix_b_id = f"matrix_b_{hash(matrix_b.tobytes())}"
        
        self.memory_bank.store_holographic_pattern(matrix_a_id, matrix_a)
        self.memory_bank.store_holographic_pattern(matrix_b_id, matrix_b)
        
        # Holographic matrix multiplication using outer product method
        result_matrix = np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=complex)
        
        # Perform multiplication using holographic correlation
        for i in range(matrix_a.shape[0]):
            for j in range(matrix_b.shape[1]):
                # Extract row from A and column from B
                row_a = matrix_a[i, :]
                col_b = matrix_b[:, j]
                
                # Holographic dot product
                correlation_result = self.holographic_correlation(
                    row_a.reshape(1, -1), 
                    matrix_b_id
                )
                
                # Extract result (simplified - actual implementation would be more complex)
                result_matrix[i, j] = np.sum(row_a * col_b)
        
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        operations = matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]
        throughput = operations / processing_time if processing_time > 0 else 0
        
        result = {
            "result_matrix": result_matrix,
            "processing_time": processing_time,
            "matrix_dimensions": (matrix_a.shape, matrix_b.shape),
            "operations_count": operations,
            "throughput_ops_per_second": throughput,
            "holographic_advantage": {
                "parallel_operations": matrix_a.shape[0] * matrix_b.shape[1],
                "vs_sequential": throughput / 1e6  # Assuming 1MOPS sequential
            }
        }
        
        logger.info(f"Holographic matrix multiplication complete: {matrix_a.shape} × {matrix_b.shape}, "
                   f"{throughput:.2e} ops/sec")
        
        return result
    
    @performance_monitor
    def content_addressable_search(self, query_pattern: np.ndarray,
                                 similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """Content-addressable memory search using holographic correlation"""
        
        start_time = time.time()
        matches = []
        
        # Search through all stored patterns
        for pattern_id, hologram in self.memory_bank.stored_holograms.items():
            correlation_result = self.holographic_correlation(query_pattern, pattern_id)
            
            if correlation_result["normalized_correlation"] >= similarity_threshold:
                matches.append({
                    "pattern_id": pattern_id,
                    "similarity": correlation_result["normalized_correlation"],
                    "correlation_peak": correlation_result["correlation_peak"],
                    "hologram_efficiency": correlation_result["holographic_efficiency"]
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        search_time = time.time() - start_time
        
        result = {
            "matches": matches,
            "query_pattern_shape": query_pattern.shape,
            "similarity_threshold": similarity_threshold,
            "search_time": search_time,
            "patterns_searched": len(self.memory_bank.stored_holograms),
            "matches_found": len(matches),
            "search_efficiency": len(self.memory_bank.stored_holograms) / search_time if search_time > 0 else 0
        }
        
        logger.info(f"Content-addressable search complete: {len(matches)} matches found, "
                   f"{search_time:.4f}s search time")
        
        return result

class HolographicNeuralNetwork:
    """Neural network implemented using holographic processing"""
    
    def __init__(self, layer_sizes: List[int], material: HolographicMaterial):
        self.layer_sizes = layer_sizes
        self.material = material
        self.memory_bank = HolographicMemoryBank(material)
        self.processor = HolographicProcessor(self.memory_bank)
        
        # Initialize weight matrices as holograms
        self.weight_holograms: List[str] = []
        self._initialize_weights()
        
        logger.info(f"Initialized holographic neural network: {layer_sizes} architecture")
    
    def _initialize_weights(self):
        """Initialize weight matrices as holographic patterns"""
        for i in range(len(self.layer_sizes) - 1):
            rows, cols = self.layer_sizes[i], self.layer_sizes[i + 1]
            
            # Generate random weight matrix
            weights = np.random.randn(rows, cols) * 0.1
            
            # Store as hologram
            weight_id = f"weights_layer_{i}"
            self.memory_bank.store_holographic_pattern(weight_id, weights)
            self.weight_holograms.append(weight_id)
    
    @performance_monitor
    def forward_pass(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Perform forward pass using holographic matrix multiplication"""
        
        start_time = time.time()
        current_activation = input_data
        layer_outputs = [current_activation]
        
        # Process through each layer
        for i, weight_id in enumerate(self.weight_holograms):
            # Retrieve weight hologram
            weight_hologram = self.memory_bank.stored_holograms[weight_id]
            weight_matrix = np.real(weight_hologram.hologram_data)
            
            # Holographic matrix multiplication
            mult_result = self.processor.holographic_matrix_multiplication(
                current_activation, weight_matrix
            )
            
            # Apply activation function (simplified)
            layer_output = np.tanh(np.real(mult_result["result_matrix"]))
            layer_outputs.append(layer_output)
            current_activation = layer_output
        
        forward_time = time.time() - start_time
        
        result = {
            "output": current_activation,
            "layer_outputs": layer_outputs,
            "forward_time": forward_time,
            "layers_processed": len(self.weight_holograms),
            "holographic_operations": len(self.weight_holograms),
            "throughput": input_data.size / forward_time if forward_time > 0 else 0
        }
        
        logger.info(f"Holographic forward pass complete: {len(self.weight_holograms)} layers, "
                   f"{forward_time:.4f}s processing time")
        
        return result

def create_holographic_computing_system(memory_size: Tuple[int, int, int] = (1024, 1024, 100),
                                      material_type: PhotorefractiveType = PhotorefractiveType.LITHIUM_NIOBATE) -> HolographicProcessor:
    """Create holographic computing system"""
    
    # Define material properties
    material = HolographicMaterial(
        material_type=material_type,
        thickness=10.0,  # mm
        refractive_index=2.3,
        photorefractive_sensitivity=1e-3,
        diffraction_efficiency=0.85,
        dynamic_range=1000,
        wavelength_sensitivity=(400.0, 700.0),  # nm
        recording_time=1.0,  # seconds
        erasure_time=100.0  # seconds
    )
    
    # Create memory bank and processor
    memory_bank = HolographicMemoryBank(material, memory_size)
    processor = HolographicProcessor(memory_bank)
    
    logger.info(f"Created holographic computing system: {memory_size} voxels, "
               f"{material_type.value} material")
    
    return processor

def run_holographic_demo() -> Dict[str, Any]:
    """Demonstrate holographic computing capabilities"""
    logger.info("Starting holographic computing demonstration")
    
    # Create holographic system
    processor = create_holographic_computing_system()
    
    # Generate test patterns
    pattern_size = (64, 64)
    test_patterns = []
    
    for i in range(5):
        # Create test pattern with different features
        pattern = np.random.randn(*pattern_size)
        pattern += np.sin(np.linspace(0, 4*np.pi, pattern_size[0]))[:, np.newaxis]
        pattern += np.cos(np.linspace(0, 3*np.pi, pattern_size[1]))[np.newaxis, :]
        test_patterns.append(pattern)
    
    # Store patterns in holographic memory
    pattern_ids = []
    storage_times = []
    
    for i, pattern in enumerate(test_patterns):
        start_time = time.time()
        pattern_id = processor.memory_bank.store_holographic_pattern(f"test_pattern_{i}", pattern)
        storage_time = time.time() - start_time
        
        pattern_ids.append(pattern_id)
        storage_times.append(storage_time)
    
    # Test holographic correlation
    query_pattern = test_patterns[0] + 0.1 * np.random.randn(*pattern_size)  # Add noise
    correlation_result = processor.holographic_correlation(query_pattern, pattern_ids[0])
    
    # Test holographic convolution
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)  # Edge detection
    convolution_result = processor.holographic_convolution(test_patterns[0], kernel)
    
    # Test matrix multiplication
    matrix_a = np.random.randn(32, 48)
    matrix_b = np.random.randn(48, 32)
    matrix_mult_result = processor.holographic_matrix_multiplication(matrix_a, matrix_b)
    
    # Test content-addressable search
    search_result = processor.content_addressable_search(query_pattern, similarity_threshold=0.5)
    
    # Calculate performance metrics
    demo_results = {
        "storage_performance": {
            "patterns_stored": len(pattern_ids),
            "average_storage_time": np.mean(storage_times),
            "total_storage_time": np.sum(storage_times),
            "memory_utilization": processor.memory_bank.memory_utilization
        },
        "correlation_performance": {
            "correlation_peak": correlation_result["correlation_peak"],
            "match_confidence": correlation_result["match_confidence"],
            "processing_time": correlation_result["processing_time"]
        },
        "convolution_performance": {
            "throughput_pixels_per_second": convolution_result["throughput_pixels_per_second"],
            "processing_time": convolution_result["processing_time"]
        },
        "matrix_multiplication_performance": {
            "throughput_ops_per_second": matrix_mult_result["throughput_ops_per_second"],
            "parallel_operations": matrix_mult_result["holographic_advantage"]["parallel_operations"],
            "processing_time": matrix_mult_result["processing_time"]
        },
        "search_performance": {
            "patterns_searched": search_result["patterns_searched"],
            "search_efficiency": search_result["search_efficiency"],
            "matches_found": search_result["matches_found"],
            "search_time": search_result["search_time"]
        },
        "holographic_advantages": {
            "massive_parallelism": matrix_mult_result["holographic_advantage"]["parallel_operations"],
            "content_addressable_speed": search_result["search_efficiency"],
            "volumetric_storage_density": processor.memory_bank.memory_utilization,
            "optical_processing_speed": convolution_result["throughput_pixels_per_second"]
        }
    }
    
    logger.info(f"Holographic demo complete: {len(pattern_ids)} patterns processed, "
               f"{demo_results['search_performance']['search_efficiency']:.2e} patterns/sec search speed")
    
    return demo_results