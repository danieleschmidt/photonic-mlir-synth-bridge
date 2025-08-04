"""
Input validation and sanitization for Photonic MLIR compiler.
"""

from typing import List, Dict, Any, Union, Optional
import re

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    # Mock when torch is not available
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    np = None

from .exceptions import ValidationError, PowerBudgetExceededError, WavelengthConflictError


class InputValidator:
    """Comprehensive input validation for photonic compilation"""
    
    # Valid wavelength ranges for different PDKs (in nm)
    VALID_WAVELENGTH_RANGES = {
        "AIM_Photonics_PDK": (1500, 1600),
        "IMEC_SiPhotonics": (1520, 1580),
        "GlobalFoundries": (1530, 1570),
    }
    
    # Maximum supported tensor dimensions
    MAX_TENSOR_DIMS = 6
    MAX_TENSOR_SIZE = 1e9  # 1B elements
    
    # Power consumption limits (mW)
    MIN_POWER_BUDGET = 1.0
    MAX_POWER_BUDGET = 1000.0
    
    @staticmethod
    def validate_model(model) -> Dict[str, Any]:
        """Validate PyTorch model for photonic compilation"""
        if not TORCH_AVAILABLE:
            raise ValidationError("PyTorch not available for model validation")
            
        if not isinstance(model, nn.Module):
            raise ValidationError(
                "Model must be a PyTorch nn.Module",
                parameter="model",
                expected="nn.Module",
                got=type(model).__name__
            )
        
        # Check if model is in eval mode for compilation
        if model.training:
            raise ValidationError(
                "Model must be in eval mode for compilation. Call model.eval()"
            )
        
        # Analyze model complexity
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1e9:  # 1B parameters
            raise ValidationError(
                f"Model too large for photonic implementation: {total_params:,} parameters"
            )
        
        # Check for unsupported operations
        unsupported_ops = InputValidator._find_unsupported_operations(model)
        if unsupported_ops:
            raise ValidationError(
                f"Model contains unsupported operations: {unsupported_ops}"
            )
        
        return {
            "total_parameters": total_params,
            "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
            "supported": True
        }
    
    @staticmethod
    def _find_unsupported_operations(model) -> List[str]:
        """Find operations not supported by photonic backend"""
        unsupported = []
        
        for name, module in model.named_modules():
            # Check for operations that are difficult to implement photonically
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                unsupported.append(f"Recurrent layer: {name}")
            elif isinstance(module, nn.MultiheadAttention):
                unsupported.append(f"Multi-head attention: {name}")
            elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                unsupported.append(f"Instance normalization: {name}")
            # Add more unsupported operations as needed
        
        return unsupported
    
    @staticmethod
    def validate_input_tensor(tensor, name: str = "input") -> Dict[str, Any]:
        """Validate input tensor for photonic processing"""
        if not TORCH_AVAILABLE:
            raise ValidationError("PyTorch not available for tensor validation")
            
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                f"{name} must be a torch.Tensor",
                parameter=name,
                expected="torch.Tensor",
                got=type(tensor).__name__
            )
        
        # Check tensor dimensions
        if tensor.dim() > InputValidator.MAX_TENSOR_DIMS:
            raise ValidationError(
                f"{name} has too many dimensions: {tensor.dim()} > {InputValidator.MAX_TENSOR_DIMS}"
            )
        
        # Check tensor size
        if tensor.numel() > InputValidator.MAX_TENSOR_SIZE:
            raise ValidationError(
                f"{name} is too large: {tensor.numel():,} elements > {InputValidator.MAX_TENSOR_SIZE:,}"
            )
        
        # Check for invalid values
        if torch.isnan(tensor).any():
            raise ValidationError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValidationError(f"{name} contains infinite values")
        
        # Check data type
        if tensor.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            raise ValidationError(
                f"{name} has unsupported dtype: {tensor.dtype}",
                parameter=f"{name}.dtype",
                expected="float32, float64, complex64, or complex128",
                got=str(tensor.dtype)
            )
        
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
            "memory_mb": tensor.numel() * tensor.element_size() / (1024 ** 2)
        }
    
    @staticmethod
    def validate_wavelengths(wavelengths: List[float], pdk: str = "AIM_Photonics_PDK") -> Dict[str, Any]:
        """Validate wavelength specifications"""
        # Check type - handle case where numpy is not available
        valid_types = (list, tuple)
        if TORCH_AVAILABLE and np is not None:
            valid_types = (list, tuple, np.ndarray)
        
        if not isinstance(wavelengths, valid_types):
            raise ValidationError(
                "Wavelengths must be a list, tuple, or numpy array",
                parameter="wavelengths",
                expected="list/tuple/ndarray",
                got=type(wavelengths).__name__
            )
        
        wavelengths = list(wavelengths)
        
        if len(wavelengths) == 0:
            raise ValidationError("At least one wavelength must be specified")
        
        if len(wavelengths) > 32:  # Practical limit for WDM
            raise ValidationError(
                f"Too many wavelengths: {len(wavelengths)} > 32"
            )
        
        # Check wavelength values
        if pdk in InputValidator.VALID_WAVELENGTH_RANGES:
            min_wl, max_wl = InputValidator.VALID_WAVELENGTH_RANGES[pdk]
            for i, wl in enumerate(wavelengths):
                if not isinstance(wl, (int, float)):
                    raise ValidationError(
                        f"Wavelength[{i}] must be numeric, got {type(wl).__name__}"
                    )
                
                if not (min_wl <= wl <= max_wl):
                    raise ValidationError(
                        f"Wavelength[{i}] = {wl}nm outside valid range [{min_wl}, {max_wl}]nm for {pdk}"
                    )
        
        # Check for duplicates
        if len(set(wavelengths)) != len(wavelengths):
            duplicates = [wl for wl in set(wavelengths) if wavelengths.count(wl) > 1]
            raise WavelengthConflictError(duplicates)
        
        # Check minimum spacing (avoid crosstalk)
        sorted_wl = sorted(wavelengths)
        min_spacing = 0.8  # nm (typical for DWDM)
        for i in range(1, len(sorted_wl)):
            spacing = sorted_wl[i] - sorted_wl[i-1]
            if spacing < min_spacing:
                raise ValidationError(
                    f"Wavelength spacing too small: {spacing:.2f}nm < {min_spacing}nm "
                    f"between {sorted_wl[i-1]}nm and {sorted_wl[i]}nm"
                )
        
        return {
            "count": len(wavelengths),
            "channel_count": len(wavelengths),  # Alias for compatibility
            "range": (min(wavelengths), max(wavelengths)),
            "spacing": [sorted_wl[i] - sorted_wl[i-1] for i in range(1, len(sorted_wl))],
            "valid": True
        }
    
    @staticmethod
    def validate_power_budget(power_budget: float) -> Dict[str, Any]:
        """Validate power budget specification"""
        if not isinstance(power_budget, (int, float)):
            raise ValidationError(
                "Power budget must be numeric",
                parameter="power_budget",
                expected="int/float",
                got=type(power_budget).__name__
            )
        
        if power_budget <= 0:
            raise ValidationError(
                f"Power budget must be positive, got {power_budget}"
            )
        
        if power_budget < InputValidator.MIN_POWER_BUDGET:
            raise ValidationError(
                f"Power budget too low: {power_budget}mW < {InputValidator.MIN_POWER_BUDGET}mW"
            )
        
        if power_budget > InputValidator.MAX_POWER_BUDGET:
            raise ValidationError(
                f"Power budget too high: {power_budget}mW > {InputValidator.MAX_POWER_BUDGET}mW"
            )
        
        return {
            "value": power_budget,
            "unit": "mW",
            "valid": True
        }
    
    @staticmethod
    def validate_optimization_level(level: int) -> Dict[str, Any]:
        """Validate optimization level"""
        if not isinstance(level, int):
            raise ValidationError(
                "Optimization level must be an integer",
                parameter="optimization_level",
                expected="int",
                got=type(level).__name__
            )
        
        if not (0 <= level <= 3):
            raise ValidationError(
                f"Optimization level must be 0-3, got {level}"
            )
        
        return {
            "level": level,
            "description": {
                0: "No optimization",
                1: "Basic wavelength allocation",
                2: "Thermal optimization + wavelength allocation", 
                3: "Full optimization (thermal + power gating + noise reduction)"
            }[level]
        }
    
    @staticmethod
    def sanitize_string_input(input_str: str, max_length: int = 256, 
                            allowed_chars: str = None) -> str:
        """Sanitize string inputs to prevent injection attacks"""
        if not isinstance(input_str, str):
            raise ValidationError(
                "Input must be a string",
                expected="str",
                got=type(input_str).__name__
            )
        
        if len(input_str) > max_length:
            raise ValidationError(
                f"String too long: {len(input_str)} > {max_length} characters"
            )
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        
        # If allowed_chars specified, filter to only those
        if allowed_chars:
            sanitized = ''.join(c for c in sanitized if c in allowed_chars)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_path(file_path: str) -> str:
        """Validate and sanitize file paths"""
        if not isinstance(file_path, str):
            raise ValidationError(
                "File path must be a string",
                parameter="file_path",
                expected="str",
                got=type(file_path).__name__
            )
        
        # Security: prevent path traversal
        if '..' in file_path or file_path.startswith('/'):
            raise ValidationError(
                "Invalid file path: path traversal not allowed"
            )
        
        # Sanitize path
        sanitized_path = InputValidator.sanitize_string_input(
            file_path, 
            max_length=512,
            allowed_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        )
        
        return sanitized_path


class CircuitValidator:
    """Validate compiled photonic circuits"""
    
    @staticmethod
    def validate_power_consumption(circuit_config: Dict[str, Any]) -> None:
        """Validate that circuit doesn't exceed power budget"""
        estimated_power = CircuitValidator._estimate_circuit_power(circuit_config)
        budget = circuit_config.get("power_budget", 100.0)
        
        if estimated_power > budget:
            raise PowerBudgetExceededError(estimated_power, budget)
    
    @staticmethod
    def _estimate_circuit_power(config: Dict[str, Any]) -> float:
        """Rough power estimation for validation"""
        base_power = 10.0  # mW
        wavelength_count = len(config.get("wavelengths", [1550]))
        
        # Estimate based on complexity (simplified)
        power_per_wavelength = 2.0  # mW per wavelength channel
        estimated = base_power + wavelength_count * power_per_wavelength
        
        return estimated