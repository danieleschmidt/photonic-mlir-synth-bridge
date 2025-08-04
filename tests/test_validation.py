"""
Tests for validation and security modules.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from photonic_mlir.validation import InputValidator, CircuitValidator
from photonic_mlir.security import SecurityConfig, SecureFileHandler, ModelValidator
from photonic_mlir.exceptions import ValidationError, PowerBudgetExceededError, WavelengthConflictError


class TestInputValidator:
    """Test InputValidator functionality"""
    
    def test_model_validation_success(self, simple_model):
        """Test successful model validation"""
        result = InputValidator.validate_model(simple_model)
        
        assert result["supported"] is True
        assert result["total_parameters"] > 0
        assert result["model_size_mb"] > 0
    
    def test_model_validation_training_mode(self):
        """Test model validation fails for training mode"""
        model = nn.Linear(10, 5)
        model.train()  # Should be in eval mode
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_model(model)
        
        assert "eval mode" in str(exc_info.value)
    
    def test_model_validation_too_large(self):
        """Test model validation fails for oversized models"""
        # Mock a very large model by patching parameter count
        model = nn.Linear(10, 5)
        model.eval()
        
        # Temporarily patch parameter count
        original_numel = torch.Tensor.numel
        torch.Tensor.numel = lambda self: 2e9  # 2 billion parameters
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                InputValidator.validate_model(model)
            assert "too large" in str(exc_info.value)
        finally:
            torch.Tensor.numel = original_numel
    
    def test_input_tensor_validation_success(self, example_input):
        """Test successful input tensor validation"""
        result = InputValidator.validate_input_tensor(example_input)
        
        assert result["shape"] == list(example_input.shape)
        assert result["numel"] == example_input.numel()
        assert result["memory_mb"] > 0
    
    def test_input_tensor_validation_non_tensor(self):
        """Test input validation fails for non-tensors"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_input_tensor("not a tensor")
        
        assert "torch.Tensor" in str(exc_info.value)
    
    def test_input_tensor_validation_nan_values(self):
        """Test input validation detects NaN values"""
        invalid_tensor = torch.tensor([[1.0, float('nan'), 3.0]])
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_input_tensor(invalid_tensor)
        
        assert "NaN" in str(exc_info.value)
    
    def test_input_tensor_validation_inf_values(self):
        """Test input validation detects infinite values"""
        invalid_tensor = torch.tensor([[1.0, float('inf'), 3.0]])
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_input_tensor(invalid_tensor)
        
        assert "infinite" in str(exc_info.value)
    
    def test_input_tensor_validation_too_many_dims(self):
        """Test input validation fails for tensors with too many dimensions"""
        # Create tensor with more than MAX_TENSOR_DIMS dimensions
        shape = [2] * (InputValidator.MAX_TENSOR_DIMS + 1)
        large_tensor = torch.zeros(shape)
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_input_tensor(large_tensor)
        
        assert "too many dimensions" in str(exc_info.value)
    
    def test_wavelength_validation_success(self, test_wavelengths):
        """Test successful wavelength validation"""
        result = InputValidator.validate_wavelengths(test_wavelengths)
        
        assert result["valid"] is True
        assert result["count"] == len(test_wavelengths)
        assert result["range"][0] == min(test_wavelengths)
        assert result["range"][1] == max(test_wavelengths)
    
    def test_wavelength_validation_duplicates(self):
        """Test wavelength validation detects duplicates"""
        wavelengths = [1550.0, 1551.0, 1550.0]  # Duplicate
        
        with pytest.raises(WavelengthConflictError) as exc_info:
            InputValidator.validate_wavelengths(wavelengths)
        
        assert 1550.0 in exc_info.value.conflicting_wavelengths
    
    def test_wavelength_validation_out_of_range(self):
        """Test wavelength validation for out-of-range values"""
        wavelengths = [1400.0, 1550.0]  # 1400nm is outside typical range
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_wavelengths(wavelengths, pdk="AIM_Photonics_PDK")
        
        assert "outside valid range" in str(exc_info.value)
    
    def test_wavelength_validation_too_close(self):
        """Test wavelength validation for insufficient spacing"""
        wavelengths = [1550.0, 1550.1]  # Too close together
        
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_wavelengths(wavelengths)
        
        assert "spacing too small" in str(exc_info.value)
    
    def test_power_budget_validation_success(self):
        """Test successful power budget validation"""
        result = InputValidator.validate_power_budget(100.0)
        
        assert result["valid"] is True
        assert result["value"] == 100.0
        assert result["unit"] == "mW"
    
    def test_power_budget_validation_negative(self):
        """Test power budget validation fails for negative values"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_power_budget(-10.0)
        
        assert "positive" in str(exc_info.value)
    
    def test_power_budget_validation_too_high(self):
        """Test power budget validation fails for excessive values"""
        with pytest.raises(ValidationError) as exc_info:
            InputValidator.validate_power_budget(2000.0)  # Above MAX_POWER_BUDGET
        
        assert "too high" in str(exc_info.value)
    
    def test_optimization_level_validation_success(self):
        """Test successful optimization level validation"""
        for level in [0, 1, 2, 3]:
            result = InputValidator.validate_optimization_level(level)
            assert result["level"] == level
            assert "description" in result
    
    def test_optimization_level_validation_invalid(self):
        """Test optimization level validation fails for invalid levels"""
        with pytest.raises(ValidationError):
            InputValidator.validate_optimization_level(-1)
        
        with pytest.raises(ValidationError):
            InputValidator.validate_optimization_level(4)
    
    def test_string_sanitization(self):
        """Test string input sanitization"""
        # Test normal string
        clean = InputValidator.sanitize_string_input("hello_world")
        assert clean == "hello_world"
        
        # Test string with dangerous characters
        dirty = "hello<script>alert('xss')</script>world"
        clean = InputValidator.sanitize_string_input(dirty)
        assert "<script>" not in clean
        assert "alert" not in clean
        
        # Test with allowed characters filter
        mixed = "hello123!@#world"
        clean = InputValidator.sanitize_string_input(
            mixed, 
            allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789"
        )
        assert "!" not in clean
        assert "@" not in clean
    
    def test_file_path_validation(self):
        """Test file path validation and sanitization"""
        # Valid path
        valid_path = "output/results.json"
        clean = InputValidator.validate_file_path(valid_path)
        assert clean == valid_path
        
        # Path traversal attempt
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path("../../../etc/passwd")
        
        # Absolute path (not allowed)
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path("/etc/passwd")


class TestCircuitValidator:
    """Test CircuitValidator functionality"""
    
    def test_power_consumption_validation_success(self):
        """Test successful power consumption validation"""
        config = {
            "power_budget": 100.0,
            "wavelengths": [1550.0, 1551.0]
        }
        
        # Should not raise exception
        CircuitValidator.validate_power_consumption(config)
    
    def test_power_consumption_validation_exceeded(self):
        """Test power consumption validation when budget exceeded"""
        config = {
            "power_budget": 5.0,  # Very low budget
            "wavelengths": [1550.0, 1551.0, 1552.0, 1553.0]  # Many wavelengths
        }
        
        with pytest.raises(PowerBudgetExceededError) as exc_info:
            CircuitValidator.validate_power_consumption(config)
        
        assert exc_info.value.budget == 5.0
        assert exc_info.value.estimated_power > 5.0


class TestSecurityConfig:
    """Test SecurityConfig functionality"""
    
    def test_default_config(self):
        """Test default security configuration"""
        config = SecurityConfig()
        
        assert config.allowed_file_extensions
        assert config.max_file_size_mb > 0
        assert config.max_model_size_mb > 0
        assert config.sandbox_mode is True
    
    def test_config_modification(self):
        """Test security configuration modification"""
        config = SecurityConfig()
        
        original_size = config.max_file_size_mb
        config.max_file_size_mb = 200
        
        assert config.max_file_size_mb == 200
        assert config.max_file_size_mb != original_size


class TestSecureFileHandler:
    """Test SecureFileHandler functionality"""
    
    def test_file_path_validation_success(self, temp_dir):
        """Test successful file path validation"""
        handler = SecureFileHandler()
        
        # Valid file path
        valid_path = "output/test.json"
        validated = handler.validate_file_path(valid_path)
        
        assert validated is not None
        assert isinstance(validated, str)
    
    def test_file_path_validation_traversal(self):
        """Test file path validation prevents path traversal"""
        handler = SecureFileHandler()
        
        with pytest.raises(ValidationError):
            handler.validate_file_path("../../../etc/passwd")
    
    def test_file_path_validation_system_dirs(self):
        """Test file path validation prevents access to system directories"""
        handler = SecureFileHandler()
        
        with pytest.raises(ValidationError):
            handler.validate_file_path("/etc/passwd")
        
        with pytest.raises(ValidationError):
            handler.validate_file_path("/sys/kernel/version")
    
    def test_file_extension_validation(self):
        """Test file extension validation"""
        handler = SecureFileHandler()
        
        # Valid extension
        handler.validate_file_path("test.py")
        
        # Invalid extension
        with pytest.raises(ValidationError):
            handler.validate_file_path("test.exe")
    
    def test_secure_temp_dir_creation(self):
        """Test secure temporary directory creation"""
        handler = SecureFileHandler()
        
        temp_dir = handler.create_secure_temp_dir()
        
        assert temp_dir is not None
        assert isinstance(temp_dir, str)
        import os
        assert os.path.exists(temp_dir)
        
        # Cleanup
        handler.cleanup_temp_dirs()
    
    def test_secure_write_disabled(self):
        """Test secure write when file writing is disabled"""
        config = SecurityConfig()
        config.allow_file_write = False
        
        handler = SecureFileHandler(config)
        
        with pytest.raises(ValidationError):
            handler.secure_write("test.txt", "content")


class TestModelValidator:
    """Test ModelValidator functionality"""
    
    def test_model_security_validation_success(self, simple_model):
        """Test successful model security validation"""
        validator = ModelValidator()
        
        model_info = {"model_size_mb": 1.0, "total_parameters": 100}
        result = validator.validate_model_security(simple_model, model_info)
        
        assert result["secure"] is True
        assert result["model_hash"] is not None
        assert len(result["warnings"]) >= 0
        assert len(result["blocked_operations"]) == 0
    
    def test_model_security_validation_too_large(self, simple_model):
        """Test model security validation for oversized models"""
        validator = ModelValidator()
        
        # Mock large model
        model_info = {"model_size_mb": 2000.0, "total_parameters": 2e9}
        result = validator.validate_model_security(simple_model, model_info)
        
        assert result["secure"] is False
        assert any("too large" in op for op in result["blocked_operations"])
    
    def test_model_hash_generation(self, simple_model):
        """Test model hash generation for integrity checking"""
        validator = ModelValidator()
        
        hash1 = validator._generate_model_hash(simple_model)
        hash2 = validator._generate_model_hash(simple_model)
        
        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different model should have different hash
        different_model = nn.Linear(5, 3)
        hash3 = validator._generate_model_hash(different_model)
        assert hash1 != hash3


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.mark.security
    def test_end_to_end_secure_compilation(self, simple_model, example_input):
        """Test complete secure compilation pipeline"""
        from photonic_mlir.security import SecureCompilationEnvironment
        
        env = SecureCompilationEnvironment()
        
        # Create secure session
        session = env.create_compilation_session()
        assert session["session_id"] is not None
        assert session["temp_dir"] is not None
        
        # Validate inputs
        config = {"power_budget": 100.0, "wavelengths": [1550.0]}
        validation = env.validate_compilation_inputs(simple_model, example_input, config)
        
        assert validation["overall_secure"] is True
        assert validation["model_validation"]["secure"] is True
        
        # Cleanup
        env.cleanup_session()
    
    @pytest.mark.security
    def test_security_audit_logging(self):
        """Test security audit logging functionality"""
        from photonic_mlir.security import get_audit_logger
        
        audit_logger = get_audit_logger()
        
        # Test logging security events
        audit_logger.log_security_event(
            "test_event",
            {"test_detail": "test_value"},
            severity="INFO"
        )
        
        # Test logging access attempts
        audit_logger.log_access_attempt(
            "test_resource",
            "read",
            allowed=True,
            reason="valid_access"
        )
        
        audit_logger.log_access_attempt(
            "restricted_resource", 
            "write",
            allowed=False,
            reason="insufficient_permissions"
        )
    
    @pytest.mark.security
    def test_secure_hardware_interface(self):
        """Test secure hardware interface"""
        from photonic_mlir.security import SecureHardwareInterface
        
        interface = SecureHardwareInterface()
        
        # Test hardware access validation
        assert interface.validate_hardware_access("simulator", "read") is True
        assert interface.validate_hardware_access("unknown_device", "read") is False
        
        # Test configuration sanitization
        unsafe_config = {
            "device": "simulator",
            "calibration_file": "../../../etc/passwd",  # Path traversal
            "power_limit": 2000.0,  # Too high
            "malicious_key": "malicious_value"  # Unknown key
        }
        
        safe_config = interface.sanitize_hardware_config(unsafe_config)
        
        assert "malicious_key" not in safe_config
        assert safe_config.get("power_limit", 0) <= 1000.0
        assert "calibration_file" not in safe_config  # Should be filtered out