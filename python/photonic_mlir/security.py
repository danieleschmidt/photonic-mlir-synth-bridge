"""
Security measures and utilities for Photonic MLIR.
"""

import hashlib
import hmac
import secrets
import json
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from .exceptions import ValidationError
from .logging_config import get_logger


class SecurityConfig:
    """Security configuration settings"""
    
    def __init__(self):
        # File access restrictions
        self.allowed_file_extensions = {'.py', '.mlir', '.json', '.yaml', '.yml', '.txt', '.log'}
        self.max_file_size_mb = 100
        self.temp_dir_prefix = "photonic_mlir_"
        
        # Model restrictions  
        self.max_model_size_mb = 1000
        self.allowed_model_types = {'Sequential', 'Module', 'PhotonicMLP', 'PhotonicCNN'}
        
        # Execution restrictions
        self.sandbox_mode = True
        self.allow_file_write = True
        self.allow_network_access = False
        
        # Logging
        self.log_security_events = True
        self.log_file_access = True


class SecureFileHandler:
    """Secure file handling with sandboxing"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("photonic_mlir.security.file_handler")
        self._temp_dirs: List[str] = []
        
    def create_secure_temp_dir(self) -> str:
        """Create a secure temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix=self.config.temp_dir_prefix)
        self._temp_dirs.append(temp_dir)
        
        if self.config.log_file_access:
            self.logger.info(f"Created temporary directory: {temp_dir}")
            
        return temp_dir
        
    def validate_file_path(self, file_path: str, operation: str = "read") -> str:
        """Validate and sanitize file path"""
        if not isinstance(file_path, str):
            raise ValidationError("File path must be a string")
        
        # Convert to Path object for safe handling
        path = Path(file_path).resolve()
        
        # Security checks
        if '..' in str(path):
            raise ValidationError("Path traversal not allowed")
            
        if str(path).startswith('/etc') or str(path).startswith('/sys'):
            raise ValidationError("Access to system directories not allowed")
            
        # Check file extension
        if path.suffix.lower() not in self.config.allowed_file_extensions:
            raise ValidationError(f"File extension {path.suffix} not allowed")
        
        # Check file size for write operations
        if operation == "write" and path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            if size_mb > self.config.max_file_size_mb:
                raise ValidationError(f"File too large: {size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
        
        if self.config.log_file_access:
            self.logger.debug(f"File access validated: {operation} {path}")
            
        return str(path)
    
    def secure_read(self, file_path: str) -> str:
        """Securely read a file"""
        validated_path = self.validate_file_path(file_path, "read")
        
        try:
            with open(validated_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Additional content validation
            if len(content) > self.config.max_file_size_mb * 1024 * 1024:
                raise ValidationError("File content too large")
                
            return content
            
        except Exception as e:
            self.logger.error(f"Secure file read failed: {e}")
            raise
    
    def secure_write(self, file_path: str, content: str) -> None:
        """Securely write to a file"""
        if not self.config.allow_file_write:
            raise ValidationError("File writing is disabled")
            
        validated_path = self.validate_file_path(file_path, "write")
        
        # Validate content
        if len(content) > self.config.max_file_size_mb * 1024 * 1024:
            raise ValidationError("Content too large to write")
        
        # Check for potentially dangerous content
        dangerous_patterns = ['import os', 'subprocess', 'eval(', 'exec(', '__import__']
        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                self.logger.warning(f"Potentially dangerous content detected: {pattern}")
        
        try:
            # Write to temporary file first, then move (atomic operation)
            temp_path = validated_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            shutil.move(temp_path, validated_path)
            
            if self.config.log_file_access:
                self.logger.info(f"File written securely: {validated_path}")
                
        except Exception as e:
            self.logger.error(f"Secure file write failed: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def cleanup_temp_dirs(self) -> None:
        """Clean up all created temporary directories"""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                if self.config.log_file_access:
                    self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup temp dir {temp_dir}: {e}")
        
        self._temp_dirs.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_temp_dirs()


class ModelValidator:
    """Validate PyTorch models for security"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("photonic_mlir.security.model_validator")
    
    def validate_model_security(self, model, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security validation of PyTorch model"""
        validation_results = {
            "secure": True,
            "warnings": [],
            "blocked_operations": [],
            "model_hash": None
        }
        
        # Check model size
        model_size_mb = model_info.get("model_size_mb", 0)
        if model_size_mb > self.config.max_model_size_mb:
            validation_results["secure"] = False
            validation_results["blocked_operations"].append(
                f"Model too large: {model_size_mb:.1f}MB > {self.config.max_model_size_mb}MB"
            )
        
        # Check model type
        model_type = type(model).__name__
        if model_type not in self.config.allowed_model_types:
            validation_results["warnings"].append(f"Unknown model type: {model_type}")
        
        # Scan for potentially dangerous operations
        dangerous_modules = self._scan_for_dangerous_modules(model)
        if dangerous_modules:
            validation_results["blocked_operations"].extend(dangerous_modules)
            validation_results["secure"] = False
        
        # Generate model hash for integrity
        validation_results["model_hash"] = self._generate_model_hash(model)
        
        # Log security validation
        if self.config.log_security_events:
            self.logger.info("Model security validation completed", extra={
                "model_type": model_type,
                "model_size_mb": model_size_mb,
                "secure": validation_results["secure"],
                "warnings": validation_results["warnings"],
                "blocked_operations": validation_results["blocked_operations"]
            })
        
        return validation_results
    
    def _scan_for_dangerous_modules(self, model) -> List[str]:
        """Scan model for potentially dangerous operations"""
        dangerous_operations = []
        
        # Check for custom forward methods that might contain dangerous code
        for name, module in model.named_modules():
            # Check if module has custom forward method
            if hasattr(module, 'forward'):
                forward_method = getattr(module, 'forward')
                if hasattr(forward_method, '__code__'):
                    code = forward_method.__code__
                    
                    # Check for dangerous function calls
                    dangerous_names = {'eval', 'exec', 'compile', '__import__', 'open', 'subprocess'}
                    if dangerous_names.intersection(code.co_names):
                        dangerous_operations.append(f"Dangerous operations in {name}: {dangerous_names.intersection(code.co_names)}")
        
        return dangerous_operations
    
    def _generate_model_hash(self, model) -> str:
        """Generate SHA-256 hash of model parameters"""
        import torch
        
        # Concatenate all parameter tensors
        param_bytes = b''
        for param in model.parameters():
            param_bytes += param.detach().cpu().numpy().tobytes()
        
        # Generate hash
        return hashlib.sha256(param_bytes).hexdigest()


class SecureCompilationEnvironment:
    """Secure environment for model compilation"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("photonic_mlir.security.compilation")
        self.file_handler = SecureFileHandler(config)
        self.model_validator = ModelValidator(config)
        self._session_id = secrets.token_hex(16)
        
    def create_compilation_session(self) -> Dict[str, Any]:
        """Create a secure compilation session"""
        session = {
            "session_id": self._session_id,
            "created_at": datetime.now().isoformat(),
            "temp_dir": self.file_handler.create_secure_temp_dir(),
            "restrictions": {
                "sandbox_mode": self.config.sandbox_mode,
                "file_write_enabled": self.config.allow_file_write,
                "network_access": self.config.allow_network_access
            }
        }
        
        if self.config.log_security_events:
            self.logger.info("Secure compilation session created", extra=session)
            
        return session
    
    def validate_compilation_inputs(self, model, example_input, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all compilation inputs for security"""
        validation_results = {
            "model_validation": None,
            "input_validation": None,
            "config_validation": None,
            "overall_secure": True
        }
        
        # Validate model
        try:
            from .validation import InputValidator
            model_info = InputValidator.validate_model(model)
            model_security = self.model_validator.validate_model_security(model, model_info)
            validation_results["model_validation"] = model_security
            
            if not model_security["secure"]:
                validation_results["overall_secure"] = False
                
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            validation_results["overall_secure"] = False
        
        # Validate input tensor
        try:
            from .validation import InputValidator
            input_info = InputValidator.validate_input_tensor(example_input)
            validation_results["input_validation"] = {"secure": True, "info": input_info}
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            validation_results["overall_secure"] = False
        
        # Validate configuration
        config_security = self._validate_config_security(config)
        validation_results["config_validation"] = config_security
        if not config_security["secure"]:
            validation_results["overall_secure"] = False
        
        return validation_results
    
    def _validate_config_security(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compilation configuration for security"""
        security_result = {"secure": True, "issues": []}
        
        # Check power budget (prevent resource exhaustion)
        power_budget = config.get("power_budget", 100)
        if power_budget > 1000:  # mW
            security_result["issues"].append(f"Excessive power budget: {power_budget}mW")
        
        # Check wavelength count (prevent resource exhaustion)
        wavelengths = config.get("wavelengths", [])
        if len(wavelengths) > 64:
            security_result["issues"].append(f"Too many wavelengths: {len(wavelengths)}")
        
        # Check for dangerous configuration options
        dangerous_keys = ["debug_mode", "unsafe_optimizations", "disable_validation"]
        for key in dangerous_keys:
            if config.get(key, False):
                security_result["issues"].append(f"Dangerous config option enabled: {key}")
        
        if security_result["issues"]:
            security_result["secure"] = False
        
        return security_result
    
    def cleanup_session(self) -> None:
        """Clean up secure compilation session"""
        self.file_handler.cleanup_temp_dirs()
        
        if self.config.log_security_events:
            self.logger.info(f"Compilation session cleaned up: {self._session_id}")


class SecureHardwareInterface:
    """Secure interface for hardware operations"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("photonic_mlir.security.hardware")
        self._authorized_devices = {"simulator", "lightmatter_envise"}
        
    def validate_hardware_access(self, device_name: str, operation: str) -> bool:
        """Validate hardware access request"""
        if device_name not in self._authorized_devices:
            self.logger.warning(f"Unauthorized hardware access attempt: {device_name}")
            return False
        
        # Check if network access is required and allowed
        if device_name != "simulator" and not self.config.allow_network_access:
            self.logger.warning(f"Network access required but disabled for device: {device_name}")
            return False
        
        if self.config.log_security_events:
            self.logger.info(f"Hardware access validated: {device_name} for {operation}")
        
        return True
    
    def sanitize_hardware_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize hardware configuration"""
        sanitized = {}
        
        # Allow only safe configuration keys
        safe_keys = {
            "device", "calibration_file", "power_limit", "temperature_limit",
            "wavelengths", "timeout_seconds"
        }
        
        for key, value in config.items():
            if key in safe_keys:
                # Additional sanitization per key
                if key == "calibration_file" and isinstance(value, str):
                    # Validate file path
                    try:
                        file_handler = SecureFileHandler(self.config)
                        sanitized[key] = file_handler.validate_file_path(value, "read")
                    except ValidationError as e:
                        self.logger.warning(f"Invalid calibration file path: {e}")
                        continue
                elif key == "power_limit" and isinstance(value, (int, float)):
                    # Limit power to reasonable range
                    sanitized[key] = max(0.1, min(value, 1000.0))
                else:
                    sanitized[key] = value
        
        return sanitized


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self):
        self.logger = get_logger("photonic_mlir.security.audit")
        
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = "INFO") -> None:
        """Log security-related events"""
        audit_entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "details": details,
            "session_info": {
                "user": os.getenv("USER", "unknown"),
                "pid": os.getpid()
            }
        }
        
        if severity == "CRITICAL":
            self.logger.critical(f"Security event: {event_type}", extra=audit_entry)
        elif severity == "WARNING":
            self.logger.warning(f"Security event: {event_type}", extra=audit_entry)
        else:
            self.logger.info(f"Security event: {event_type}", extra=audit_entry)
    
    def log_access_attempt(self, resource: str, operation: str, 
                          allowed: bool, reason: str = None) -> None:
        """Log resource access attempts"""
        self.log_security_event(
            "resource_access",
            {
                "resource": resource,
                "operation": operation,
                "allowed": allowed,
                "reason": reason
            },
            severity="WARNING" if not allowed else "INFO"
        )


# Global instances
_global_audit_logger = AuditLogger()
_global_security_config = SecurityConfig()


def get_audit_logger() -> AuditLogger:
    """Get global audit logger"""
    return _global_audit_logger


def get_security_config() -> SecurityConfig:
    """Get global security configuration"""
    return _global_security_config


def create_secure_compilation_environment() -> SecureCompilationEnvironment:
    """Create a secure compilation environment"""
    return SecureCompilationEnvironment(_global_security_config)