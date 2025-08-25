"""
Advanced validation decorators and input sanitization
"""

import functools
import re
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    SECURITY_CHECK = "security_check"

@dataclass
class ValidationRule:
    """Single validation rule"""
    rule_type: ValidationType
    check: Callable
    error_message: str
    severity: str = "error"

class ValidationError(Exception):
    """Validation error with context"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value

class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self):
        self.rules = {}
        self.security_patterns = [
            r"<script.*?>.*?</script>",  # XSS
            r"(union|select|drop|delete|insert|update)\s+",  # SQL injection
            r"\.\.[\/]",  # Path traversal
            r"__import__\s*\(",  # Python code injection
            r"eval\s*\(",  # Eval injection
            r"exec\s*\(",  # Exec injection
        ]
        
    def add_rule(self, field: str, rule: ValidationRule):
        """Add validation rule for field"""
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)
    
    def validate_field(self, field: str, value: Any) -> Dict[str, Any]:
        """Validate single field"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_value': value
        }
        
        if field not in self.rules:
            return result
        
        for rule in self.rules[field]:
            try:
                is_valid = rule.check(value)
                if not is_valid:
                    result['valid'] = False
                    if rule.severity == 'error':
                        result['errors'].append(rule.error_message)
                    else:
                        result['warnings'].append(rule.error_message)
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f"Validation error: {e}")
        
        # Security check for string inputs
        if isinstance(value, str):
            security_result = self._security_check(value)
            if not security_result['safe']:
                result['valid'] = False
                result['errors'].extend(security_result['threats'])
                result['sanitized_value'] = security_result['sanitized']
        
        return result
    
    def _security_check(self, text: str) -> Dict[str, Any]:
        """Check for security threats in text input"""
        threats = []
        sanitized = text
        
        for pattern in self.security_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"Potential security threat detected: {pattern}")
                # Simple sanitization - remove dangerous patterns
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return {
            'safe': len(threats) == 0,
            'threats': threats,
            'sanitized': sanitized
        }
    
    def validate_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dictionary of values"""
        result = {
            'valid': True,
            'field_results': {},
            'sanitized_data': {}
        }
        
        for field, value in data.items():
            field_result = self.validate_field(field, value)
            result['field_results'][field] = field_result
            result['sanitized_data'][field] = field_result['sanitized_value']
            
            if not field_result['valid']:
                result['valid'] = False
        
        return result

# Global validator instance
_validator = InputValidator()

def get_validator() -> InputValidator:
    """Get global validator instance"""
    return _validator

def validate_input(*validation_rules):
    """Decorator for input validation"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate args based on function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            validation_errors = []
            
            # Apply validation rules
            for i, (param_name, value) in enumerate(bound_args.arguments.items()):
                if i < len(validation_rules):
                    rule = validation_rules[i]
                    if rule and not rule(value):
                        validation_errors.append(f"Validation failed for {param_name}: {value}")
            
            if validation_errors:
                raise ValidationError(f"Input validation failed: {'; '.join(validation_errors)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def type_check(expected_type: Type) -> Callable:
    """Create type checking validation rule"""
    def check(value):
        return isinstance(value, expected_type)
    return check

def range_check(min_val: Union[int, float], max_val: Union[int, float]) -> Callable:
    """Create range checking validation rule"""
    def check(value):
        try:
            return min_val <= value <= max_val
        except (TypeError, ValueError):
            return False
    return check

def format_check(pattern: str) -> Callable:
    """Create format checking validation rule"""
    def check(value):
        if not isinstance(value, str):
            return False
        return bool(re.match(pattern, value))
    return check

def setup_default_validation_rules():
    """Set up common validation rules"""
    validator = get_validator()
    
    # Model name validation
    validator.add_rule('model_name', ValidationRule(
        ValidationType.FORMAT_CHECK,
        lambda x: isinstance(x, str) and len(x) > 0 and len(x) < 100,
        "Model name must be non-empty string under 100 characters"
    ))
    
    # Backend validation
    valid_backends = ['lightmatter', 'analog_photonics', 'sim_only', 'hardware']
    validator.add_rule('backend', ValidationRule(
        ValidationType.TYPE_CHECK,
        lambda x: str(x).lower() in valid_backends,
        f"Backend must be one of: {valid_backends}"
    ))
    
    # Wavelength validation
    validator.add_rule('wavelength', ValidationRule(
        ValidationType.RANGE_CHECK,
        lambda x: isinstance(x, (int, float)) and 1000 <= x <= 2000,
        "Wavelength must be between 1000-2000 nm"
    ))
    
    # Power budget validation
    validator.add_rule('power_budget', ValidationRule(
        ValidationType.RANGE_CHECK,
        lambda x: isinstance(x, (int, float)) and 0 < x <= 1000,
        "Power budget must be between 0-1000 mW"
    ))

# Initialize default rules
setup_default_validation_rules()
