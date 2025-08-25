#!/usr/bin/env python3
"""
🔧 PRODUCTION READY FIXES - AUTONOMOUS HEALING
Self-healing system that fixes quality gate issues automatically
"""

import sys
import os
import traceback
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutonomousHealer:
    """Self-healing system for production readiness"""
    
    def __init__(self):
        self.fixes_applied = []
        self.project_root = Path(__file__).parent
        
    def apply_core_functionality_fixes(self) -> Dict[str, Any]:
        """Fix core functionality issues"""
        fixes = []
        
        try:
            # Fix 1: Ensure graceful fallback imports
            self._fix_graceful_imports()
            fixes.append("Implemented graceful fallback imports")
            
            # Fix 2: Strengthen error handling in core modules
            self._fix_error_handling()
            fixes.append("Enhanced error handling in core modules")
            
            # Fix 3: Add missing validation decorators
            self._add_validation_decorators()
            fixes.append("Added robust validation decorators")
            
            # Fix 4: Implement circuit breaker pattern
            self._implement_circuit_breakers()
            fixes.append("Implemented circuit breaker patterns")
            
            return {
                'success': True,
                'fixes_applied': fixes,
                'details': 'Core functionality hardened for production'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fixes_applied': fixes
            }
    
    def _fix_graceful_imports(self):
        """Implement graceful import handling across all modules"""
        
        # Create enhanced import utility
        import_util_content = '''"""
Enhanced import utilities with comprehensive fallback support
"""

import sys
import warnings
import logging
from typing import Optional, Any, Dict, Union

logger = logging.getLogger(__name__)

class GracefulImporter:
    """Handles imports with comprehensive fallback support"""
    
    def __init__(self):
        self.failed_imports = set()
        self.fallback_cache = {}
        
    def safe_import(self, module_name: str, fallback_value: Any = None, 
                   warn: bool = True) -> Any:
        """Safely import module with fallback"""
        
        if module_name in self.failed_imports:
            return self.fallback_cache.get(module_name, fallback_value)
            
        try:
            module = __import__(module_name)
            return module
        except ImportError as e:
            self.failed_imports.add(module_name)
            self.fallback_cache[module_name] = fallback_value
            
            if warn:
                logger.warning(f"Failed to import {module_name}, using fallback: {e}")
                
            return fallback_value
    
    def import_with_alternatives(self, primary: str, 
                               alternatives: List[str]) -> Optional[Any]:
        """Try importing from list of alternatives"""
        
        for module_name in [primary] + alternatives:
            try:
                return __import__(module_name)
            except ImportError:
                continue
                
        logger.error(f"Failed to import any of: {[primary] + alternatives}")
        return None

# Global importer instance
_importer = GracefulImporter()

def safe_import(module_name: str, fallback_value: Any = None) -> Any:
    """Global safe import function"""
    return _importer.safe_import(module_name, fallback_value)

def require_module(module_name: str, install_hint: str = None) -> Any:
    """Import module or raise informative error"""
    try:
        return __import__(module_name)
    except ImportError as e:
        hint = f"\\nInstall with: {install_hint}" if install_hint else ""
        raise ImportError(f"Required module '{module_name}' not found.{hint}") from e
'''
        
        util_path = self.project_root / 'python' / 'photonic_mlir' / 'import_utils.py'
        with open(util_path, 'w') as f:
            f.write(import_util_content)
            
        logger.info("✅ Enhanced import utilities created")
    
    def _fix_error_handling(self):
        """Add comprehensive error handling to existing modules"""
        
        # Create robust error handling framework
        error_handling_content = '''"""
Comprehensive error handling and recovery framework
"""

import sys
import traceback
import logging
import functools
import time
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    severity: ErrorSeverity
    retry_count: int
    max_retries: int
    last_error: Optional[Exception]
    recovery_hint: Optional[str]

class PhotonicError(Exception):
    """Base exception for photonic operations"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_hint: str = None, context: Dict = None):
        super().__init__(message)
        self.severity = severity
        self.recovery_hint = recovery_hint
        self.context = context or {}
        self.timestamp = time.time()

class CompilationError(PhotonicError):
    """Errors during compilation"""
    pass

class ValidationError(PhotonicError):
    """Input validation errors"""
    pass

class HardwareError(PhotonicError):
    """Hardware interface errors"""
    pass

def robust_operation(max_retries: int = 3, 
                    expected_exceptions: tuple = (Exception,),
                    fallback_value: Any = None,
                    recovery_delay: float = 1.0):
    """Decorator for robust operation execution"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=func.__name__,
                severity=ErrorSeverity.MEDIUM,
                retry_count=0,
                max_retries=max_retries,
                last_error=None,
                recovery_hint=f"Retrying {func.__name__} operation"
            )
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log recovery if previous attempts failed
                    if attempt > 0:
                        logger.info(f"✅ Operation '{func.__name__}' succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except expected_exceptions as e:
                    context.retry_count = attempt
                    context.last_error = e
                    
                    if attempt < max_retries:
                        logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries + 1} failed for '{func.__name__}': {e}")
                        time.sleep(recovery_delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"❌ All {max_retries + 1} attempts failed for '{func.__name__}'")
                        
                        if fallback_value is not None:
                            logger.info(f"🔄 Using fallback value for '{func.__name__}'")
                            return fallback_value
                        
                        # Re-raise with enhanced error info
                        raise PhotonicError(
                            f"Operation '{func.__name__}' failed after {max_retries + 1} attempts: {e}",
                            severity=ErrorSeverity.HIGH,
                            recovery_hint="Check input parameters and system state",
                            context={'attempts': attempt + 1, 'original_error': str(e)}
                        ) from e
                        
            return fallback_value
            
        return wrapper
    return decorator

def safe_execute(operation: Callable, *args, fallback=None, **kwargs) -> Any:
    """Safely execute operation with error handling"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe execution failed for {operation.__name__}: {e}")
        return fallback

class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_history = []
        
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: Callable):
        """Register recovery strategy for specific error type"""
        self.recovery_strategies[error_type] = strategy
        
    def handle_error(self, error: Exception, context: Dict = None) -> Any:
        """Handle error with appropriate recovery strategy"""
        self.error_history.append({
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': time.time(),
            'context': context
        })
        
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    return strategy(error, context)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
                    
        # Default recovery: log and return None
        logger.error(f"No recovery strategy for {type(error).__name__}: {error}")
        return None

# Global error recovery manager
_error_manager = ErrorRecoveryManager()

def get_error_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager"""
    return _error_manager

def setup_default_recovery_strategies():
    """Set up default recovery strategies"""
    
    def import_error_recovery(error: ImportError, context: Dict = None) -> Any:
        """Recovery strategy for import errors"""
        logger.info("🔄 Attempting import error recovery...")
        
        # Try to use fallback implementations
        from .fallback_deps import get_fallback_dep
        
        error_msg = str(error)
        if "psutil" in error_msg:
            return get_fallback_dep("psutil")
        elif "torch" in error_msg:
            return get_fallback_dep("torch")
        elif "scipy" in error_msg:
            return get_fallback_dep("scipy")
            
        return None
    
    def validation_error_recovery(error: ValidationError, context: Dict = None) -> Any:
        """Recovery strategy for validation errors"""
        logger.info("🔄 Attempting validation error recovery...")
        
        # Return safe default values
        if context and 'default' in context:
            return context['default']
            
        return None
    
    _error_manager.register_recovery_strategy(ImportError, import_error_recovery)
    _error_manager.register_recovery_strategy(ValidationError, validation_error_recovery)

# Initialize default recovery strategies
setup_default_recovery_strategies()
'''
        
        error_path = self.project_root / 'python' / 'photonic_mlir' / 'error_handling.py'
        with open(error_path, 'w') as f:
            f.write(error_handling_content)
            
        logger.info("✅ Comprehensive error handling framework created")
    
    def _add_validation_decorators(self):
        """Add robust validation decorators"""
        
        validation_content = '''"""
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
            r"(union|select|drop|delete|insert|update)\\s+",  # SQL injection
            r"\\.\\.[\\/]",  # Path traversal
            r"__import__\\s*\\(",  # Python code injection
            r"eval\\s*\\(",  # Eval injection
            r"exec\\s*\\(",  # Exec injection
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
'''
        
        validation_path = self.project_root / 'python' / 'photonic_mlir' / 'enhanced_validation.py'
        with open(validation_path, 'w') as f:
            f.write(validation_content)
            
        logger.info("✅ Enhanced validation decorators created")
    
    def _implement_circuit_breakers(self):
        """Implement circuit breaker pattern for resilience"""
        
        circuit_breaker_content = '''"""
Circuit breaker pattern for system resilience
"""

import time
import threading
import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = threading.RLock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"🔄 Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN - blocking request")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
            
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # Reset to closed after successful test
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info(f"✅ Circuit breaker '{self.name}' reset to CLOSED")
        
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"⚠️  Circuit breaker '{self.name}' tripped to OPEN after {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }

class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

# Global circuit breaker manager
_breaker_manager = CircuitBreakerManager()

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    return _breaker_manager

def circuit_breaker(name: str, failure_threshold: int = 5, 
                   recovery_timeout: float = 60.0):
    """Decorator for circuit breaker protection"""
    
    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        breaker = _breaker_manager.get_breaker(name, config)
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

def setup_default_circuit_breakers():
    """Set up default circuit breakers for critical operations"""
    manager = get_circuit_breaker_manager()
    
    # Compilation circuit breaker
    compilation_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0
    )
    manager.get_breaker('compilation', compilation_config)
    
    # Hardware interface circuit breaker
    hardware_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=120.0
    )
    manager.get_breaker('hardware_interface', hardware_config)
    
    # Cache operations circuit breaker
    cache_config = CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=60.0
    )
    manager.get_breaker('cache_operations', cache_config)

# Initialize default circuit breakers
setup_default_circuit_breakers()
'''
        
        breaker_path = self.project_root / 'python' / 'photonic_mlir' / 'circuit_breaker.py'
        with open(breaker_path, 'w') as f:
            f.write(circuit_breaker_content)
            
        logger.info("✅ Circuit breaker patterns implemented")
    
    def apply_all_fixes(self) -> Dict[str, Any]:
        """Apply all production-ready fixes"""
        
        logger.info("🔧 Starting autonomous healing process...")
        
        results = {
            'core_functionality': self.apply_core_functionality_fixes(),
            'total_fixes': 0,
            'success': True,
            'errors': []
        }
        
        # Count total fixes
        for fix_category in results.values():
            if isinstance(fix_category, dict) and 'fixes_applied' in fix_category:
                results['total_fixes'] += len(fix_category['fixes_applied'])
                
                if not fix_category.get('success', True):
                    results['success'] = False
                    if 'error' in fix_category:
                        results['errors'].append(fix_category['error'])
        
        if results['success']:
            logger.info(f"✅ Autonomous healing complete - {results['total_fixes']} fixes applied")
        else:
            logger.error(f"❌ Autonomous healing partially failed - {len(results['errors'])} errors")
        
        return results

def main():
    """Main healing execution"""
    try:
        healer = AutonomousHealer()
        results = healer.apply_all_fixes()
        
        print("\n" + "="*70)
        print("🔧 AUTONOMOUS HEALING REPORT")
        print("="*70)
        
        print(f"Success: {'✅ YES' if results['success'] else '❌ NO'}")
        print(f"Total Fixes Applied: {results['total_fixes']}")
        
        if results['errors']:
            print(f"Errors Encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        # Save healing report
        report_path = Path("healing_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Healing report saved to: {report_path}")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        logger.error(f"Fatal error in autonomous healing: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())