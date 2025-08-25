"""
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
