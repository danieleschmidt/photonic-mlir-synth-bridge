"""
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
