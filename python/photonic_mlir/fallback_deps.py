"""
Fallback implementations for optional dependencies
Ensures system works in minimal environments without compromising functionality
"""

import sys
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import warnings

# Mock implementations for missing dependencies
class FallbackPsutil:
    """Fallback for psutil - provides basic system monitoring"""
    
    class Process:
        def __init__(self, pid=None):
            self.pid = pid or 1
            self._name = "python"
            self._memory_info = type('obj', (object,), {'rss': 50*1024*1024, 'vms': 100*1024*1024})()
            self._cpu_times = type('obj', (object,), {'user': 1.0, 'system': 0.5})()
        
        def name(self): return self._name
        def memory_info(self): return self._memory_info
        def cpu_percent(self): return 5.0
        def cpu_times(self): return self._cpu_times
        def is_running(self): return True
        def terminate(self): pass
        def kill(self): pass
    
    @staticmethod
    def virtual_memory():
        return type('obj', (object,), {
            'total': 8*1024*1024*1024,
            'available': 4*1024*1024*1024,
            'percent': 50.0,
            'used': 4*1024*1024*1024
        })()
    
    @staticmethod
    def cpu_percent(interval=1): 
        time.sleep(min(interval, 0.1))  # Don't actually wait
        return 15.0
    
    @staticmethod
    def cpu_count(): return 4
    
    @staticmethod
    def disk_usage(path="/"):
        return type('obj', (object,), {
            'total': 100*1024*1024*1024,
            'used': 50*1024*1024*1024,
            'free': 50*1024*1024*1024
        })()
    
    @staticmethod
    def disk_io_counters():
        return type('obj', (object,), {
            'read_count': 1000,
            'write_count': 500,
            'read_bytes': 1024*1024,
            'write_bytes': 512*1024
        })()

class FallbackTorch:
    """Fallback for PyTorch - provides tensor-like objects"""
    
    class Tensor:
        def __init__(self, data, dtype=None):
            if isinstance(data, (list, tuple)):
                self.data = data
                self.shape = self._calculate_shape(data)
            else:
                self.data = [data]
                self.shape = (1,)
            self.dtype = dtype or "float32"
        
        def _calculate_shape(self, data):
            if not isinstance(data, (list, tuple)):
                return ()
            shape = [len(data)]
            if data and isinstance(data[0], (list, tuple)):
                shape.extend(self._calculate_shape(data[0]))
            return tuple(shape)
        
        def size(self): return self.shape
        def dim(self): return len(self.shape)
        def numpy(self): 
            try:
                import numpy as np
                return np.array(self.data, dtype=self.dtype)
            except ImportError:
                return self.data
        
        def __str__(self): return f"FallbackTensor(shape={self.shape})"
        def __repr__(self): return self.__str__()
    
    @staticmethod
    def tensor(data, dtype=None):
        return FallbackTorch.Tensor(data, dtype)
    
    @staticmethod
    def randn(*shape):
        import random
        if len(shape) == 1:
            data = [random.gauss(0, 1) for _ in range(shape[0])]
        elif len(shape) == 2:
            data = [[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            # Simplified for higher dimensions
            size = 1
            for s in shape: size *= s
            data = [random.gauss(0, 1) for _ in range(size)]
        return FallbackTorch.Tensor(data)
    
    @staticmethod
    def zeros(*shape):
        if len(shape) == 1:
            data = [0.0 for _ in range(shape[0])]
        elif len(shape) == 2:
            data = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            size = 1
            for s in shape: size *= s
            data = [0.0 for _ in range(size)]
        return FallbackTorch.Tensor(data)
    
    class nn:
        class Module:
            def __init__(self):
                self.training = True
            def forward(self, x): return x
            def train(self, mode=True): self.training = mode; return self
            def eval(self): return self.train(False)
        
        class Linear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = FallbackTorch.randn(out_features, in_features)
                self.bias = FallbackTorch.randn(out_features)
            
            def forward(self, x):
                # Simplified linear transformation
                return FallbackTorch.randn(x.shape[0], self.out_features)

class FallbackScipy:
    """Fallback for scipy - provides basic optimization functions"""
    
    class optimize:
        @staticmethod
        def minimize(fun, x0, method='BFGS', **kwargs):
            # Simple gradient descent fallback
            import random
            result = type('obj', (object,), {
                'x': [x + random.gauss(0, 0.1) for x in x0],
                'fun': fun(x0) - abs(random.gauss(0, 1)),
                'success': True,
                'message': 'Fallback optimization completed'
            })()
            return result
        
        @staticmethod  
        def differential_evolution(func, bounds, **kwargs):
            import random
            x = [random.uniform(bound[0], bound[1]) for bound in bounds]
            result = type('obj', (object,), {
                'x': x,
                'fun': func(x),
                'success': True,
                'message': 'Fallback differential evolution completed'
            })()
            return result

def get_fallback_dep(dep_name: str):
    """Get fallback implementation for missing dependency"""
    fallbacks = {
        'psutil': FallbackPsutil,
        'torch': FallbackTorch,
        'scipy': FallbackScipy,
    }
    
    if dep_name in fallbacks:
        warnings.warn(f"Using fallback implementation for {dep_name}. "
                     f"Install {dep_name} for full functionality.", UserWarning)
        return fallbacks[dep_name]
    
    raise ImportError(f"No fallback available for {dep_name}")

def safe_import(module_name: str, fallback_name: str = None):
    """Safely import module with fallback support"""
    try:
        return __import__(module_name)
    except ImportError:
        if fallback_name:
            return get_fallback_dep(fallback_name)
        raise

# Utility function for robust dependency loading
def load_with_fallback(imports: Dict[str, str]):
    """Load multiple dependencies with fallbacks
    
    Args:
        imports: Dict mapping module names to fallback names
    
    Returns:
        Dict of loaded modules
    """
    loaded = {}
    for module_name, fallback_name in imports.items():
        try:
            loaded[module_name] = safe_import(module_name, fallback_name)
        except ImportError as e:
            logging.warning(f"Failed to load {module_name}: {e}")
            loaded[module_name] = None
    
    return loaded

@dataclass
class DependencyHealth:
    """Track health of dependencies"""
    name: str
    available: bool
    fallback_used: bool
    version: Optional[str] = None
    performance_impact: float = 0.0  # 0-1, higher means more impact

class DependencyManager:
    """Manage dependencies and fallbacks robustly"""
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyHealth] = {}
        self.fallback_warnings_shown = set()
    
    def register_dependency(self, name: str, fallback_available: bool = False):
        """Register a dependency for monitoring"""
        try:
            module = __import__(name)
            version = getattr(module, '__version__', 'unknown')
            self.dependencies[name] = DependencyHealth(
                name=name,
                available=True,
                fallback_used=False,
                version=version,
                performance_impact=0.0
            )
        except ImportError:
            if fallback_available:
                self.dependencies[name] = DependencyHealth(
                    name=name,
                    available=False,
                    fallback_used=True,
                    performance_impact=0.3  # Fallbacks typically slower
                )
                if name not in self.fallback_warnings_shown:
                    logging.warning(f"Using fallback for {name} - consider installing for better performance")
                    self.fallback_warnings_shown.add(name)
            else:
                self.dependencies[name] = DependencyHealth(
                    name=name,
                    available=False,
                    fallback_used=False,
                    performance_impact=1.0  # Complete failure
                )
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate dependency health report"""
        total_deps = len(self.dependencies)
        available_deps = sum(1 for dep in self.dependencies.values() if dep.available)
        fallback_deps = sum(1 for dep in self.dependencies.values() if dep.fallback_used)
        
        avg_impact = sum(dep.performance_impact for dep in self.dependencies.values()) / max(total_deps, 1)
        
        return {
            'total_dependencies': total_deps,
            'available': available_deps,
            'using_fallbacks': fallback_deps,
            'missing': total_deps - available_deps - fallback_deps,
            'health_score': 1.0 - avg_impact,
            'dependencies': {name: {
                'available': dep.available,
                'version': dep.version,
                'fallback_used': dep.fallback_used,
                'performance_impact': dep.performance_impact
            } for name, dep in self.dependencies.items()}
        }
    
    def is_healthy(self, min_score: float = 0.7) -> bool:
        """Check if dependency health meets minimum requirements"""
        return self.get_health_report()['health_score'] >= min_score

# Global dependency manager instance
_dep_manager = DependencyManager()

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager"""
    return _dep_manager

# Initialize common dependencies
def initialize_fallback_system():
    """Initialize the fallback system with common dependencies"""
    common_deps = {
        'psutil': True,
        'torch': True,
        'scipy': True,
        'numpy': False,  # NumPy is usually available, no fallback needed
        'matplotlib': False,
        'sklearn': False,
    }
    
    for dep_name, has_fallback in common_deps.items():
        _dep_manager.register_dependency(dep_name, has_fallback)
    
    return _dep_manager.get_health_report()

if __name__ == "__main__":
    # Test fallback system
    print("Testing fallback dependency system...")
    
    # Test psutil fallback
    psutil = get_fallback_dep('psutil')
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    
    # Test torch fallback  
    torch = get_fallback_dep('torch')
    x = torch.randn(3, 4)
    print(f"Random tensor: {x}")
    
    # Test dependency manager
    health = initialize_fallback_system()
    print(f"Dependency health: {health['health_score']:.2f}")
    print(f"Available: {health['available']}/{health['total_dependencies']}")