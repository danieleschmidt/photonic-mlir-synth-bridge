"""
Health checks and monitoring for Photonic MLIR system.
"""

import time
import psutil
import torch
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from .logging_config import get_logger
from .exceptions import HardwareError, SimulationError


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


class HealthChecker:
    """Base class for health checks"""
    
    def __init__(self, name: str, timeout_seconds: float = 10.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.logger = get_logger(f"photonic_mlir.health.{name}")
        
    def check(self) -> HealthCheckResult:
        """Perform the health check"""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            status, message, details = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                timestamp=timestamp,
                duration_ms=duration_ms
            )
            
            self.logger.debug(f"Health check completed: {self.name}", extra={
                "status": status.value,
                "duration_ms": duration_ms,
                "details": details
            })
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check failed: {self.name}", exc_info=True)
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=timestamp,
                duration_ms=duration_ms
            )
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method to implement specific health check logic"""
        raise NotImplementedError


class SystemResourcesHealthCheck(HealthChecker):
    """Check system resource availability"""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0,
                 disk_threshold: float = 90.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk_percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine status
        critical_issues = []
        warning_issues = []
        
        if cpu_percent > self.cpu_threshold:
            critical_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        elif cpu_percent > self.cpu_threshold * 0.8:
            warning_issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
        if memory_percent > self.memory_threshold:
            critical_issues.append(f"High memory usage: {memory_percent:.1f}%")
        elif memory_percent > self.memory_threshold * 0.8:
            warning_issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
        if disk_percent > self.disk_threshold:
            critical_issues.append(f"High disk usage: {disk_percent:.1f}%")
        elif disk_percent > self.disk_threshold * 0.8:
            warning_issues.append(f"Elevated disk usage: {disk_percent:.1f}%")
        
        if critical_issues:
            return HealthStatus.CRITICAL, "; ".join(critical_issues), details
        elif warning_issues:
            return HealthStatus.WARNING, "; ".join(warning_issues), details
        else:
            return HealthStatus.HEALTHY, "System resources normal", details


class PyTorchHealthCheck(HealthChecker):
    """Check PyTorch and CUDA availability"""
    
    def __init__(self):
        super().__init__("pytorch")
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        details = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Test basic tensor operations
        try:
            x = torch.randn(100, 100)
            y = torch.mm(x, x)
            details["cpu_computation_test"] = True
        except Exception as e:
            return HealthStatus.CRITICAL, f"PyTorch CPU computation failed: {e}", details
        
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                x_cuda = torch.randn(100, 100, device='cuda')
                y_cuda = torch.mm(x_cuda, x_cuda)
                details["cuda_computation_test"] = True
                details["cuda_memory_allocated"] = torch.cuda.memory_allocated() / (1024**2)  # MB
            except Exception as e:
                return HealthStatus.WARNING, f"CUDA computation failed: {e}", details
        
        return HealthStatus.HEALTHY, "PyTorch working normally", details


class MLIRHealthCheck(HealthChecker):
    """Check MLIR availability and basic functionality"""
    
    def __init__(self):
        super().__init__("mlir")
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        details = {}
        
        # Check if MLIR bindings are available
        try:
            from ..bindings import PhotonicMLIRPythonModule as native
            details["native_bindings_available"] = True
            
            # Test basic MLIR context creation
            context = native.create_photonic_context()
            details["context_creation_test"] = True
            
        except ImportError:
            details["native_bindings_available"] = False
            return HealthStatus.WARNING, "MLIR native bindings not available (running in simulation mode)", details
        except Exception as e:
            details["native_bindings_available"] = True
            return HealthStatus.CRITICAL, f"MLIR functionality test failed: {e}", details
        
        return HealthStatus.HEALTHY, "MLIR working normally", details


class PhotonicHardwareHealthCheck(HealthChecker):
    """Check photonic hardware connectivity and status"""
    
    def __init__(self, device_name: str = "simulator"):
        super().__init__(f"hardware_{device_name}")
        self.device_name = device_name
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        from .simulation import HardwareInterface, PhotonicDevice
        
        details = {"device": self.device_name}
        
        try:
            # Test hardware interface
            interface = HardwareInterface(self.device_name)
            connected = interface.connect()
            
            if not connected:
                return HealthStatus.CRITICAL, f"Failed to connect to {self.device_name}", details
            
            # Get device status
            status = interface.get_status()
            details.update(status)
            
            # Check for any critical status indicators
            if not status.get("calibration_valid", True):
                return HealthStatus.WARNING, "Device calibration invalid", details
            
            if status.get("temperature", 25) > 80:  # High temperature
                return HealthStatus.WARNING, f"High device temperature: {status['temperature']}°C", details
            
            return HealthStatus.HEALTHY, f"Hardware {self.device_name} operational", details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Hardware check failed: {e}", details


class CompilationHealthCheck(HealthChecker):
    """Test end-to-end compilation pipeline"""
    
    def __init__(self):
        super().__init__("compilation_pipeline")
    
    def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        from .compiler import PhotonicCompiler, PhotonicBackend
        from .pytorch_frontend import PhotonicMLP
        
        details = {}
        
        try:
            # Create simple test model
            model = PhotonicMLP(input_size=10, hidden_sizes=[5], num_classes=2)
            model.eval()
            
            example_input = torch.randn(1, 10)
            
            # Test compilation
            compiler = PhotonicCompiler(
                backend=PhotonicBackend.SIMULATION_ONLY,
                wavelengths=[1550.0],
                power_budget=50.0
            )
            
            circuit = compiler.compile(model, example_input, optimization_level=1)
            details["compilation_successful"] = True
            
            # Test HLS generation
            hls_code = circuit.generate_hls()
            details["hls_generation_successful"] = len(hls_code) > 0
            
            return HealthStatus.HEALTHY, "Compilation pipeline working", details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Compilation pipeline failed: {e}", details


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.checks: Dict[str, HealthChecker] = {}
        self.logger = get_logger("photonic_mlir.health_monitor")
        self.last_results: Dict[str, HealthCheckResult] = {}
        
        # Register default health checks
        self.register_default_checks()
    
    def register_default_checks(self) -> None:
        """Register default health checks"""
        self.add_check(SystemResourcesHealthCheck())
        self.add_check(PyTorchHealthCheck())
        self.add_check(MLIRHealthCheck())
        self.add_check(PhotonicHardwareHealthCheck())
        self.add_check(CompilationHealthCheck())
    
    def add_check(self, checker: HealthChecker) -> None:
        """Add a health check to the monitor"""
        self.checks[checker.name] = checker
        self.logger.debug(f"Added health check: {checker.name}")
    
    def remove_check(self, name: str) -> None:
        """Remove a health check from the monitor"""
        if name in self.checks:
            del self.checks[name]
            self.logger.debug(f"Removed health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.checks:
            raise ValueError(f"Health check not found: {name}")
        
        result = self.checks[name].check()
        self.last_results[name] = result
        
        # Log significant status changes
        self._log_status_change(result)
        
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.checks:
            try:
                results[name] = self.run_check(name)
            except Exception as e:
                self.logger.error(f"Failed to run health check {name}: {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check execution failed: {e}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    duration_ms=0.0
                )
        
        # Log overall system health
        self._log_overall_health(results)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        # Overall status is the worst individual status
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health"""
        if not self.last_results:
            return {"status": "unknown", "message": "No health checks run yet"}
        
        overall_status = self.get_overall_status()
        
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for result in self.last_results.values() 
                if result.status == status
            )
        
        critical_issues = [
            result.message for result in self.last_results.values()
            if result.status == HealthStatus.CRITICAL
        ]
        
        warning_issues = [
            result.message for result in self.last_results.values()
            if result.status == HealthStatus.WARNING
        ]
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.last_results),
            "status_breakdown": status_counts,
            "critical_issues": critical_issues,
            "warning_issues": warning_issues,
            "last_check_time": max(
                result.timestamp for result in self.last_results.values()
            ).isoformat() if self.last_results else None
        }
    
    def _log_status_change(self, result: HealthCheckResult) -> None:
        """Log when health status changes"""
        previous_result = self.last_results.get(result.name)
        
        if previous_result and previous_result.status != result.status:
            self.logger.warning(f"Health status changed for {result.name}: "
                              f"{previous_result.status.value} -> {result.status.value}",
                              extra={
                                  "check_name": result.name,
                                  "previous_status": previous_result.status.value,
                                  "new_status": result.status.value,
                                  "message": result.message
                              })
    
    def _log_overall_health(self, results: Dict[str, HealthCheckResult]) -> None:
        """Log overall system health status"""
        overall_status = self.get_overall_status()
        
        self.logger.info(f"System health check completed - Status: {overall_status.value}",
                        extra={
                            "overall_status": overall_status.value,
                            "total_checks": len(results),
                            "results_summary": {
                                name: {"status": result.status.value, "duration_ms": result.duration_ms}
                                for name, result in results.items()
                            }
                        })


# Global health monitor instance
_global_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    return _global_monitor


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return summary"""
    monitor = get_health_monitor()
    monitor.run_all_checks()
    return monitor.get_health_summary()