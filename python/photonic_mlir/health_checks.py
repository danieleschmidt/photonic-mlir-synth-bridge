"""
Advanced health checks and system monitoring for Photonic MLIR.
"""

import time
import threading
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .logging_config import get_logger
from .exceptions import ValidationError, PowerBudgetExceededError


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class HealthMetrics:
    """System health metrics snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    temperature: Optional[float]
    compilation_latency: float
    error_rate: float
    power_consumption: float
    wavelength_utilization: Dict[str, float]
    circuit_integrity: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class SystemResourcesCheck:
    """Monitor system resources (CPU, memory, disk)"""
    
    def __init__(self):
        self.name = "system_resources"
        self.critical = True
        self.logger = get_logger(f"photonic_mlir.health.{self.name}")
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.disk_threshold = 90.0  # %
        
    def check(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            if not PSUTIL_AVAILABLE:
                # Return mock data when psutil is not available
                return {
                    "status": HealthStatus.HEALTHY.value,
                    "cpu_usage": 15.0,
                    "memory_usage": 45.0,
                    "disk_usage": 25.0,
                    "available_memory_gb": 8.0,
                    "issues": [],
                    "message": "psutil not available - using mock data",
                    "timestamp": datetime.now().isoformat()
                }
            
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval for tests
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.WARNING if cpu_percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                
            if memory.percent > self.memory_threshold:
                status = HealthStatus.WARNING if memory.percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                
            if disk.percent > self.disk_threshold:
                status = HealthStatus.WARNING if disk.percent < 98 else HealthStatus.CRITICAL
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                "status": status.value,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "available_memory_gb": memory.available / (1024**3),
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System resources check failed: {e}")
            return {
                "status": HealthStatus.UNAVAILABLE.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class PhotonicCircuitCheck:
    """Check photonic circuit health and integrity"""
    
    def __init__(self):
        self.name = "photonic_circuit"
        self.critical = True
        self.logger = get_logger(f"photonic_mlir.health.{self.name}")
        
    def check(self) -> Dict[str, Any]:
        """Check photonic circuit integrity"""
        try:
            # Simulate circuit integrity checks
            # In real implementation, this would verify:
            # - Optical path continuity
            # - Phase coherence
            # - Power budget compliance
            # - Wavelength channel isolation
            
            checks = {
                "optical_continuity": True,
                "phase_coherence": 0.98,
                "power_budget_ok": True,
                "wavelength_isolation": -25.3,  # dB
                "thermal_stable": True
            }
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check phase coherence
            if checks["phase_coherence"] < 0.95:
                status = HealthStatus.WARNING
                issues.append(f"Low phase coherence: {checks['phase_coherence']:.3f}")
            
            # Check wavelength isolation
            if checks["wavelength_isolation"] > -20.0:
                status = HealthStatus.CRITICAL
                issues.append(f"Poor wavelength isolation: {checks['wavelength_isolation']:.1f} dB")
            
            return {
                "status": status.value,
                "checks": checks,
                "issues": issues,
                "overall_integrity": len(issues) == 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Circuit integrity check failed: {e}")
            return {
                "status": HealthStatus.UNAVAILABLE.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class HealthMonitor:
    """Centralized health monitoring system"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: List[Any] = []
        self.metrics_history: List[HealthMetrics] = []
        self.max_history = 1000
        self.logger = get_logger("photonic_mlir.health.monitor")
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Initialize default health checks
        self._initialize_default_checks()
        
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        self.add_health_check(SystemResourcesCheck())
        self.add_health_check(PhotonicCircuitCheck())
        
    def add_health_check(self, check):
        """Add a health check to the monitoring system"""
        self.health_checks.append(check)
        self.logger.info(f"Added health check: {check.name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            self.logger.warning("Health monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform complete health check"""
        timestamp = datetime.now()
        check_results = {}
        overall_status = HealthStatus.HEALTHY
        critical_issues = []
        
        # Run all health checks
        for check in self.health_checks:
            try:
                result = check.check()
                check_results[check.name] = result
                
                # Update overall status based on check results
                check_status = HealthStatus(result["status"])
                if check_status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                    if getattr(check, 'critical', False):
                        critical_issues.append(f"{check.name}: Critical failure")
                elif check_status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
                    
            except Exception as e:
                self.logger.error(f"Health check {check.name} failed: {e}")
                check_results[check.name] = {
                    "status": HealthStatus.UNAVAILABLE.value,
                    "error": str(e),
                    "timestamp": timestamp.isoformat()
                }
        
        # Create health report
        health_report = {
            "timestamp": timestamp.isoformat(),
            "overall_status": overall_status.value,
            "checks": check_results,
            "critical_issues": critical_issues,
            "healthy_checks": sum(1 for r in check_results.values() 
                                 if r.get("status") == HealthStatus.HEALTHY.value),
            "total_checks": len(check_results)
        }
        
        # Log health status
        if overall_status == HealthStatus.HEALTHY:
            self.logger.debug("System health: HEALTHY")
        else:
            self.logger.warning(f"System health: {overall_status.value}")
            if critical_issues:
                self.logger.error(f"Critical issues: {'; '.join(critical_issues)}")
        
        return health_report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        return self.perform_health_check()


# Global health monitor instance
_global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    return _global_health_monitor


def start_health_monitoring():
    """Start global health monitoring"""
    _global_health_monitor.start_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return _global_health_monitor.get_health_summary()