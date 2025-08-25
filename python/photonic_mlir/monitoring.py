"""
Advanced monitoring and observability for Photonic MLIR operations.
"""

import time
import threading

# Robust dependency loading with fallbacks
try:
    import psutil
except ImportError:
    from .fallback_deps import get_fallback_dep
    psutil = get_fallback_dep('psutil')
import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque
import statistics

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None

from .logging_config import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class CompilationMetrics:
    """Compilation-specific metrics"""
    model_name: str
    parameter_count: int
    compilation_time_ms: float
    optimization_level: int
    backend: str
    wavelength_count: int
    power_budget_mw: float
    memory_peak_mb: float
    success: bool
    mlir_size_bytes: int
    hls_size_bytes: Optional[int] = None
    error_details: Optional[str] = None


@dataclass
class SimulationMetrics:
    """Simulation-specific metrics"""
    circuit_name: str
    batch_size: int
    simulation_time_ms: float
    throughput_tops: float
    power_consumption_mw: float
    accuracy: float
    latency_us: float
    pdk: str
    temperature_k: float
    monte_carlo_runs: int
    success: bool


class MetricsCollector:
    """Collect and aggregate performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.performance_history: deque = deque(maxlen=max_history)
        self.compilation_history: deque = deque(maxlen=max_history)
        self.simulation_history: deque = deque(maxlen=max_history)
        self.logger = get_logger("photonic_mlir.monitoring.metrics")
        self._lock = threading.Lock()
        
    def record_performance(self, metrics: PerformanceMetrics):
        """Record general performance metrics"""
        with self._lock:
            self.performance_history.append(metrics)
            
        self.logger.debug("Performance metrics recorded", extra={
            "operation": metrics.operation,
            "duration_ms": metrics.duration_ms,
            "memory_usage_mb": metrics.memory_usage_mb,
            "success": metrics.success
        })
    
    def record_compilation(self, metrics: CompilationMetrics):
        """Record compilation-specific metrics"""
        with self._lock:
            self.compilation_history.append(metrics)
            
        self.logger.info("Compilation metrics recorded", extra={
            "model_name": metrics.model_name,
            "compilation_time_ms": metrics.compilation_time_ms,
            "success": metrics.success,
            "parameter_count": metrics.parameter_count
        })
    
    def record_simulation(self, metrics: SimulationMetrics):
        """Record simulation-specific metrics"""
        with self._lock:
            self.simulation_history.append(metrics)
            
        self.logger.info("Simulation metrics recorded", extra={
            "circuit_name": metrics.circuit_name,
            "throughput_tops": metrics.throughput_tops,
            "power_consumption_mw": metrics.power_consumption_mw,
            "success": metrics.success
        })
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics"""
        with self._lock:
            metrics = list(self.performance_history)
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        if not metrics:
            return {"total_operations": 0}
        
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        
        durations = [m.duration_ms for m in successful]
        memory_usage = [m.memory_usage_mb for m in successful]
        cpu_usage = [m.cpu_percent for m in successful]
        
        return {
            "total_operations": len(metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(metrics) * 100 if metrics else 0,
            "duration_stats": {
                "mean_ms": statistics.mean(durations) if durations else 0,
                "median_ms": statistics.median(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "std_ms": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "memory_stats": {
                "mean_mb": statistics.mean(memory_usage) if memory_usage else 0,
                "peak_mb": max(memory_usage) if memory_usage else 0,
                "min_mb": min(memory_usage) if memory_usage else 0
            },
            "cpu_stats": {
                "mean_percent": statistics.mean(cpu_usage) if cpu_usage else 0,
                "peak_percent": max(cpu_usage) if cpu_usage else 0
            }
        }
    
    def get_compilation_summary(self) -> Dict[str, Any]:
        """Get compilation performance summary"""
        with self._lock:
            compilations = list(self.compilation_history)
        
        if not compilations:
            return {"total_compilations": 0}
        
        successful = [c for c in compilations if c.success]
        failed = [c for c in compilations if not c.success]
        
        # Group by optimization level
        by_opt_level = defaultdict(list)
        for c in successful:
            by_opt_level[c.optimization_level].append(c.compilation_time_ms)
        
        # Group by backend
        by_backend = defaultdict(list)
        for c in successful:
            by_backend[c.backend].append(c.compilation_time_ms)
        
        return {
            "total_compilations": len(compilations),
            "successful_compilations": len(successful),
            "failed_compilations": len(failed),
            "success_rate": len(successful) / len(compilations) * 100 if compilations else 0,
            "by_optimization_level": {
                level: {
                    "count": len(times),
                    "mean_time_ms": statistics.mean(times),
                    "median_time_ms": statistics.median(times)
                }
                for level, times in by_opt_level.items()
            },
            "by_backend": {
                backend: {
                    "count": len(times),
                    "mean_time_ms": statistics.mean(times),
                    "median_time_ms": statistics.median(times)
                }
                for backend, times in by_backend.items()
            },
            "compilation_time_stats": {
                "mean_ms": statistics.mean([c.compilation_time_ms for c in successful]) if successful else 0,
                "median_ms": statistics.median([c.compilation_time_ms for c in successful]) if successful else 0,
                "fastest_ms": min([c.compilation_time_ms for c in successful]) if successful else 0,
                "slowest_ms": max([c.compilation_time_ms for c in successful]) if successful else 0
            }
        }
    
    def export_metrics(self, output_path: str):
        """Export all metrics to JSON file"""
        with self._lock:
            data = {
                "performance_metrics": [asdict(m) for m in self.performance_history],
                "compilation_metrics": [asdict(m) for m in self.compilation_history], 
                "simulation_metrics": [asdict(m) for m in self.simulation_history],
                "export_timestamp": time.time()
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")


class ResourceMonitor:
    """Monitor system resource usage during operations"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_data: List[Dict[str, Any]] = []
        self.logger = get_logger("photonic_mlir.monitoring.resources")
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.resource_data.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        self.logger.debug("Resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return resource usage summary"""
        if not self.monitoring:
            return {}
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        summary = self._generate_resource_summary()
        self.logger.debug("Resource monitoring stopped", extra=summary)
        return summary
    
    def _monitor_resources(self):
        """Background resource monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # System-wide metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                
                # Process-specific metrics
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                # GPU metrics (if available and PyTorch present)
                gpu_memory_mb = 0
                gpu_utilization = 0
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        # Note: GPU utilization requires nvidia-ml-py
                    except Exception:
                        pass
                
                sample = {
                    "timestamp": time.time(),
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_info.percent,
                        "memory_available_gb": memory_info.available / (1024**3),
                        "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                        "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
                    },
                    "process": {
                        "memory_rss_mb": process_memory.rss / (1024**2),
                        "memory_vms_mb": process_memory.vms / (1024**2),
                        "cpu_percent": process_cpu
                    },
                    "gpu": {
                        "memory_mb": gpu_memory_mb,
                        "utilization_percent": gpu_utilization
                    }
                }
                
                self.resource_data.append(sample)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                break
    
    def _generate_resource_summary(self) -> Dict[str, Any]:
        """Generate resource usage summary from collected data"""
        if not self.resource_data:
            return {}
        
        # Extract time series data
        cpu_usage = [sample["system"]["cpu_percent"] for sample in self.resource_data]
        memory_usage = [sample["system"]["memory_percent"] for sample in self.resource_data]
        process_memory = [sample["process"]["memory_rss_mb"] for sample in self.resource_data]
        process_cpu = [sample["process"]["cpu_percent"] for sample in self.resource_data]
        
        duration = self.resource_data[-1]["timestamp"] - self.resource_data[0]["timestamp"]
        
        return {
            "monitoring_duration_seconds": duration,
            "sample_count": len(self.resource_data),
            "system_cpu": {
                "mean_percent": statistics.mean(cpu_usage),
                "peak_percent": max(cpu_usage),
                "min_percent": min(cpu_usage)
            },
            "system_memory": {
                "mean_percent": statistics.mean(memory_usage),
                "peak_percent": max(memory_usage)
            },
            "process_memory": {
                "mean_mb": statistics.mean(process_memory),
                "peak_mb": max(process_memory),
                "min_mb": min(process_memory)
            },
            "process_cpu": {
                "mean_percent": statistics.mean(process_cpu),
                "peak_percent": max(process_cpu)
            }
        }


@contextmanager
def performance_monitor(operation: str, metrics_collector: Optional[MetricsCollector] = None):
    """Context manager for automatic performance monitoring"""
    if metrics_collector is None:
        metrics_collector = _global_metrics_collector
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
    start_cpu_time = psutil.Process().cpu_times()
    
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring()
    
    success = True
    error_message = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        end_cpu_time = psutil.Process().cpu_times()
        
        resource_summary = resource_monitor.stop_monitoring()
        
        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        memory_usage_mb = max(end_memory, start_memory)
        cpu_percent = resource_summary.get("process_cpu", {}).get("mean_percent", 0)
        
        metrics = PerformanceMetrics(
            timestamp=start_time,
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            context=resource_summary
        )
        
        metrics_collector.record_performance(metrics)


class AlertSystem:
    """Alert system for monitoring thresholds"""
    
    def __init__(self):
        self.logger = get_logger("photonic_mlir.monitoring.alerts")
        self.alert_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.thresholds = {
            "compilation_time_ms": 30000,  # 30 seconds
            "memory_usage_mb": 4000,       # 4GB
            "cpu_percent": 90,             # 90%
            "error_rate_percent": 10       # 10%
        }
        
    def add_alert_handler(self, alert_type: str, handler: Callable):
        """Add alert handler for specific alert type"""
        self.alert_handlers[alert_type].append(handler)
        self.logger.debug(f"Alert handler added for {alert_type}")
    
    def check_thresholds(self, metrics_collector: MetricsCollector):
        """Check metrics against thresholds and trigger alerts"""
        perf_summary = metrics_collector.get_performance_summary()
        comp_summary = metrics_collector.get_compilation_summary()
        
        # Check compilation time threshold
        avg_comp_time = comp_summary.get("compilation_time_stats", {}).get("mean_ms", 0)
        if avg_comp_time > self.thresholds["compilation_time_ms"]:
            self._trigger_alert("compilation_slow", {
                "average_time_ms": avg_comp_time,
                "threshold_ms": self.thresholds["compilation_time_ms"]
            })
        
        # Check memory usage
        peak_memory = perf_summary.get("memory_stats", {}).get("peak_mb", 0)
        if peak_memory > self.thresholds["memory_usage_mb"]:
            self._trigger_alert("memory_high", {
                "peak_memory_mb": peak_memory,
                "threshold_mb": self.thresholds["memory_usage_mb"]
            })
        
        # Check error rate
        error_rate = 100 - perf_summary.get("success_rate", 100)
        if error_rate > self.thresholds["error_rate_percent"]:
            self._trigger_alert("error_rate_high", {
                "error_rate_percent": error_rate,
                "threshold_percent": self.thresholds["error_rate_percent"]
            })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger alert of specific type"""
        alert_data = {
            "alert_type": alert_type,
            "timestamp": time.time(),
            "details": details,
            "severity": "WARNING"
        }
        
        self.logger.warning(f"Alert triggered: {alert_type}", extra=alert_data)
        
        # Call registered handlers
        for handler in self.alert_handlers[alert_type]:
            try:
                handler(alert_data)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.logger = get_logger("photonic_mlir.monitoring.health")
        
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # Check available memory
        memory_info = psutil.virtual_memory()
        memory_check = {
            "status": "healthy",
            "available_gb": memory_info.available / (1024**3),
            "percent_used": memory_info.percent
        }
        
        if memory_info.percent > 90:
            memory_check["status"] = "warning"
            memory_check["message"] = "Low memory available"
        elif memory_info.percent > 95:
            memory_check["status"] = "critical"
            memory_check["message"] = "Very low memory available"
            
        health_report["checks"]["memory"] = memory_check
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_check = {
            "status": "healthy",
            "free_gb": disk_usage.free / (1024**3),
            "percent_used": (disk_usage.used / disk_usage.total) * 100
        }
        
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        if disk_percent > 90:
            disk_check["status"] = "warning"
            disk_check["message"] = "Low disk space"
        elif disk_percent > 95:
            disk_check["status"] = "critical"
            disk_check["message"] = "Very low disk space"
            
        health_report["checks"]["disk"] = disk_check
        
        # Check PyTorch availability
        torch_check = {
            "status": "healthy" if TORCH_AVAILABLE else "warning",
            "available": TORCH_AVAILABLE,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        if not TORCH_AVAILABLE:
            torch_check["message"] = "PyTorch not available - limited functionality"
            
        health_report["checks"]["pytorch"] = torch_check
        
        # Determine overall status
        statuses = [check["status"] for check in health_report["checks"].values()]
        if "critical" in statuses:
            health_report["overall_status"] = "critical"
        elif "warning" in statuses:
            health_report["overall_status"] = "warning"
            
        self.logger.info(f"Health check completed: {health_report['overall_status']}")
        return health_report


# Global instances
_global_metrics_collector = MetricsCollector()
_global_alert_system = AlertSystem()
_global_health_checker = HealthChecker()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _global_metrics_collector


def get_alert_system() -> AlertSystem:
    """Get global alert system"""
    return _global_alert_system


def get_health_checker() -> HealthChecker:
    """Get global health checker"""
    return _global_health_checker


def setup_default_alerts():
    """Setup default alert handlers"""
    def log_alert(alert_data):
        logger = get_logger("photonic_mlir.monitoring.alerts.default")
        logger.warning(f"ALERT: {alert_data['alert_type']}", extra=alert_data)
    
    alert_system = get_alert_system()
    alert_system.add_alert_handler("compilation_slow", log_alert)
    alert_system.add_alert_handler("memory_high", log_alert)
    alert_system.add_alert_handler("error_rate_high", log_alert)