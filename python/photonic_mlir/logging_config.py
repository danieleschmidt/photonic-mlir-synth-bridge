"""
Logging configuration and utilities for Photonic MLIR.
"""

import logging
import logging.config
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class PhotonicMLIRLogger:
    """Custom logger for Photonic MLIR with structured logging"""
    
    def __init__(self, name: str = "photonic_mlir"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._configured = False
        
    def configure(self, 
                  level: str = "INFO",
                  log_file: Optional[str] = None,
                  enable_console: bool = True,
                  enable_structured: bool = True) -> None:
        """Configure logging with various options"""
        
        if self._configured:
            return
            
        # Create logs directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure formatters
        formatters = {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s"
            } if enable_structured else "standard"
        }
        
        # Configure handlers
        handlers = {}
        
        if enable_console:
            handlers["console"] = {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            }
        
        if log_file:
            handlers["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5
            }
            
            if enable_structured:
                handlers["json_file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": level,
                    "formatter": "json",
                    "filename": log_file.replace(".log", "_structured.json"),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5
                }
        
        # Configure logger
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": {
                self.name: {
                    "level": level,
                    "handlers": list(handlers.keys()),
                    "propagate": False
                }
            }
        }
        
        logging.config.dictConfig(config)
        self._configured = True
        
        # Log configuration
        self.logger.info(f"Logging configured - Level: {level}, Handlers: {list(handlers.keys())}")
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        if not self._configured:
            self.configure()
        return self.logger


# Global logger instance
_global_logger = PhotonicMLIRLogger()


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance with optional custom name"""
    if name:
        return PhotonicMLIRLogger(name).get_logger()
    return _global_logger.get_logger()


def configure_logging(level: str = "INFO",
                     log_file: Optional[str] = None,
                     enable_console: bool = True,
                     enable_structured: bool = True) -> None:
    """Configure global logging settings"""
    _global_logger.configure(level, log_file, enable_console, enable_structured)


class CompilationLogger:
    """Specialized logger for compilation process"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"compile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = get_logger(f"photonic_mlir.compilation.{self.session_id}")
        self.start_time = datetime.now()
        
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log information about the model being compiled"""
        self.logger.info("Model compilation started", extra={
            "session_id": self.session_id,
            "model_info": model_info,
            "timestamp": self.start_time.isoformat()
        })
    
    def log_stage(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Log compilation stage progress"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"Compilation stage: {stage}", extra={
            "session_id": self.session_id,
            "stage": stage,
            "elapsed_seconds": elapsed,
            "details": details or {}
        })
    
    def log_optimization(self, pass_name: str, metrics: Dict[str, Any]) -> None:
        """Log optimization pass results"""
        self.logger.info(f"Optimization pass completed: {pass_name}", extra={
            "session_id": self.session_id,
            "pass_name": pass_name,
            "metrics": metrics
        })
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log compilation errors with context"""
        self.logger.error(f"Compilation error: {str(error)}", extra={
            "session_id": self.session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "traceback": True
        }, exc_info=True)
    
    def log_completion(self, success: bool, final_metrics: Dict[str, Any] = None) -> None:
        """Log compilation completion"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        status = "SUCCESS" if success else "FAILED"
        
        self.logger.info(f"Compilation {status}", extra={
            "session_id": self.session_id,
            "success": success,
            "total_time_seconds": total_time,
            "final_metrics": final_metrics or {}
        })


class SimulationLogger:
    """Specialized logger for simulation process"""
    
    def __init__(self, sim_id: str = None):
        self.sim_id = sim_id or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = get_logger(f"photonic_mlir.simulation.{self.sim_id}")
        
    def log_simulation_start(self, config: Dict[str, Any]) -> None:
        """Log simulation start with configuration"""
        self.logger.info("Simulation started", extra={
            "simulation_id": self.sim_id,
            "config": config
        })
    
    def log_device_model(self, pdk: str, models: Dict[str, Any]) -> None:
        """Log device model information"""
        self.logger.debug(f"Using device models for {pdk}", extra={
            "simulation_id": self.sim_id,
            "pdk": pdk,
            "device_models": models
        })
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log simulation metrics"""
        self.logger.info("Simulation metrics calculated", extra={
            "simulation_id": self.sim_id,
            "metrics": metrics
        })
    
    def log_hardware_comparison(self, sim_results: Dict[str, float], 
                               hw_results: Dict[str, float]) -> None:
        """Log hardware vs simulation comparison"""
        self.logger.info("Hardware comparison completed", extra={
            "simulation_id": self.sim_id,
            "simulation_results": sim_results,
            "hardware_results": hw_results
        })


class HardwareLogger:
    """Specialized logger for hardware operations"""
    
    def __init__(self, device: str):
        self.device = device
        self.logger = get_logger(f"photonic_mlir.hardware.{device}")
        
    def log_connection(self, success: bool, details: Dict[str, Any] = None) -> None:
        """Log hardware connection attempt"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Hardware connection {status}", extra={
            "device": self.device,
            "success": success,
            "details": details or {}
        })
    
    def log_calibration(self, stage: str, results: Dict[str, Any] = None) -> None:
        """Log calibration progress"""
        self.logger.info(f"Calibration stage: {stage}", extra={
            "device": self.device,
            "calibration_stage": stage,
            "results": results or {}
        })
    
    def log_execution(self, circuit_info: Dict[str, Any], 
                     results: Dict[str, float]) -> None:
        """Log circuit execution on hardware"""
        self.logger.info("Circuit execution completed", extra={
            "device": self.device,
            "circuit_info": circuit_info,
            "execution_results": results
        })
    
    def log_status(self, status: Dict[str, Any]) -> None:
        """Log hardware status"""
        self.logger.debug("Hardware status update", extra={
            "device": self.device,
            "status": status
        })


# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.logger = get_logger("photonic_mlir.performance")
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Operation completed: {self.operation}", extra={
                "operation": self.operation,
                "duration_seconds": duration,
                "success": True
            })
        else:
            self.logger.error(f"Operation failed: {self.operation}", extra={
                "operation": self.operation,
                "duration_seconds": duration,
                "success": False,
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })
    
    def log_metric(self, metric_name: str, value: float, unit: str = None) -> None:
        """Log a performance metric"""
        self.logger.info(f"Performance metric: {metric_name}", extra={
            "operation": self.operation,
            "metric_name": metric_name,
            "value": value,
            "unit": unit
        })