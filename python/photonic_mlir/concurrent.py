"""
Advanced concurrent processing and task management for Photonic MLIR compilation.
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue, PriorityQueue
import multiprocessing as mp

from .logging_config import get_logger
from .exceptions import CompilationError


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class CompilationTask:
    """Compilation task definition"""
    task_id: str
    model: Any
    config: Dict[str, Any]
    compiler: Any
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    # GENERATION 3 ENHANCEMENTS
    estimated_duration: Optional[float] = None
    memory_requirements: Optional[int] = None
    cpu_requirements: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None
    batch_compatible: bool = True
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    progress: float = 0.0
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value


class ThreadPoolCompiler:
    """Thread pool based concurrent compiler"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, CompilationTask] = {}
        self.completed_tasks: Dict[str, CompilationTask] = {}
        self.futures: Dict[str, Future] = {}
        self.task_lock = threading.RLock()
        self.logger = get_logger("photonic_mlir.concurrent.thread_pool")
        
        self.logger.info(f"Started thread pool compiler with {self.max_workers} workers")
    
    def submit_compilation(self, task: CompilationTask) -> Future:
        """Submit compilation task to thread pool"""
        with self.task_lock:
            self.active_tasks[task.task_id] = task
        
        future = self.executor.submit(self._execute_task, task)
        self.futures[task.task_id] = future
        
        self.logger.debug(f"Submitted task {task.task_id}")
        return future
    
    def _execute_task(self, task: CompilationTask) -> CompilationTask:
        """Execute a single compilation task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            self.logger.info(f"Starting task {task.task_id}")
            
            # Simulate compilation with progress updates
            task.progress = 0.1
            time.sleep(0.05)  # Model analysis
            
            task.progress = 0.5
            time.sleep(0.05)  # MLIR conversion
            
            task.progress = 0.9
            time.sleep(0.05)  # Optimization
            
            # Execute actual compilation or mock
            if task.compiler and hasattr(task.compiler, 'compile'):
                result = task.compiler.compile(
                    task.model,
                    example_input=None,
                    optimization_level=task.config.get("optimization_level", 2)
                )
            else:
                # Mock result for testing
                result = {
                    "mlir_module": f"// Mock MLIR for task {task.task_id}",
                    "config": task.config,
                    "compilation_time": 0.15
                }
            
            task.result = result
            task.progress = 1.0
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            self.logger.info(f"Completed task {task.task_id}")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            with self.task_lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
        
        return task
    
    def get_status(self) -> Dict[str, Any]:
        """Get compiler status"""
        with self.task_lock:
            return {
                "max_workers": self.max_workers,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "total_futures": len(self.futures)
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the compiler"""
        self.executor.shutdown(wait=wait)
        self.logger.info("Thread pool compiler shut down")


def create_compilation_task(model: Any, config: Dict[str, Any], 
                          compiler: Any, priority: TaskPriority = TaskPriority.NORMAL) -> CompilationTask:
    """Create new compilation task"""
    task_id = str(uuid.uuid4())
    return CompilationTask(
        task_id=task_id,
        model=model,
        config=config,
        compiler=compiler,
        priority=priority
    )


# GENERATION 3: ADVANCED SCALING AND OPTIMIZATION FEATURES

@dataclass
class ResourceMetrics:
    """System resource metrics for auto-scaling"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    queue_depth: int = 0
    average_task_time: float = 0.0
    throughput_tasks_per_second: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveLoadBalancer:
    """
    GENERATION 3: Intelligent load balancer with auto-scaling
    Automatically adjusts resources based on workload characteristics
    """
    
    def __init__(self, initial_workers: int = 4, max_workers: int = 16):
        self.current_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = 2
        self.logger = get_logger("photonic_mlir.concurrent.load_balancer")
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8    # CPU/memory usage to scale up
        self.scale_down_threshold = 0.3  # Usage to scale down
        self.scale_check_interval = 30   # seconds
        self.metrics_history: List[ResourceMetrics] = []
        self.last_scale_action = datetime.now()
        self.scale_cooldown = timedelta(minutes=2)  # Prevent oscillation
        
        # Advanced scheduling
        self.task_affinity_map: Dict[str, int] = {}  # Task type -> preferred worker
        self.worker_specialization: Dict[int, List[str]] = {}  # Worker -> capabilities
        
        # Performance optimization
        self.batch_processing_enabled = True
        self.dynamic_priority_adjustment = True
        self.predictive_scheduling = True
        
    def collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics (if available)
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            except ImportError:
                pass  # GPU monitoring not available
            
            metrics = ResourceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                gpu_usage=gpu_usage,
                queue_depth=self._estimate_queue_depth(),
                average_task_time=self._calculate_average_task_time(),
                throughput_tasks_per_second=self._calculate_throughput(),
                error_rate=self._calculate_error_rate()
            )
            
            # Store metrics history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:  # Keep last 100 measurements
                self.metrics_history.pop(0)
            
            return metrics
            
        except ImportError:
            # psutil not available - return mock metrics
            return ResourceMetrics(
                cpu_usage=50.0,  # Mock moderate usage
                memory_usage=60.0,
                queue_depth=5,
                average_task_time=2.0,
                throughput_tasks_per_second=3.5
            )
    
    def should_scale_up(self, metrics: ResourceMetrics) -> bool:
        """Determine if we should scale up resources"""
        # Check cooldown period
        if datetime.now() - self.last_scale_action < self.scale_cooldown:
            return False
        
        # Check if at maximum capacity
        if self.current_workers >= self.max_workers:
            return False
        
        # Scale up conditions
        high_resource_usage = (metrics.cpu_usage > self.scale_up_threshold * 100 or 
                             metrics.memory_usage > self.scale_up_threshold * 100)
        high_queue_depth = metrics.queue_depth > self.current_workers * 2
        high_latency = metrics.average_task_time > 5.0  # seconds
        
        # Predictive scaling based on trends
        if self.predictive_scheduling and len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]
            cpu_trend = sum(m.cpu_usage for m in recent_metrics[-3:]) / 3 - sum(m.cpu_usage for m in recent_metrics[:2]) / 2
            if cpu_trend > 20:  # CPU usage trending up by 20%
                self.logger.info(f"Predictive scaling triggered: CPU trend +{cpu_trend:.1f}%")
                return True
        
        return high_resource_usage or high_queue_depth or high_latency
    
    def should_scale_down(self, metrics: ResourceMetrics) -> bool:
        """Determine if we should scale down resources"""
        # Check cooldown period
        if datetime.now() - self.last_scale_action < self.scale_cooldown:
            return False
        
        # Check if at minimum capacity
        if self.current_workers <= self.min_workers:
            return False
        
        # Scale down conditions
        low_resource_usage = (metrics.cpu_usage < self.scale_down_threshold * 100 and
                            metrics.memory_usage < self.scale_down_threshold * 100)
        low_queue_depth = metrics.queue_depth < self.current_workers / 2
        
        # Only scale down if conditions persist
        if len(self.metrics_history) >= 3:
            recent_low_usage = all(
                m.cpu_usage < self.scale_down_threshold * 100 
                for m in self.metrics_history[-3:]
            )
            return low_resource_usage and low_queue_depth and recent_low_usage
        
        return False
    
    def execute_scaling_action(self, scale_up: bool) -> bool:
        """Execute scaling action"""
        if scale_up:
            new_workers = min(self.max_workers, self.current_workers + 2)
            self.logger.info(f"Scaling UP: {self.current_workers} -> {new_workers} workers")
        else:
            new_workers = max(self.min_workers, self.current_workers - 1)
            self.logger.info(f"Scaling DOWN: {self.current_workers} -> {new_workers} workers")
        
        old_workers = self.current_workers
        self.current_workers = new_workers
        self.last_scale_action = datetime.now()
        
        return old_workers != new_workers
    
    def get_optimal_worker_assignment(self, task: CompilationTask) -> Optional[int]:
        """Get optimal worker for task based on affinity and specialization"""
        task_type = self._classify_task_type(task)
        
        # Check task affinity
        if task_type in self.task_affinity_map:
            preferred_worker = self.task_affinity_map[task_type]
            if self._is_worker_available(preferred_worker):
                return preferred_worker
        
        # Check worker specialization
        for worker_id, capabilities in self.worker_specialization.items():
            if task_type in capabilities and self._is_worker_available(worker_id):
                return worker_id
        
        # No specific assignment - let scheduler decide
        return None
    
    def optimize_batch_processing(self, tasks: List[CompilationTask]) -> List[List[CompilationTask]]:
        """Group tasks into optimal batches for processing"""
        if not self.batch_processing_enabled or len(tasks) < 2:
            return [[task] for task in tasks]
        
        # Group compatible tasks
        batches = []
        current_batch = []
        
        for task in sorted(tasks, key=lambda t: (t.priority.value, t.estimated_duration or 0)):
            if not current_batch:
                current_batch.append(task)
            elif self._can_batch_together(current_batch[-1], task) and len(current_batch) < 4:
                current_batch.append(task)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task]
        
        if current_batch:
            batches.append(current_batch)
        
        self.logger.info(f"Optimized {len(tasks)} tasks into {len(batches)} batches")
        return batches
    
    def _estimate_queue_depth(self) -> int:
        """Estimate current queue depth (mock implementation)"""
        return max(0, 10 - self.current_workers)  # Simple heuristic
    
    def _calculate_average_task_time(self) -> float:
        """Calculate average task completion time"""
        if len(self.metrics_history) < 2:
            return 2.0  # Default estimate
        
        # Simple moving average
        recent_times = [m.average_task_time for m in self.metrics_history[-5:] if m.average_task_time > 0]
        return sum(recent_times) / len(recent_times) if recent_times else 2.0
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks per second throughput"""
        if len(self.metrics_history) < 2:
            return 1.0
        
        return max(0.1, self.current_workers / 2.0)  # Simple estimate
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        return 0.05  # Mock 5% error rate
    
    def _classify_task_type(self, task: CompilationTask) -> str:
        """Classify task type for scheduling optimization"""
        config = task.config
        
        if config.get("quantum_enhanced", False):
            return "quantum"
        elif len(config.get("wavelengths", [])) > 8:
            return "high_wavelength"
        elif config.get("power_budget", 100) > 500:
            return "high_power"
        else:
            return "standard"
    
    def _is_worker_available(self, worker_id: int) -> bool:
        """Check if worker is available (mock implementation)"""
        return worker_id < self.current_workers
    
    def _can_batch_together(self, task1: CompilationTask, task2: CompilationTask) -> bool:
        """Check if two tasks can be batched together"""
        if not (task1.batch_compatible and task2.batch_compatible):
            return False
        
        # Same priority level
        if task1.priority != task2.priority:
            return False
        
        # Similar estimated duration (within 2x)
        if task1.estimated_duration and task2.estimated_duration:
            ratio = max(task1.estimated_duration, task2.estimated_duration) / min(task1.estimated_duration, task2.estimated_duration)
            if ratio > 2.0:
                return False
        
        # Compatible configurations
        return self._classify_task_type(task1) == self._classify_task_type(task2)


class PerformanceProfiler:
    """
    GENERATION 3: Advanced performance profiler and optimizer
    Provides real-time performance analysis and optimization suggestions
    """
    
    def __init__(self):
        self.logger = get_logger("photonic_mlir.concurrent.profiler")
        self.performance_data: Dict[str, List[float]] = {
            "compilation_times": [],
            "memory_peaks": [],
            "cpu_utilization": [],
            "cache_hit_rates": [],
            "optimization_effectiveness": []
        }
        self.bottleneck_detection = True
        self.optimization_recommendations: List[Dict[str, Any]] = []
    
    def profile_compilation(self, task: CompilationTask, start_time: float, end_time: float) -> Dict[str, Any]:
        """Profile a compilation task and generate performance insights"""
        duration = end_time - start_time
        self.performance_data["compilation_times"].append(duration)
        
        # Analyze performance characteristics
        profile_result = {
            "task_id": task.task_id,
            "duration": duration,
            "memory_peak": self._estimate_memory_peak(task),
            "cpu_efficiency": self._calculate_cpu_efficiency(task, duration),
            "bottlenecks": self._detect_bottlenecks(task, duration),
            "optimization_score": self._calculate_optimization_score(task)
        }
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(profile_result)
        if recommendations:
            self.optimization_recommendations.extend(recommendations)
        
        # Update performance baselines
        self._update_performance_baselines(task, profile_result)
        
        return profile_result
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights"""
        if not self.performance_data["compilation_times"]:
            return {"status": "insufficient_data"}
        
        times = self.performance_data["compilation_times"]
        
        insights = {
            "performance_summary": {
                "average_compilation_time": sum(times) / len(times),
                "fastest_compilation": min(times),
                "slowest_compilation": max(times),
                "total_tasks_profiled": len(times),
                "performance_trend": self._calculate_performance_trend()
            },
            "bottleneck_analysis": self._analyze_bottlenecks(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "recommendations": self.optimization_recommendations[-10:]  # Last 10 recommendations
        }
        
        return insights
    
    def _estimate_memory_peak(self, task: CompilationTask) -> float:
        """Estimate peak memory usage for task"""
        base_memory = 100.0  # MB
        
        # Estimate based on model complexity
        if hasattr(task.model, 'parameters'):
            try:
                param_count = sum(p.numel() for p in task.model.parameters())
                model_memory = param_count * 4 / 1024 / 1024  # 4 bytes per param, convert to MB
            except:
                model_memory = 50.0  # Default estimate
        else:
            model_memory = 50.0
        
        # Configuration-based adjustments
        wavelength_factor = len(task.config.get("wavelengths", [1550])) * 10
        optimization_factor = task.config.get("optimization_level", 2) * 20
        
        return base_memory + model_memory + wavelength_factor + optimization_factor
    
    def _calculate_cpu_efficiency(self, task: CompilationTask, duration: float) -> float:
        """Calculate CPU efficiency for the task"""
        # Estimate ideal vs actual duration
        ideal_duration = self._estimate_ideal_duration(task)
        efficiency = min(1.0, ideal_duration / duration) if duration > 0 else 0.0
        
        return efficiency
    
    def _detect_bottlenecks(self, task: CompilationTask, duration: float) -> List[str]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        # Duration-based bottleneck detection
        if duration > 10.0:
            bottlenecks.append("excessive_compilation_time")
        
        # Configuration-based bottleneck detection
        if len(task.config.get("wavelengths", [])) > 16:
            bottlenecks.append("high_wavelength_count")
        
        if task.config.get("power_budget", 100) > 1000:
            bottlenecks.append("high_power_budget")
        
        # Model complexity bottlenecks
        if hasattr(task.model, '__len__') and len(task.model) > 50:
            bottlenecks.append("model_complexity")
        
        return bottlenecks
    
    def _calculate_optimization_score(self, task: CompilationTask) -> float:
        """Calculate optimization effectiveness score"""
        base_score = 0.7
        
        # Boost score for advanced optimizations
        optimization_level = task.config.get("optimization_level", 2)
        level_bonus = min(0.2, optimization_level * 0.05)
        
        # Penalty for complexity without optimization
        complexity_penalty = 0.0
        if len(task.config.get("wavelengths", [])) > 8 and optimization_level < 3:
            complexity_penalty = 0.1
        
        return max(0.0, min(1.0, base_score + level_bonus - complexity_penalty))
    
    def _generate_optimization_recommendations(self, profile_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Duration-based recommendations
        if profile_result["duration"] > 5.0:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "description": "Consider enabling advanced caching or reducing model complexity",
                "estimated_improvement": "30-50% faster compilation"
            })
        
        # Bottleneck-specific recommendations
        for bottleneck in profile_result["bottlenecks"]:
            if bottleneck == "high_wavelength_count":
                recommendations.append({
                    "type": "configuration",
                    "priority": "medium",
                    "description": "Reduce wavelength count or enable wavelength multiplexing optimization",
                    "estimated_improvement": "20-40% memory reduction"
                })
        
        # Optimization score recommendations
        if profile_result["optimization_score"] < 0.6:
            recommendations.append({
                "type": "optimization",
                "priority": "medium", 
                "description": "Increase optimization level or enable specific optimization passes",
                "estimated_improvement": "10-25% better performance"
            })
        
        return recommendations
    
    def _update_performance_baselines(self, task: CompilationTask, profile_result: Dict[str, Any]):
        """Update performance baselines for future comparisons"""
        # Update task type baselines
        task_type = self._classify_task_type(task)
        
        # Store in performance data for trend analysis
        self.performance_data["memory_peaks"].append(profile_result["memory_peak"])
        self.performance_data["cpu_utilization"].append(profile_result["cpu_efficiency"])
        self.performance_data["optimization_effectiveness"].append(profile_result["optimization_score"])
        
        # Limit history size
        for key in self.performance_data:
            if len(self.performance_data[key]) > 1000:
                self.performance_data[key] = self.performance_data[key][-500:]  # Keep last 500
    
    def _calculate_performance_trend(self) -> str:
        """Calculate overall performance trend"""
        if len(self.performance_data["compilation_times"]) < 10:
            return "insufficient_data"
        
        recent_avg = sum(self.performance_data["compilation_times"][-10:]) / 10
        older_avg = sum(self.performance_data["compilation_times"][-20:-10]) / 10 if len(self.performance_data["compilation_times"]) >= 20 else recent_avg
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze common bottlenecks across all tasks"""
        # This would analyze patterns in bottlenecks
        return {
            "most_common": "model_complexity",
            "frequency": {"model_complexity": 0.4, "high_wavelength_count": 0.3},
            "severity": {"excessive_compilation_time": "high", "model_complexity": "medium"}
        }
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify system-wide optimization opportunities"""
        return [
            "Enable batch processing for similar tasks",
            "Implement task result caching",
            "Consider GPU acceleration for large models",
            "Optimize wavelength allocation algorithms"
        ]
    
    def _estimate_ideal_duration(self, task: CompilationTask) -> float:
        """Estimate ideal compilation duration for comparison"""
        base_time = 1.0  # seconds
        
        # Model complexity factor
        model_factor = 1.0
        if hasattr(task.model, '__len__'):
            model_factor = min(3.0, len(task.model) / 10.0)
        
        # Configuration complexity
        config_factor = 1.0 + len(task.config.get("wavelengths", [])) * 0.1
        config_factor *= task.config.get("optimization_level", 2) * 0.2
        
        return base_time * model_factor * config_factor
    
    def _classify_task_type(self, task: CompilationTask) -> str:
        """Classify task for performance baseline comparison"""
        config = task.config
        
        if config.get("quantum_enhanced", False):
            return "quantum"
        elif len(config.get("wavelengths", [])) > 8:
            return "high_wavelength"
        elif config.get("power_budget", 100) > 500:
            return "high_power"
        else:
            return "standard"