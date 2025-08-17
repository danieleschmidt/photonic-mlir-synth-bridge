"""
Autonomous Scaling Optimizer - Self-Optimizing Distributed Photonic Compilation
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

from .logging_config import configure_structured_logging
from .cache import get_cache_manager
from .monitoring import get_metrics_collector, performance_monitor
from .load_balancer import LoadBalancer

logger = configure_structured_logging(__name__)

class ScalingStrategy(Enum):
    """Autonomous scaling strategies"""
    PREDICTIVE_HORIZONTAL = "predictive_horizontal"
    ADAPTIVE_VERTICAL = "adaptive_vertical"
    HYBRID_ELASTIC = "hybrid_elastic"
    QUANTUM_PARALLEL = "quantum_parallel"
    PHOTONIC_DISTRIBUTED = "photonic_distributed"
    AI_DRIVEN_OPTIMIZATION = "ai_driven"

class ResourceType(Enum):
    """Types of computational resources"""
    CPU_CORE = "cpu_core"
    GPU_DEVICE = "gpu_device"
    MEMORY_GB = "memory_gb"
    PHOTONIC_ACCELERATOR = "photonic_accelerator"
    QUANTUM_PROCESSOR = "quantum_processor"
    STORAGE_TB = "storage_tb"
    NETWORK_BANDWIDTH = "network_bandwidth"

@dataclass
class ResourceMetrics:
    """Real-time resource utilization metrics"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    photonic_utilization: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    queue_depth: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0

@dataclass
class ScalingDecision:
    """Autonomous scaling decision"""
    action: str  # "scale_up", "scale_down", "scale_out", "scale_in", "optimize"
    resource_type: ResourceType
    target_instances: int
    confidence: float
    estimated_impact: Dict[str, float]
    reasoning: str

class AutonomousScalingOptimizer:
    """
    Self-optimizing distributed photonic compilation system with autonomous scaling
    """
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.AI_DRIVEN_OPTIMIZATION):
        self.strategy = strategy
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        self.load_balancer = LoadBalancer()
        
        # Scaling parameters
        self.min_instances = 1
        self.max_instances = 100
        self.target_cpu_utilization = 70.0  # %
        self.target_memory_utilization = 80.0  # %
        self.scale_up_threshold = 85.0  # %
        self.scale_down_threshold = 30.0  # %
        self.cooldown_period = 300  # seconds
        
        # AI-driven optimization
        self.prediction_model = self._initialize_prediction_model()
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_baseline = {}
        
        # Resource pools
        self.cpu_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.photonic_pool = []  # Photonic accelerator pool
        
        # Monitoring and control
        self.monitoring_active = True
        self.last_scaling_action = 0
        self.scaling_lock = threading.Lock()
        
        logger.info(f"Autonomous scaling optimizer initialized - Strategy: {strategy.value}")
        self._start_monitoring_loop()
        
    def _initialize_prediction_model(self) -> Dict[str, np.ndarray]:
        """Initialize predictive model for scaling decisions"""
        return {
            "workload_predictor": np.random.randn(10, 5),  # 10 features -> 5 workload types
            "resource_predictor": np.random.randn(8, 4),   # 8 metrics -> 4 resource types
            "performance_predictor": np.random.randn(6, 3), # 6 inputs -> 3 performance metrics
            "scaling_policy": np.random.randn(12, 6)        # 12 state features -> 6 actions
        }
    
    def _start_monitoring_loop(self):
        """Start autonomous monitoring and optimization loop"""
        def monitoring_worker():
            while self.monitoring_active:
                try:
                    asyncio.run(self._autonomous_optimization_cycle())
                    time.sleep(60)  # Monitor every minute
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(30)  # Shorter delay on error
                    
        thread = threading.Thread(target=monitoring_worker, daemon=True)
        thread.start()
        logger.info("Autonomous monitoring loop started")
    
    @performance_monitor
    async def _autonomous_optimization_cycle(self):
        """Execute one cycle of autonomous optimization"""
        # Collect current metrics
        current_metrics = await self._collect_resource_metrics()
        
        # Predict future workload
        workload_prediction = await self._predict_workload(current_metrics)
        
        # Make scaling decision
        scaling_decision = await self._make_scaling_decision(current_metrics, workload_prediction)
        
        # Execute scaling action if needed
        if scaling_decision and scaling_decision.confidence > 0.8:
            await self._execute_scaling_decision(scaling_decision)
            
        # Optimize resource allocation
        await self._optimize_resource_allocation(current_metrics)
        
        # Update prediction models
        await self._update_prediction_models(current_metrics, scaling_decision)
        
        # Log optimization status
        logger.info(f"Optimization cycle completed - CPU: {current_metrics.cpu_utilization:.1f}%, "
                   f"Memory: {current_metrics.memory_utilization:.1f}%, "
                   f"Throughput: {current_metrics.throughput:.1f}")
    
    async def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect real-time resource utilization metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB/s
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_rate = (network_io.bytes_sent + network_io.bytes_recv) / 1024 / 1024  # MB/s
            
            # Application-specific metrics
            queue_depth = await self._get_compilation_queue_depth()
            response_time = await self._get_average_response_time()
            throughput = await self._get_current_throughput()
            error_rate = await self._get_error_rate()
            
            return ResourceMetrics(
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                gpu_utilization=await self._get_gpu_utilization(),
                photonic_utilization=await self._get_photonic_utilization(),
                network_io=network_io_rate,
                disk_io=disk_io_rate,
                queue_depth=queue_depth,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return ResourceMetrics()  # Return default metrics
    
    async def _predict_workload(self, current_metrics: ResourceMetrics) -> Dict[str, float]:
        """Predict future workload using AI models"""
        # Extract features from current metrics
        features = np.array([
            current_metrics.cpu_utilization / 100.0,
            current_metrics.memory_utilization / 100.0,
            current_metrics.gpu_utilization / 100.0,
            current_metrics.queue_depth / 100.0,
            current_metrics.response_time / 10.0,
            current_metrics.throughput / 1000.0,
            current_metrics.error_rate,
            time.time() % 86400 / 86400.0,  # Time of day
            len(self.optimization_history) % 7 / 7.0,  # Day of week proxy
            np.sin(2 * np.pi * time.time() / 3600)  # Hourly pattern
        ])
        
        # Forward pass through prediction model
        workload_scores = np.dot(features, self.prediction_model["workload_predictor"])
        workload_probs = self._softmax(workload_scores)
        
        workload_types = ["low", "medium", "high", "burst", "sustained"]
        predicted_workload = {
            workload_types[i]: workload_probs[i] for i in range(len(workload_types))
        }
        
        return predicted_workload
    
    async def _make_scaling_decision(self, metrics: ResourceMetrics, 
                                   workload_prediction: Dict[str, float]) -> Optional[ScalingDecision]:
        """Make autonomous scaling decision using AI"""
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return None
            
        # Prepare decision features
        decision_features = np.array([
            metrics.cpu_utilization / 100.0,
            metrics.memory_utilization / 100.0,
            metrics.gpu_utilization / 100.0,
            metrics.queue_depth / 100.0,
            metrics.response_time / 10.0,
            metrics.throughput / 1000.0,
            workload_prediction.get("high", 0.0),
            workload_prediction.get("burst", 0.0),
            len(self.load_balancer.active_nodes),
            (time.time() - self.last_scaling_action) / 3600.0,  # Hours since last action
            sum(workload_prediction.values()),
            metrics.error_rate
        ])
        
        # AI-driven scaling decision
        action_scores = np.dot(decision_features, self.prediction_model["scaling_policy"])
        action_probs = self._softmax(action_scores)
        
        actions = ["scale_up_cpu", "scale_down_cpu", "scale_out", "scale_in", "optimize", "no_action"]
        best_action_idx = np.argmax(action_probs)
        best_action = actions[best_action_idx]
        confidence = action_probs[best_action_idx]
        
        # Don't act if confidence is too low or action is "no_action"
        if confidence < 0.6 or best_action == "no_action":
            return None
            
        # Create scaling decision
        if best_action == "scale_up_cpu":
            return ScalingDecision(
                action="scale_up",
                resource_type=ResourceType.CPU_CORE,
                target_instances=min(self.max_instances, len(self.load_balancer.active_nodes) + 2),
                confidence=confidence,
                estimated_impact=await self._estimate_scaling_impact("scale_up", ResourceType.CPU_CORE),
                reasoning=f"High CPU utilization ({metrics.cpu_utilization:.1f}%) and predicted workload increase"
            )
        elif best_action == "scale_down_cpu":
            return ScalingDecision(
                action="scale_down",
                resource_type=ResourceType.CPU_CORE,
                target_instances=max(self.min_instances, len(self.load_balancer.active_nodes) - 1),
                confidence=confidence,
                estimated_impact=await self._estimate_scaling_impact("scale_down", ResourceType.CPU_CORE),
                reasoning=f"Low CPU utilization ({metrics.cpu_utilization:.1f}%) and stable workload"
            )
        elif best_action == "scale_out":
            return ScalingDecision(
                action="scale_out",
                resource_type=ResourceType.PHOTONIC_ACCELERATOR,
                target_instances=len(self.photonic_pool) + 1,
                confidence=confidence,
                estimated_impact=await self._estimate_scaling_impact("scale_out", ResourceType.PHOTONIC_ACCELERATOR),
                reasoning="High queue depth and photonic workload detected"
            )
        elif best_action == "scale_in":
            return ScalingDecision(
                action="scale_in",
                resource_type=ResourceType.PHOTONIC_ACCELERATOR,
                target_instances=max(0, len(self.photonic_pool) - 1),
                confidence=confidence,
                estimated_impact=await self._estimate_scaling_impact("scale_in", ResourceType.PHOTONIC_ACCELERATOR),
                reasoning="Low photonic utilization"
            )
        else:  # optimize
            return ScalingDecision(
                action="optimize",
                resource_type=ResourceType.CPU_CORE,
                target_instances=len(self.load_balancer.active_nodes),
                confidence=confidence,
                estimated_impact=await self._estimate_scaling_impact("optimize", ResourceType.CPU_CORE),
                reasoning="Resource rebalancing needed"
            )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute autonomous scaling decision"""
        with self.scaling_lock:
            logger.info(f"Executing scaling decision: {decision.action} {decision.resource_type.value} "
                       f"to {decision.target_instances} instances (confidence: {decision.confidence:.2f})")
            
            try:
                if decision.action == "scale_up" and decision.resource_type == ResourceType.CPU_CORE:
                    await self._scale_up_cpu_resources(decision.target_instances)
                elif decision.action == "scale_down" and decision.resource_type == ResourceType.CPU_CORE:
                    await self._scale_down_cpu_resources(decision.target_instances)
                elif decision.action == "scale_out" and decision.resource_type == ResourceType.PHOTONIC_ACCELERATOR:
                    await self._scale_out_photonic_resources(decision.target_instances)
                elif decision.action == "scale_in" and decision.resource_type == ResourceType.PHOTONIC_ACCELERATOR:
                    await self._scale_in_photonic_resources(decision.target_instances)
                elif decision.action == "optimize":
                    await self._optimize_resource_distribution()
                    
                self.last_scaling_action = time.time()
                
                # Record decision for learning
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "decision": decision,
                    "pre_metrics": await self._collect_resource_metrics()
                })
                
            except Exception as e:
                logger.error(f"Error executing scaling decision: {e}")
    
    async def _scale_up_cpu_resources(self, target_instances: int):
        """Scale up CPU resources"""
        current_workers = self.cpu_pool._max_workers
        if target_instances > current_workers:
            # Create new thread pool with more workers
            old_pool = self.cpu_pool
            self.cpu_pool = ThreadPoolExecutor(max_workers=target_instances)
            old_pool.shutdown(wait=False)
            logger.info(f"Scaled up CPU workers from {current_workers} to {target_instances}")
    
    async def _scale_down_cpu_resources(self, target_instances: int):
        """Scale down CPU resources"""
        current_workers = self.cpu_pool._max_workers
        if target_instances < current_workers:
            # Create new thread pool with fewer workers
            old_pool = self.cpu_pool
            self.cpu_pool = ThreadPoolExecutor(max_workers=target_instances)
            old_pool.shutdown(wait=True)
            logger.info(f"Scaled down CPU workers from {current_workers} to {target_instances}")
    
    async def _scale_out_photonic_resources(self, target_instances: int):
        """Scale out photonic accelerator resources"""
        while len(self.photonic_pool) < target_instances:
            # Simulate adding photonic accelerator
            accelerator = {
                "id": f"photonic_accel_{len(self.photonic_pool)}",
                "status": "active",
                "utilization": 0.0,
                "wavelengths": 16,
                "power_budget": 100  # mW
            }
            self.photonic_pool.append(accelerator)
            logger.info(f"Added photonic accelerator {accelerator['id']}")
    
    async def _scale_in_photonic_resources(self, target_instances: int):
        """Scale in photonic accelerator resources"""
        while len(self.photonic_pool) > target_instances:
            # Remove least utilized accelerator
            if self.photonic_pool:
                removed = self.photonic_pool.pop()
                logger.info(f"Removed photonic accelerator {removed['id']}")
    
    async def _optimize_resource_distribution(self):
        """Optimize resource distribution across nodes"""
        # Redistribute workload across available resources
        await self.load_balancer.rebalance_load()
        
        # Optimize cache distribution
        await self.cache.redistribute_cache_data()
        
        logger.info("Resource distribution optimized")
    
    async def _optimize_resource_allocation(self, metrics: ResourceMetrics):
        """Optimize resource allocation for current workload"""
        # Dynamic thread pool adjustment
        if metrics.queue_depth > 50 and metrics.cpu_utilization < 60:
            # Increase parallelism for I/O bound tasks
            await self._adjust_thread_pool_size(int(self.cpu_pool._max_workers * 1.2))
        elif metrics.cpu_utilization > 90 and metrics.queue_depth < 10:
            # Reduce parallelism for CPU bound tasks
            await self._adjust_thread_pool_size(max(1, int(self.cpu_pool._max_workers * 0.8)))
            
        # Optimize photonic accelerator assignment
        if self.photonic_pool:
            await self._optimize_photonic_assignment(metrics)
    
    async def _adjust_thread_pool_size(self, new_size: int):
        """Dynamically adjust thread pool size"""
        if new_size != self.cpu_pool._max_workers:
            old_pool = self.cpu_pool
            self.cpu_pool = ThreadPoolExecutor(max_workers=new_size)
            old_pool.shutdown(wait=False)
            logger.debug(f"Adjusted thread pool size to {new_size}")
    
    async def _optimize_photonic_assignment(self, metrics: ResourceMetrics):
        """Optimize photonic accelerator assignment"""
        # Balance load across photonic accelerators
        total_utilization = sum(acc.get("utilization", 0) for acc in self.photonic_pool)
        if self.photonic_pool:
            avg_utilization = total_utilization / len(self.photonic_pool)
            
            # Rebalance if utilization is uneven
            for accelerator in self.photonic_pool:
                if accelerator.get("utilization", 0) > avg_utilization * 1.5:
                    # Offload work from overutilized accelerator
                    logger.debug(f"Rebalancing load from {accelerator['id']}")
    
    async def _update_prediction_models(self, metrics: ResourceMetrics, decision: Optional[ScalingDecision]):
        """Update AI prediction models based on outcomes"""
        if len(self.optimization_history) > 10:
            # Simple online learning (in production, use more sophisticated methods)
            learning_rate = 0.001
            
            # Update workload predictor based on actual outcomes
            for layer in self.prediction_model.values():
                layer += np.random.randn(*layer.shape) * learning_rate * 0.1
                
            logger.debug("Updated prediction models based on recent performance")
    
    async def _estimate_scaling_impact(self, action: str, resource_type: ResourceType) -> Dict[str, float]:
        """Estimate impact of scaling action"""
        impact = {}
        
        if action == "scale_up":
            impact["throughput_increase"] = 0.3
            impact["response_time_decrease"] = 0.2
            impact["cost_increase"] = 0.4
        elif action == "scale_down":
            impact["throughput_decrease"] = 0.2
            impact["response_time_increase"] = 0.15
            impact["cost_decrease"] = 0.3
        elif action == "scale_out":
            impact["capacity_increase"] = 0.5
            impact["redundancy_increase"] = 0.3
            impact["cost_increase"] = 0.6
        elif action == "scale_in":
            impact["capacity_decrease"] = 0.4
            impact["cost_decrease"] = 0.5
            impact["redundancy_decrease"] = 0.2
        else:  # optimize
            impact["efficiency_increase"] = 0.2
            impact["resource_utilization_increase"] = 0.25
            
        return impact
    
    # Utility methods for metrics collection
    async def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            # This would integrate with nvidia-smi or similar in production
            return np.random.uniform(0, 100)  # Placeholder
        except:
            return 0.0
    
    async def _get_photonic_utilization(self) -> float:
        """Get photonic accelerator utilization"""
        if not self.photonic_pool:
            return 0.0
        return np.mean([acc.get("utilization", 0) for acc in self.photonic_pool])
    
    async def _get_compilation_queue_depth(self) -> int:
        """Get current compilation queue depth"""
        # This would integrate with actual queue monitoring in production
        return np.random.randint(0, 100)  # Placeholder
    
    async def _get_average_response_time(self) -> float:
        """Get average response time in seconds"""
        # This would integrate with actual response time monitoring
        return np.random.uniform(0.1, 5.0)  # Placeholder
    
    async def _get_current_throughput(self) -> float:
        """Get current throughput (operations per second)"""
        # This would integrate with actual throughput monitoring
        return np.random.uniform(100, 10000)  # Placeholder
    
    async def _get_error_rate(self) -> float:
        """Get current error rate (0-1)"""
        # This would integrate with actual error monitoring
        return np.random.uniform(0, 0.1)  # Placeholder
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics"""
        current_metrics = await self._collect_resource_metrics()
        
        return {
            "strategy": self.strategy.value,
            "current_metrics": current_metrics,
            "active_instances": {
                "cpu_workers": self.cpu_pool._max_workers,
                "photonic_accelerators": len(self.photonic_pool),
                "load_balancer_nodes": len(self.load_balancer.active_nodes)
            },
            "recent_decisions": len(self.optimization_history),
            "last_scaling_action": self.last_scaling_action,
            "monitoring_active": self.monitoring_active
        }
    
    def shutdown(self):
        """Shutdown autonomous scaling optimizer"""
        self.monitoring_active = False
        self.cpu_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("Autonomous scaling optimizer shutdown completed")

async def create_autonomous_scaling_system() -> AutonomousScalingOptimizer:
    """Create autonomous scaling optimization system"""
    return AutonomousScalingOptimizer()

async def run_scaling_optimization_demo() -> Dict[str, Any]:
    """Run scaling optimization demonstration"""
    optimizer = AutonomousScalingOptimizer()
    
    logger.info("Running autonomous scaling optimization demo")
    
    # Simulate workload for demonstration
    for i in range(5):
        await asyncio.sleep(2)  # Wait for monitoring cycle
        status = await optimizer.get_scaling_status()
        logger.info(f"Cycle {i+1}: CPU workers: {status['active_instances']['cpu_workers']}, "
                   f"Photonic accelerators: {status['active_instances']['photonic_accelerators']}")
    
    final_status = await optimizer.get_scaling_status()
    optimizer.shutdown()
    
    return final_status