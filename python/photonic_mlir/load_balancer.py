"""
Advanced load balancing and auto-scaling for distributed photonic compilation.
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import statistics
from datetime import datetime, timedelta

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .logging_config import get_logger
from .concurrent import CompilationTask, TaskStatus, TaskPriority
from .monitoring import get_metrics_collector, PerformanceMetrics


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"


class NodeStatus(Enum):
    """Node status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy" 
    OFFLINE = "offline"


@dataclass
class WorkerNode:
    """Represents a computation worker node"""
    node_id: str
    host: str = "localhost"
    port: int = 8080
    weight: float = 1.0
    max_concurrent_tasks: int = 4
    
    # Dynamic metrics
    current_load: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_response_time: float = 0.0
    avg_response_time: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    last_health_check: Optional[datetime] = None
    
    # Performance history
    response_times: List[float] = field(default_factory=lambda: [])
    cpu_usage_history: List[float] = field(default_factory=lambda: [])
    memory_usage_history: List[float] = field(default_factory=lambda: [])
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.last_response_time = response_time
        self.response_times.append(response_time)
        
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        if self.response_times:
            self.avg_response_time = statistics.mean(self.response_times)
    
    def calculate_load_score(self) -> float:
        """Calculate current load score (0.0 = no load, 1.0+ = overloaded)"""
        task_load = self.active_tasks / self.max_concurrent_tasks
        
        # Factor in response time degradation
        response_penalty = 0.0
        if self.response_times:
            baseline_time = 1.0  # 1 second baseline
            current_avg = self.avg_response_time
            if current_avg > baseline_time:
                response_penalty = (current_avg - baseline_time) / baseline_time * 0.5
        
        return min(task_load + response_penalty, 2.0)
    
    def is_available(self) -> bool:
        """Check if node is available for new tasks"""
        return (self.status == NodeStatus.HEALTHY and 
                self.active_tasks < self.max_concurrent_tasks)


class HealthChecker:
    """Health checking for worker nodes"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = get_logger("photonic_mlir.load_balancer.health_checker")
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
    
    def start(self, nodes: Dict[str, WorkerNode]):
        """Start health checking"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._health_check_loop, 
            args=(nodes,),
            daemon=True
        )
        self._check_thread.start()
        self.logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        self.logger.info("Health checker stopped")
    
    def _health_check_loop(self, nodes: Dict[str, WorkerNode]):
        """Main health checking loop"""
        while self._running:
            try:
                for node in nodes.values():
                    self._check_node_health(node)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(5.0)
    
    def _check_node_health(self, node: WorkerNode):
        """Check health of individual node"""
        try:
            # Mock health check - in reality would ping the node
            start_time = time.time()
            
            # Simulate health check based on current load
            if node.calculate_load_score() > 1.5:
                # Overloaded
                node.status = NodeStatus.DEGRADED
                response_time = 2.0
            elif node.calculate_load_score() > 0.8:
                # High load
                node.status = NodeStatus.HEALTHY
                response_time = 1.0
            else:
                # Normal
                node.status = NodeStatus.HEALTHY
                response_time = 0.5
            
            # Add some randomness
            if NUMPY_AVAILABLE:
                response_time += np.random.normal(0, 0.1)
                response_time = max(0.1, response_time)
            
            node.last_health_check = datetime.now()
            
            # Log status changes
            if hasattr(node, '_last_logged_status') and node._last_logged_status != node.status:
                self.logger.info(f"Node {node.node_id} status changed: {node._last_logged_status} -> {node.status}")
            
            node._last_logged_status = node.status
            
        except Exception as e:
            self.logger.error(f"Failed to check health of node {node.node_id}: {e}")
            node.status = NodeStatus.UNHEALTHY


class LoadBalancer:
    """Advanced load balancer with multiple strategies and auto-scaling"""
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 health_check_interval: float = 30.0):
        self.strategy = strategy
        self.nodes: Dict[str, WorkerNode] = {}
        self.logger = get_logger("photonic_mlir.load_balancer")
        
        # Round robin counter
        self._round_robin_index = 0
        self._selection_lock = threading.RLock()
        
        # Health checking
        self.health_checker = HealthChecker(health_check_interval)
        
        # Auto-scaling parameters
        self.auto_scaling_enabled = True
        self.scale_up_threshold = 0.8  # Average load threshold for scaling up
        self.scale_down_threshold = 0.3  # Average load threshold for scaling down
        self.min_nodes = 1
        self.max_nodes = 8
        
        # Metrics
        self.total_tasks_assigned = 0
        self.task_assignment_times: List[float] = []
        
        self.logger.info(f"Load balancer initialized with {strategy.value} strategy")
    
    def add_node(self, node: WorkerNode):
        """Add a worker node"""
        self.nodes[node.node_id] = node
        self.logger.info(f"Added node {node.node_id} with weight {node.weight}")
        
        # Start health checker if this is the first node
        if len(self.nodes) == 1:
            self.health_checker.start(self.nodes)
    
    def remove_node(self, node_id: str):
        """Remove a worker node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id}")
    
    def select_node(self, task: CompilationTask) -> Optional[WorkerNode]:
        """Select optimal node for task assignment"""
        with self._selection_lock:
            start_time = time.time()
            
            available_nodes = [node for node in self.nodes.values() if node.is_available()]
            
            if not available_nodes:
                self.logger.warning("No available nodes for task assignment")
                return None
            
            selected_node = None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected_node = self._select_round_robin(available_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                selected_node = self._select_least_loaded(available_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected_node = self._select_weighted_round_robin(available_nodes)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                selected_node = self._select_best_response_time(available_nodes)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                selected_node = self._select_adaptive(available_nodes, task)
            
            if selected_node:
                selected_node.active_tasks += 1
                self.total_tasks_assigned += 1
                
                selection_time = time.time() - start_time
                self.task_assignment_times.append(selection_time)
                
                self.logger.debug(f"Selected node {selected_node.node_id} for task {task.task_id}")
            
            return selected_node
    
    def _select_round_robin(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round robin selection"""
        node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return node
    
    def _select_least_loaded(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least load"""
        return min(nodes, key=lambda n: n.calculate_load_score())
    
    def _select_weighted_round_robin(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin based on node weights"""
        # Simplified weighted selection
        total_weight = sum(node.weight for node in nodes)
        weights = [node.weight / total_weight for node in nodes]
        
        if NUMPY_AVAILABLE:
            idx = np.random.choice(len(nodes), p=weights)
        else:
            # Fallback without numpy
            import random
            rand_val = random.random()
            cumulative = 0.0
            for i, weight in enumerate(weights):
                cumulative += weight
                if rand_val <= cumulative:
                    idx = i
                    break
            else:
                idx = 0
        
        return nodes[idx]
    
    def _select_best_response_time(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with best average response time"""
        return min(nodes, key=lambda n: n.avg_response_time or float('inf'))
    
    def _select_adaptive(self, nodes: List[WorkerNode], task: CompilationTask) -> WorkerNode:
        """Adaptive selection based on multiple factors"""
        def score_node(node: WorkerNode) -> float:
            load_score = node.calculate_load_score()
            response_score = (node.avg_response_time or 1.0) / 5.0  # Normalize to ~0-1
            
            # Task priority factor
            priority_factor = 1.0
            if task.priority == TaskPriority.HIGH:
                priority_factor = 0.8  # Prefer less loaded nodes for high priority
            elif task.priority == TaskPriority.CRITICAL:
                priority_factor = 0.6
            
            # Weight factor
            weight_factor = 1.0 / node.weight if node.weight > 0 else 1.0
            
            # Combined score (lower is better)
            return (load_score + response_score) * priority_factor * weight_factor
        
        return min(nodes, key=score_node)
    
    def task_completed(self, node_id: str, task: CompilationTask, response_time: float):
        """Notify that a task has completed on a node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.active_tasks = max(0, node.active_tasks - 1)
            node.update_response_time(response_time)
            
            if task.status == TaskStatus.COMPLETED:
                node.completed_tasks += 1
            elif task.status == TaskStatus.FAILED:
                node.failed_tasks += 1
            
            self.logger.debug(f"Task {task.task_id} completed on node {node_id}")
            
            # Check for auto-scaling opportunities
            if self.auto_scaling_enabled:
                self._check_auto_scaling()
    
    def _check_auto_scaling(self):
        """Check if auto-scaling is needed"""
        if not self.nodes:
            return
        
        avg_load = statistics.mean(node.calculate_load_score() for node in self.nodes.values())
        
        # Scale up if average load is high
        if avg_load > self.scale_up_threshold and len(self.nodes) < self.max_nodes:
            self._scale_up()
        
        # Scale down if average load is low
        elif avg_load < self.scale_down_threshold and len(self.nodes) > self.min_nodes:
            self._scale_down()
    
    def _scale_up(self):
        """Add new worker node"""
        new_node_id = f"worker_{len(self.nodes) + 1}_{int(time.time())}"
        new_node = WorkerNode(
            node_id=new_node_id,
            host="localhost",
            port=8080 + len(self.nodes),
            weight=1.0,
            max_concurrent_tasks=4
        )
        
        self.add_node(new_node)
        self.logger.info(f"Scaled up: Added node {new_node_id}")
    
    def _scale_down(self):
        """Remove least utilized worker node"""
        if len(self.nodes) <= self.min_nodes:
            return
        
        # Find node with lowest utilization
        least_utilized = min(
            self.nodes.values(),
            key=lambda n: n.active_tasks + n.calculate_load_score()
        )
        
        # Only remove if it has no active tasks
        if least_utilized.active_tasks == 0:
            self.remove_node(least_utilized.node_id)
            self.logger.info(f"Scaled down: Removed node {least_utilized.node_id}")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        if not self.nodes:
            return {"node_count": 0}
        
        node_stats = []
        total_active_tasks = 0
        total_completed_tasks = 0
        total_failed_tasks = 0
        load_scores = []
        response_times = []
        
        for node in self.nodes.values():
            node_stats.append({
                "node_id": node.node_id,
                "status": node.status.value,
                "active_tasks": node.active_tasks,
                "completed_tasks": node.completed_tasks,
                "failed_tasks": node.failed_tasks,
                "load_score": node.calculate_load_score(),
                "avg_response_time": node.avg_response_time,
                "weight": node.weight
            })
            
            total_active_tasks += node.active_tasks
            total_completed_tasks += node.completed_tasks
            total_failed_tasks += node.failed_tasks
            load_scores.append(node.calculate_load_score())
            if node.avg_response_time > 0:
                response_times.append(node.avg_response_time)
        
        cluster_stats = {
            "node_count": len(self.nodes),
            "total_active_tasks": total_active_tasks,
            "total_completed_tasks": total_completed_tasks,
            "total_failed_tasks": total_failed_tasks,
            "total_tasks_assigned": self.total_tasks_assigned,
            "avg_cluster_load": statistics.mean(load_scores) if load_scores else 0,
            "avg_cluster_response_time": statistics.mean(response_times) if response_times else 0,
            "strategy": self.strategy.value,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "nodes": node_stats
        }
        
        return cluster_stats
    
    def shutdown(self):
        """Shutdown load balancer"""
        self.health_checker.stop()
        self.logger.info("Load balancer shutdown")


class DistributedCompiler:
    """Distributed compiler using load balancer"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, CompilationTask] = {}
        self.completed_tasks: Dict[str, CompilationTask] = {}
        
        self.logger = get_logger("photonic_mlir.distributed_compiler")
        
        # Start task dispatcher
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._start_dispatcher()
    
    def _start_dispatcher(self):
        """Start task dispatcher thread"""
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_tasks, daemon=True)
        self._dispatcher_thread.start()
        self.logger.info("Task dispatcher started")
    
    def submit_task(self, task: CompilationTask) -> str:
        """Submit compilation task for distributed execution"""
        self.task_queue.put(task)
        self.logger.debug(f"Submitted task {task.task_id} to distributed compiler")
        return task.task_id
    
    def _dispatch_tasks(self):
        """Main task dispatching loop"""
        while self._running:
            try:
                # Get next task from queue
                task = self.task_queue.get(timeout=1.0)
                
                # Select node for execution
                selected_node = self.load_balancer.select_node(task)
                
                if selected_node:
                    # Execute task on selected node
                    self._execute_on_node(task, selected_node)
                else:
                    # No available nodes, put task back in queue
                    self.task_queue.put(task)
                    time.sleep(0.5)  # Brief delay before retry
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Task dispatch error: {e}")
    
    def _execute_on_node(self, task: CompilationTask, node: WorkerNode):
        """Execute task on specific node"""
        try:
            self.active_tasks[task.task_id] = task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Mock task execution
            start_time = time.time()
            
            # Simulate variable execution time based on node load
            execution_time = 1.0 + node.calculate_load_score() * 0.5
            if NUMPY_AVAILABLE:
                execution_time += np.random.exponential(0.5)
            
            time.sleep(min(execution_time, 3.0))  # Cap at 3 seconds for demo
            
            # Mock successful completion
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = f"Compiled on node {node.node_id}"
            
            response_time = time.time() - start_time
            
            # Notify load balancer
            self.load_balancer.task_completed(node.node_id, task, response_time)
            
            # Move to completed tasks
            del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.info(f"Task {task.task_id} completed on node {node.node_id} in {response_time:.2f}s")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            response_time = time.time() - start_time if 'start_time' in locals() else 0
            self.load_balancer.task_completed(node.node_id, task, response_time)
            
            del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.error(f"Task {task.task_id} failed on node {node.node_id}: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[CompilationTask]:
        """Get status of specific task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed compiler statistics"""
        return {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "load_balancer": self.load_balancer.get_cluster_stats()
        }
    
    def shutdown(self):
        """Shutdown distributed compiler"""
        self._running = False
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)
        
        self.load_balancer.shutdown()
        self.logger.info("Distributed compiler shutdown")


# Factory functions
def create_local_cluster(num_nodes: int = 4, 
                        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE) -> DistributedCompiler:
    """Create a local cluster for distributed compilation"""
    load_balancer = LoadBalancer(strategy)
    
    # Add worker nodes
    for i in range(num_nodes):
        node = WorkerNode(
            node_id=f"worker_{i}",
            host="localhost",
            port=8080 + i,
            weight=1.0,
            max_concurrent_tasks=4
        )
        load_balancer.add_node(node)
    
    return DistributedCompiler(load_balancer)


def create_heterogeneous_cluster() -> DistributedCompiler:
    """Create cluster with different node capabilities"""
    load_balancer = LoadBalancer(LoadBalancingStrategy.ADAPTIVE)
    
    # Add different types of nodes
    nodes = [
        WorkerNode("high_perf", "localhost", 8080, weight=2.0, max_concurrent_tasks=8),
        WorkerNode("standard_1", "localhost", 8081, weight=1.0, max_concurrent_tasks=4),
        WorkerNode("standard_2", "localhost", 8082, weight=1.0, max_concurrent_tasks=4),
        WorkerNode("low_power", "localhost", 8083, weight=0.5, max_concurrent_tasks=2),
    ]
    
    for node in nodes:
        load_balancer.add_node(node)
    
    return DistributedCompiler(load_balancer)


if __name__ == "__main__":
    # Demo usage
    compiler = create_local_cluster(num_nodes=3)
    
    # Submit some tasks
    from .concurrent import create_compilation_task, TaskPriority
    
    for i in range(10):
        task = create_compilation_task(
            model=None,
            config={"optimization_level": 2},
            compiler=None,
            priority=TaskPriority.NORMAL
        )
        compiler.submit_task(task)
    
    # Let tasks execute
    time.sleep(5)
    
    # Print stats
    stats = compiler.get_stats()
    print(f"Cluster stats: {stats}")
    
    compiler.shutdown()