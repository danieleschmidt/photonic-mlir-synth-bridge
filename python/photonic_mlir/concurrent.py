"""
Concurrent processing and resource pooling for Photonic MLIR.
"""

import threading
import concurrent.futures
import multiprocessing as mp
import queue
import time
from typing import Any, Dict, List, Optional, Callable, Union, Iterator
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import weakref

from .logging_config import get_logger
from .exceptions import PhotonicMLIRError


class ProcessingStrategy(Enum):
    """Parallel processing strategies"""
    THREAD_BASED = "thread"
    PROCESS_BASED = "process"
    ASYNC_BASED = "async"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a computation task"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[float] = None
    callback: Optional[Callable] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ResourcePool:
    """Generic resource pool with lifecycle management"""
    
    def __init__(self, 
                 resource_factory: Callable,
                 max_size: int = 10,
                 min_size: int = 2,
                 timeout_seconds: float = 30.0):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size
        self.timeout_seconds = timeout_seconds
        
        self._pool = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.RLock()
        self._shutdown = False
        
        self.logger = get_logger("photonic_mlir.concurrent.resource_pool")
        
        # Pre-populate with minimum resources
        self._populate_minimum()
    
    def _populate_minimum(self) -> None:
        """Populate pool with minimum number of resources"""
        with self._lock:
            while self._created_count < self.min_size:
                resource = self._create_resource()
                if resource:
                    self._pool.put(resource)
    
    def _create_resource(self) -> Any:
        """Create a new resource"""
        try:
            resource = self.resource_factory()
            self._created_count += 1
            self.logger.debug(f"Created resource #{self._created_count}")
            return resource
        except Exception as e:
            self.logger.error(f"Failed to create resource: {e}")
            return None
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool"""
        if self._shutdown:
            raise PhotonicMLIRError("Resource pool is shut down")
        
        timeout = timeout or self.timeout_seconds
        
        try:
            # Try to get existing resource
            resource = self._pool.get(timeout=0.1)
            self.logger.debug("Acquired existing resource from pool")
            return resource
        except queue.Empty:
            pass
        
        # Create new resource if under limit
        with self._lock:
            if self._created_count < self.max_size:
                resource = self._create_resource()
                if resource:
                    return resource
        
        # Wait for available resource
        try:
            resource = self._pool.get(timeout=timeout)
            self.logger.debug("Acquired resource after waiting")
            return resource
        except queue.Empty:
            raise PhotonicMLIRError(f"Resource acquisition timeout after {timeout}s")
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool"""
        if self._shutdown:
            return
        
        try:
            self._pool.put(resource, timeout=0.1)
            self.logger.debug("Released resource back to pool")
        except queue.Full:
            # Pool is full, discard resource
            self._cleanup_resource(resource)
            with self._lock:
                self._created_count -= 1
    
    def _cleanup_resource(self, resource: Any) -> None:
        """Cleanup a resource (override for custom cleanup)"""
        if hasattr(resource, 'close'):
            try:
                resource.close()
            except:
                pass
    
    def shutdown(self) -> None:
        """Shutdown the resource pool"""
        self._shutdown = True
        
        # Cleanup all resources
        while True:
            try:
                resource = self._pool.get(timeout=0.1)
                self._cleanup_resource(resource)
            except queue.Empty:
                break
        
        self.logger.info("Resource pool shut down")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "created_count": self._created_count,
            "available_count": self._pool.qsize(),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "shutdown": self._shutdown
        }


class CompilationWorkerPool:
    """Specialized worker pool for compilation tasks"""
    
    def __init__(self, 
                 max_workers: int = None,
                 strategy: ProcessingStrategy = ProcessingStrategy.HYBRID):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.strategy = strategy
        
        self.logger = get_logger("photonic_mlir.concurrent.compilation_pool")
        
        # Initialize executors based on strategy
        self._thread_executor = None
        self._process_executor = None
        self._shutdown = False
        
        self._initialize_executors()
        
        # Task queue with priority support
        self._task_queue = queue.PriorityQueue()
        self._results = {}
        self._lock = threading.RLock()
        
        # Worker thread for task dispatching
        self._dispatcher_thread = threading.Thread(target=self._dispatch_tasks, daemon=True)
        self._dispatcher_thread.start()
    
    def _initialize_executors(self) -> None:
        """Initialize executor pools based on strategy"""
        if self.strategy in [ProcessingStrategy.THREAD_BASED, ProcessingStrategy.HYBRID]:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="photonic_compilation"
            )
        
        if self.strategy in [ProcessingStrategy.PROCESS_BASED, ProcessingStrategy.HYBRID]:
            self._process_executor = ProcessPoolExecutor(
                max_workers=min(4, self.max_workers),  # Fewer processes due to overhead
                mp_context=mp.get_context('spawn')
            )
    
    def submit_compilation(self, 
                          compilation_func: Callable,
                          model: Any,
                          config: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL,
                          timeout_seconds: Optional[float] = None) -> str:
        """Submit compilation task"""
        if self._shutdown:
            raise PhotonicMLIRError("Worker pool is shut down")
        
        task_id = f"compile_{int(time.time() * 1000000)}"
        
        task = Task(
            id=task_id,
            func=compilation_func,
            args=(model,),
            kwargs={"config": config},
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Use negative priority for queue (higher priority = lower number)
        priority_value = -priority.value
        self._task_queue.put((priority_value, time.time(), task))
        
        self.logger.info(f"Submitted compilation task: {task_id}")
        return task_id
    
    def submit_optimization(self,
                          optimization_func: Callable,
                          circuit: Any,
                          passes: List[Any],
                          priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit optimization task"""
        task_id = f"optimize_{int(time.time() * 1000000)}"
        
        task = Task(
            id=task_id,
            func=optimization_func,
            args=(circuit, passes),
            kwargs={},
            priority=priority
        )
        
        priority_value = -priority.value
        self._task_queue.put((priority_value, time.time(), task))
        
        self.logger.info(f"Submitted optimization task: {task_id}")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of submitted task"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self._results:
                    result = self._results.pop(task_id)
                    if isinstance(result, Exception):
                        raise result
                    return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise PhotonicMLIRError(f"Task {task_id} timeout after {timeout}s")
            
            time.sleep(0.1)
    
    def _dispatch_tasks(self) -> None:
        """Dispatcher thread that assigns tasks to appropriate executors"""
        while not self._shutdown:
            try:
                # Get next task with timeout
                priority, timestamp, task = self._task_queue.get(timeout=1.0)
                
                # Choose executor based on task characteristics
                executor = self._choose_executor(task)
                
                if executor:
                    future = executor.submit(self._execute_task, task)
                    future.add_done_callback(lambda f, tid=task.id: self._handle_result(tid, f))
                else:
                    # Execute directly if no executor available
                    self._execute_task_direct(task)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in task dispatcher: {e}")
    
    def _choose_executor(self, task: Task) -> Optional[concurrent.futures.Executor]:
        """Choose appropriate executor for task"""
        # For now, use simple heuristics
        # In practice, would analyze task characteristics
        
        if self.strategy == ProcessingStrategy.THREAD_BASED:
            return self._thread_executor
        elif self.strategy == ProcessingStrategy.PROCESS_BASED:
            return self._process_executor
        elif self.strategy == ProcessingStrategy.HYBRID:
            # Use processes for heavy compilation, threads for optimization
            if "compile" in task.func.__name__:
                return self._process_executor
            else:
                return self._thread_executor
        
        return self._thread_executor
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task"""
        try:
            self.logger.debug(f"Executing task: {task.id}")
            start_time = time.time()
            
            result = task.func(*task.args, **task.kwargs)
            
            duration = time.time() - start_time
            self.logger.info(f"Task {task.id} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            raise
    
    def _execute_task_direct(self, task: Task) -> None:
        """Execute task directly in dispatcher thread"""
        try:
            result = self._execute_task(task)
            with self._lock:
                self._results[task.id] = result
        except Exception as e:
            with self._lock:
                self._results[task.id] = e
    
    def _handle_result(self, task_id: str, future: Future) -> None:
        """Handle task completion"""
        try:
            result = future.result()
            with self._lock:
                self._results[task_id] = result
        except Exception as e:
            with self._lock:
                self._results[task_id] = e
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool"""
        self._shutdown = True
        
        if self._thread_executor:
            self._thread_executor.shutdown(wait=wait)
        
        if self._process_executor:
            self._process_executor.shutdown(wait=wait)
        
        self.logger.info("Worker pool shut down")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        stats = {
            "strategy": self.strategy.value,
            "max_workers": self.max_workers,
            "pending_tasks": self._task_queue.qsize(),
            "completed_results": len(self._results),
            "shutdown": self._shutdown
        }
        
        if self._thread_executor:
            stats["thread_executor"] = {
                "max_workers": self._thread_executor._max_workers,
            }
        
        if self._process_executor:
            stats["process_executor"] = {
                "max_workers": self._process_executor._max_workers,
            }
        
        return stats


class SimulationWorkerPool:
    """Specialized worker pool for simulation tasks"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, mp.cpu_count())
        
        self.logger = get_logger("photonic_mlir.concurrent.simulation_pool")
        
        # Use thread pool for simulation (I/O bound with device communication)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="photonic_simulation"
        )
        
        self._active_simulations = {}
        self._lock = threading.RLock()
        self._shutdown = False
    
    def submit_simulation(self,
                         simulation_func: Callable,
                         circuit: Any,
                         test_inputs: Any,
                         config: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit simulation task"""
        if self._shutdown:
            raise PhotonicMLIRError("Simulation pool is shut down")
        
        task_id = f"sim_{int(time.time() * 1000000)}"
        
        future = self._executor.submit(
            self._execute_simulation,
            simulation_func,
            circuit,
            test_inputs,
            config
        )
        
        with self._lock:
            self._active_simulations[task_id] = {
                "future": future,
                "started_at": time.time(),
                "priority": priority
            }
        
        self.logger.info(f"Submitted simulation task: {task_id}")
        return task_id
    
    def get_simulation_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get simulation result"""
        with self._lock:
            if task_id not in self._active_simulations:
                raise PhotonicMLIRError(f"Simulation task not found: {task_id}")
            
            future = self._active_simulations[task_id]["future"]
        
        try:
            result = future.result(timeout=timeout)
            
            # Cleanup completed task
            with self._lock:
                if task_id in self._active_simulations:
                    del self._active_simulations[task_id]
            
            return result
            
        except concurrent.futures.TimeoutError:
            raise PhotonicMLIRError(f"Simulation {task_id} timeout after {timeout}s")
        except Exception as e:
            # Cleanup failed task
            with self._lock:
                if task_id in self._active_simulations:
                    del self._active_simulations[task_id]
            raise
    
    def _execute_simulation(self,
                           simulation_func: Callable,
                           circuit: Any,
                           test_inputs: Any,
                           config: Dict[str, Any]) -> Any:
        """Execute simulation task"""
        try:
            self.logger.debug("Executing simulation task")
            start_time = time.time()
            
            result = simulation_func(circuit, test_inputs, **config)
            
            duration = time.time() - start_time
            self.logger.info(f"Simulation completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
    
    def cancel_simulation(self, task_id: str) -> bool:
        """Cancel a running simulation"""
        with self._lock:
            if task_id in self._active_simulations:
                future = self._active_simulations[task_id]["future"]
                cancelled = future.cancel()
                
                if cancelled:
                    del self._active_simulations[task_id]
                    self.logger.info(f"Cancelled simulation: {task_id}")
                
                return cancelled
        
        return False
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown simulation pool"""
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        
        with self._lock:
            self._active_simulations.clear()
        
        self.logger.info("Simulation pool shut down")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation pool statistics"""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_simulations": len(self._active_simulations),
                "shutdown": self._shutdown
            }


class ConcurrencyManager:
    """Central manager for concurrent operations"""
    
    def __init__(self):
        self.compilation_pool = CompilationWorkerPool()
        self.simulation_pool = SimulationWorkerPool()
        
        # Resource pools for different components
        self._resource_pools = {}
        
        self.logger = get_logger("photonic_mlir.concurrent.manager")
        
        # Performance monitoring
        self._start_time = time.time()
        self._task_count = 0
        self._lock = threading.RLock()
    
    def create_resource_pool(self,
                           name: str,
                           resource_factory: Callable,
                           max_size: int = 10,
                           min_size: int = 2) -> ResourcePool:
        """Create a named resource pool"""
        with self._lock:
            if name in self._resource_pools:
                return self._resource_pools[name]
            
            pool = ResourcePool(resource_factory, max_size, min_size)
            self._resource_pools[name] = pool
            
            self.logger.info(f"Created resource pool: {name}")
            return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get existing resource pool"""
        with self._lock:
            return self._resource_pools.get(name)
    
    def submit_compilation(self, *args, **kwargs) -> str:
        """Submit compilation task to worker pool"""
        with self._lock:
            self._task_count += 1
        return self.compilation_pool.submit_compilation(*args, **kwargs)
    
    def submit_simulation(self, *args, **kwargs) -> str:
        """Submit simulation task to worker pool"""
        with self._lock:
            self._task_count += 1
        return self.simulation_pool.submit_simulation(*args, **kwargs)
    
    def batch_compile(self,
                     compilation_requests: List[Dict[str, Any]],
                     max_concurrent: int = None) -> List[str]:
        """Submit multiple compilation tasks in batch"""
        max_concurrent = max_concurrent or self.compilation_pool.max_workers
        
        task_ids = []
        for request in compilation_requests:
            task_id = self.submit_compilation(**request)
            task_ids.append(task_id)
            
            # Throttle submissions to avoid overwhelming the system
            if len(task_ids) % max_concurrent == 0:
                time.sleep(0.1)
        
        self.logger.info(f"Submitted {len(task_ids)} compilation tasks in batch")
        return task_ids
    
    def wait_for_completion(self,
                           task_ids: List[str],
                           timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete"""
        results = {}
        errors = {}
        
        start_time = time.time()
        
        for task_id in task_ids:
            try:
                remaining_timeout = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                
                if task_id.startswith("compile_"):
                    result = self.compilation_pool.get_result(task_id, remaining_timeout)
                elif task_id.startswith("sim_"):
                    result = self.simulation_pool.get_simulation_result(task_id, remaining_timeout)
                else:
                    raise PhotonicMLIRError(f"Unknown task type: {task_id}")
                
                results[task_id] = result
                
            except Exception as e:
                errors[task_id] = str(e)
        
        return {"results": results, "errors": errors}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency system statistics"""
        with self._lock:
            uptime = time.time() - self._start_time
            
            stats = {
                "uptime_seconds": uptime,
                "total_tasks_submitted": self._task_count,
                "compilation_pool": self.compilation_pool.get_stats(),
                "simulation_pool": self.simulation_pool.get_stats(),
                "resource_pools": {
                    name: pool.get_stats()
                    for name, pool in self._resource_pools.items()
                }
            }
        
        return stats
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all concurrent operations"""
        self.logger.info("Shutting down concurrency manager")
        
        # Shutdown worker pools
        self.compilation_pool.shutdown(wait=wait)
        self.simulation_pool.shutdown(wait=wait)
        
        # Shutdown resource pools
        with self._lock:
            for pool in self._resource_pools.values():
                pool.shutdown()
            self._resource_pools.clear()
        
        self.logger.info("Concurrency manager shut down")


# Global concurrency manager
_global_concurrency_manager = ConcurrencyManager()


def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager"""
    return _global_concurrency_manager


def parallel_compile(compilation_requests: List[Dict[str, Any]],
                    max_concurrent: int = None,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
    """Utility function for parallel compilation"""
    manager = get_concurrency_manager()
    
    task_ids = manager.batch_compile(compilation_requests, max_concurrent)
    results = manager.wait_for_completion(task_ids, timeout)
    
    return results


def async_simulation(simulation_func: Callable,
                    circuits: List[Any],
                    test_inputs: List[Any],
                    configs: List[Dict[str, Any]]) -> List[str]:
    """Utility function for asynchronous simulation"""
    manager = get_concurrency_manager()
    
    task_ids = []
    for circuit, inputs, config in zip(circuits, test_inputs, configs):
        task_id = manager.submit_simulation(
            simulation_func, circuit, inputs, config
        )
        task_ids.append(task_id)
    
    return task_ids