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