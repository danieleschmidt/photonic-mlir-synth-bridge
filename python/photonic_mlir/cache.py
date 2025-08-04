"""
Intelligent caching system for Photonic MLIR compilation and simulation.
"""

import hashlib
import pickle
import json
import time
import threading
from typing import Any, Dict, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

from .logging_config import get_logger
from .security import SecureFileHandler


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self) -> None:
        """Update access metadata"""
        self.accessed_at = datetime.now()
        self.access_count += 1


class InMemoryCache:
    """High-performance in-memory cache with multiple eviction policies"""
    
    def __init__(self, 
                 max_size_mb: float = 512,
                 policy: CachePolicy = CachePolicy.LRU,
                 default_ttl_seconds: Optional[float] = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        
        self.logger = get_logger("photonic_mlir.cache.memory")
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access metadata
            entry.update_access()
            self._hits += 1
            
            self.logger.debug(f"Cache hit: {key}")
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in cache"""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Rough estimate if can't serialize
            
            ttl = ttl_seconds or self.default_ttl_seconds
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure we have space
            self._ensure_space(size_bytes)
            
            # Add entry
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            
            self.logger.debug(f"Cache put: {key} ({size_bytes} bytes)")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "entries": len(self._cache),
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024)
            }
    
    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure there's enough space for new entry"""
        while (self._current_size_bytes + needed_bytes > self.max_size_bytes 
               and self._cache):
            self._evict_one()
    
    def _evict_one(self) -> None:
        """Evict one entry based on policy"""
        if not self._cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].accessed_at)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].access_count)
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            if expired:
                oldest_key = expired[0]
            else:
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k].created_at)
        else:  # SIZE_BASED
            # Remove largest entry
            oldest_key = max(self._cache.keys(),
                           key=lambda k: self._cache[k].size_bytes)
        
        self._remove_entry(oldest_key)
        self._evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes


class PersistentCache:
    """Persistent disk-based cache for long-term storage"""
    
    def __init__(self, cache_dir: str = ".photonic_cache", max_size_mb: float = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self.file_handler = SecureFileHandler()
        self.logger = get_logger("photonic_mlir.cache.persistent")
        
        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return None
            
            entry_info = self._metadata[key_hash]
            
            # Check expiration
            if self._is_expired(entry_info):
                self._remove_entry(key_hash)
                return None
            
            # Load from disk
            try:
                cache_file = self.cache_dir / f"{key_hash}.pkl"
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                entry_info["accessed_at"] = datetime.now().isoformat()
                entry_info["access_count"] += 1
                self._save_metadata()
                
                self.logger.debug(f"Persistent cache hit: {key}")
                return value
                
            except Exception as e:
                self.logger.warning(f"Failed to load from persistent cache: {e}")
                self._remove_entry(key_hash)
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in persistent cache"""
        with self._lock:
            key_hash = self._hash_key(key)
            
            try:
                # Serialize to disk
                cache_file = self.cache_dir / f"{key_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update metadata
                size_bytes = cache_file.stat().st_size
                now = datetime.now().isoformat()
                
                self._metadata[key_hash] = {
                    "key": key,
                    "created_at": now,
                    "accessed_at": now,
                    "access_count": 1,
                    "size_bytes": size_bytes,
                    "ttl_seconds": ttl_seconds
                }
                
                self._save_metadata()
                self._cleanup_if_needed()
                
                self.logger.debug(f"Persistent cache put: {key} ({size_bytes} bytes)")
                
            except Exception as e:
                self.logger.error(f"Failed to save to persistent cache: {e}")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from persistent cache"""
        with self._lock:
            key_hash = self._hash_key(key)
            return self._remove_entry(key_hash)
    
    def clear(self) -> None:
        """Clear all persistent cache"""
        with self._lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            self._metadata.clear()
            self._save_metadata()
            self.logger.info("Persistent cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics"""
        with self._lock:
            total_size = sum(entry["size_bytes"] for entry in self._metadata.values())
            
            return {
                "entries": len(self._metadata),
                "size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "cache_dir": str(self.cache_dir)
            }
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _is_expired(self, entry_info: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        ttl_seconds = entry_info.get("ttl_seconds")
        if ttl_seconds is None:
            return False
        
        created_at = datetime.fromisoformat(entry_info["created_at"])
        return (datetime.now() - created_at).total_seconds() > ttl_seconds
    
    def _remove_entry(self, key_hash: str) -> bool:
        """Remove entry from persistent cache"""
        if key_hash in self._metadata:
            # Remove file
            cache_file = self.cache_dir / f"{key_hash}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove metadata
            del self._metadata[key_hash]
            self._save_metadata()
            return True
        return False
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup cache if it exceeds size limit"""
        total_size = sum(entry["size_bytes"] for entry in self._metadata.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by access time (LRU)
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda x: x[1]["accessed_at"]
        )
        
        # Remove oldest entries until under limit
        for key_hash, entry_info in sorted_entries:
            self._remove_entry(key_hash)
            total_size -= entry_info["size_bytes"]
            
            if total_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                break


class CompilationCache:
    """Specialized cache for compilation results"""
    
    def __init__(self, memory_cache_mb: float = 256, persistent_cache_mb: float = 1024):
        self.memory_cache = InMemoryCache(memory_cache_mb, CachePolicy.LRU, 3600)  # 1 hour TTL
        self.persistent_cache = PersistentCache(".photonic_compilation_cache", persistent_cache_mb)
        self.logger = get_logger("photonic_mlir.cache.compilation")
    
    def get_compiled_circuit(self, model_hash: str, config_hash: str) -> Optional[Any]:
        """Get compiled circuit from cache"""
        cache_key = f"circuit_{model_hash}_{config_hash}"
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Try persistent cache
        result = self.persistent_cache.get(cache_key)
        if result is not None:
            # Promote to memory cache
            self.memory_cache.put(cache_key, result)
            return result
        
        return None
    
    def put_compiled_circuit(self, model_hash: str, config_hash: str, circuit: Any) -> None:
        """Cache compiled circuit"""
        cache_key = f"circuit_{model_hash}_{config_hash}"
        
        # Store in both caches
        self.memory_cache.put(cache_key, circuit)
        self.persistent_cache.put(cache_key, circuit, ttl_seconds=86400)  # 24 hours
        
        self.logger.info(f"Cached compiled circuit: {cache_key}")
    
    def get_optimization_result(self, circuit_hash: str, pass_name: str, config_hash: str) -> Optional[Any]:
        """Get optimization result from cache"""
        cache_key = f"opt_{circuit_hash}_{pass_name}_{config_hash}"
        
        result = self.memory_cache.get(cache_key)
        if result is None:
            result = self.persistent_cache.get(cache_key)
            if result is not None:
                self.memory_cache.put(cache_key, result)
        
        return result
    
    def put_optimization_result(self, circuit_hash: str, pass_name: str, 
                              config_hash: str, result: Any) -> None:
        """Cache optimization result"""
        cache_key = f"opt_{circuit_hash}_{pass_name}_{config_hash}"
        
        self.memory_cache.put(cache_key, result, ttl_seconds=1800)  # 30 minutes
        self.persistent_cache.put(cache_key, result, ttl_seconds=7200)  # 2 hours
    
    def invalidate_model(self, model_hash: str) -> None:
        """Invalidate all cached results for a model"""
        # This is a simplified version - in practice would need more sophisticated invalidation
        self.memory_cache.clear()
        self.logger.info(f"Invalidated cache for model: {model_hash}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "persistent_cache": self.persistent_cache.get_stats()
        }


class SimulationCache:
    """Specialized cache for simulation results"""
    
    def __init__(self, memory_cache_mb: float = 128, persistent_cache_mb: float = 512):
        self.memory_cache = InMemoryCache(memory_cache_mb, CachePolicy.LFU)
        self.persistent_cache = PersistentCache(".photonic_simulation_cache", persistent_cache_mb)
        self.logger = get_logger("photonic_mlir.cache.simulation")
    
    def get_simulation_result(self, circuit_hash: str, config_hash: str, 
                            input_hash: str) -> Optional[Any]:
        """Get simulation result from cache"""
        cache_key = f"sim_{circuit_hash}_{config_hash}_{input_hash}"
        
        result = self.memory_cache.get(cache_key)
        if result is None:
            result = self.persistent_cache.get(cache_key)
            if result is not None:
                self.memory_cache.put(cache_key, result)
        
        return result
    
    def put_simulation_result(self, circuit_hash: str, config_hash: str,
                            input_hash: str, result: Any) -> None:
        """Cache simulation result"""
        cache_key = f"sim_{circuit_hash}_{config_hash}_{input_hash}"
        
        # Simulation results are cached longer since they're expensive to compute
        self.memory_cache.put(cache_key, result, ttl_seconds=7200)  # 2 hours
        self.persistent_cache.put(cache_key, result, ttl_seconds=86400)  # 24 hours


class CacheManager:
    """Central cache management system"""
    
    def __init__(self):
        self.compilation_cache = CompilationCache()
        self.simulation_cache = SimulationCache()
        
        self.logger = get_logger("photonic_mlir.cache.manager")
        
        # Global cache settings
        self._enabled = True
        self._warmup_completed = False
    
    def enable(self) -> None:
        """Enable caching"""
        self._enabled = True
        self.logger.info("Caching enabled")
    
    def disable(self) -> None:
        """Disable caching"""
        self._enabled = False
        self.logger.info("Caching disabled")
    
    def is_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self._enabled
    
    def warmup(self) -> None:
        """Warmup caches with common operations"""
        if self._warmup_completed:
            return
        
        self.logger.info("Starting cache warmup")
        
        # Warmup would populate caches with common compilation patterns
        # This is a placeholder for actual warmup logic
        
        self._warmup_completed = True
        self.logger.info("Cache warmup completed")
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.compilation_cache.memory_cache.clear()
        self.compilation_cache.persistent_cache.clear()
        self.simulation_cache.memory_cache.clear()
        self.simulation_cache.persistent_cache.clear()
        
        self.logger.info("All caches cleared")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            "enabled": self._enabled,
            "warmup_completed": self._warmup_completed,
            "compilation_cache": self.compilation_cache.get_stats(),
            "simulation_cache": {
                "memory_cache": self.simulation_cache.memory_cache.get_stats(),
                "persistent_cache": self.simulation_cache.persistent_cache.get_stats()
            }
        }
    
    def optimize_caches(self) -> None:
        """Optimize cache performance"""
        # Cleanup expired entries
        # Optimize cache sizes based on usage patterns
        # This is a placeholder for sophisticated cache optimization
        
        self.logger.info("Cache optimization completed")


# Global cache manager
_global_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get global cache manager"""
    return _global_cache_manager


def hash_object(obj: Any) -> str:
    """Generate hash for any object"""
    try:
        # Try to pickle first for consistent hashing
        serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()
    except:
        # Fallback to string representation
        return hashlib.sha256(str(obj).encode()).hexdigest()


def cached_compilation(func: Callable) -> Callable:
    """Decorator for caching compilation results"""
    def wrapper(*args, **kwargs):
        cache_manager = get_cache_manager()
        
        if not cache_manager.is_enabled():
            return func(*args, **kwargs)
        
        # Generate cache key from function arguments
        cache_key = hash_object((func.__name__, args, kwargs))
        
        # Try to get from cache
        result = cache_manager.compilation_cache.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        cache_manager.compilation_cache.memory_cache.put(cache_key, result)
        
        return result
    
    return wrapper


def cached_simulation(func: Callable) -> Callable:
    """Decorator for caching simulation results"""  
    def wrapper(*args, **kwargs):
        cache_manager = get_cache_manager()
        
        if not cache_manager.is_enabled():
            return func(*args, **kwargs)
        
        cache_key = hash_object((func.__name__, args, kwargs))
        
        result = cache_manager.simulation_cache.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        result = func(*args, **kwargs)
        cache_manager.simulation_cache.memory_cache.put(cache_key, result)
        
        return result
    
    return wrapper