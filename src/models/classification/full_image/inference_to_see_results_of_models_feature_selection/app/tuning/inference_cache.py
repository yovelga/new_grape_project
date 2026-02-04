"""
Inference cache for probability maps.

Caches computed probability maps to avoid redundant inference during hyperparameter tuning.
"""

import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


class InferenceCache:
    """
    Cache for inference results (probability maps).
    
    Stores prob_map per image path to avoid recomputing during Optuna trials.
    Only inference needs to be cached; postprocessing is fast and varies per trial.
    """
    
    def __init__(self, max_cache_size_mb: int = 1024):
        """
        Initialize inference cache.
        
        Args:
            max_cache_size_mb: Maximum cache size in megabytes (default 1GB)
        """
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self.max_cache_size_mb = max_cache_size_mb
        self._current_size_mb: float = 0.0
    
    def _get_key(self, path: str) -> str:
        """
        Generate cache key from path.
        
        Args:
            path: File path
            
        Returns:
            Cache key (hash of normalized path)
        """
        # Normalize path
        normalized = str(Path(path).resolve())
        
        # Create hash for consistent key
        key = hashlib.md5(normalized.encode()).hexdigest()
        return key
    
    def _estimate_size_mb(self, array: np.ndarray) -> float:
        """
        Estimate array size in megabytes.
        
        Args:
            array: NumPy array
            
        Returns:
            Size in MB
        """
        bytes_size = array.nbytes
        mb_size = bytes_size / (1024 * 1024)
        return mb_size
    
    def get(self, path: str) -> Optional[np.ndarray]:
        """
        Get cached probability map for path.
        
        Args:
            path: Image file path
            
        Returns:
            Cached probability map or None if not in cache
        """
        key = self._get_key(path)
        
        if key in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache HIT for {Path(path).name}")
            return self._cache[key]
        else:
            self._cache_misses += 1
            logger.debug(f"Cache MISS for {Path(path).name}")
            return None
    
    def put(self, path: str, prob_map: np.ndarray) -> None:
        """
        Store probability map in cache.
        
        Args:
            path: Image file path
            prob_map: Probability map to cache
        """
        key = self._get_key(path)
        
        # Check if already cached
        if key in self._cache:
            return
        
        # Estimate size
        size_mb = self._estimate_size_mb(prob_map)
        
        # Check cache size limit
        if self._current_size_mb + size_mb > self.max_cache_size_mb:
            logger.warning(
                f"Cache size limit reached ({self._current_size_mb:.1f} MB). "
                f"Not caching {Path(path).name} ({size_mb:.1f} MB)"
            )
            return
        
        # Store in cache
        self._cache[key] = prob_map.copy()
        self._current_size_mb += size_mb
        
        logger.debug(f"Cached {Path(path).name} ({size_mb:.2f} MB). "
                    f"Total cache: {self._current_size_mb:.1f} MB")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._current_size_mb = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Inference cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'num_cached': len(self._cache),
            'cache_size_mb': self._current_size_mb,
            'max_size_mb': self.max_cache_size_mb,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def __len__(self) -> int:
        """Number of cached items."""
        return len(self._cache)
    
    def __contains__(self, path: str) -> bool:
        """Check if path is in cache."""
        key = self._get_key(path)
        return key in self._cache
    
    def __str__(self) -> str:
        """String representation of cache stats."""
        stats = self.get_stats()
        return (f"InferenceCache(items={stats['num_cached']}, "
                f"size={stats['cache_size_mb']:.1f}MB, "
                f"hit_rate={stats['hit_rate']:.2%})")
