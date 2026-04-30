from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass(frozen=True)
class FileState:
    """Represents the cached state of a file."""
    content: str
    read_timestamp: Optional[int | float] = None
    modify_timestamp: Optional[int | float] = None
    offset: Optional[int] = None
    limit: Optional[int | float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "read_timestamp": self.read_timestamp,
            "modify_timestamp": self.modify_timestamp,
            "offset": self.offset,
            "limit": self.limit,
        }


class FileStateCache:
    """True LRU cache for file states using OrderedDict.
    
    Automatically evicts least-recently-used items when cache is full.
    Most recently accessed items are moved to the end of the OrderedDict.
    """
    
    def __init__(self, maxsize: int = 128):
        """Initialize the cache.
        
        Args:
            maxsize: Maximum number of files to cache (default: 128)
        """
        self.maxsize = maxsize
        self._storage: OrderedDict[str, FileState] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path to absolute path."""
        return str(Path(file_path).resolve())
    
    def get_file_state(self, file_path: str) -> FileState | None:
        file_path = self._normalize_path(file_path)
        if file_path not in self._storage:
            return None
        
        self._storage.move_to_end(file_path)
        self._hits += 1
        return self._storage[file_path]

    def set(self, file_path: str, file_state: FileState) -> None:
        """Store a file state in the cache.
        
        If file already exists, it's removed and re-added to mark as recently used.
        If cache is full, the least-recently-used item (oldest) is evicted.
        
        Args:
            file_path: Absolute path to the file
            file_state: FileState object containing cached data
        """
        file_path = self._normalize_path(file_path)
        
        # If file already exists, remove it first (to update position)
        if file_path in self._storage:
            del self._storage[file_path]
        # Evict oldest item if cache is full
        elif len(self._storage) >= self.maxsize:
            oldest_path = next(iter(self._storage))  # Get first (oldest) item
            del self._storage[oldest_path]
        
        # Add to end (mark as most recently used)
        self._storage[file_path] = file_state
    
    def get(self, file_path: str) -> Optional[FileState]:
        """Retrieve a file state from the cache.
        
        Moves accessed file to end (marks as most recently used).
        Tracks hits and misses for statistics.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            FileState if found, None otherwise
        """
        file_path = self._normalize_path(file_path)
        
        if file_path in self._storage:
            # Move to end (mark as recently used)
            self._storage.move_to_end(file_path)
            self._hits += 1
            return self._storage[file_path]
        
        self._misses += 1
        return None
    
    def is_valid(self, file_path: str) -> bool:
        """Check if cached file state is still valid by comparing modification time.
        
        Validates that the file hasn't been modified since caching.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            True if cache is valid (file mtime unchanged), False otherwise
        """
        import os
        
        file_path = self._normalize_path(file_path)
        
        # Check if file is in cache
        if file_path not in self._storage:
            return False
        
        cached_state = self._storage[file_path]
        if cached_state.modify_timestamp is None:
            return False
        
        try:
            current_mtime = int(os.path.getmtime(file_path))
            return current_mtime == cached_state.modify_timestamp
        except (OSError, FileNotFoundError):
            # File doesn't exist or can't be accessed
            return False
    
    def has(self, file_path: str) -> bool:
        """Check if a file state exists in the cache.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            True if file is cached, False otherwise
        """
        file_path = self._normalize_path(file_path)
        return file_path in self._storage
    
    def delete(self, file_path: str) -> bool:
        """Remove a file state from the cache.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self._normalize_path(file_path)
        
        if file_path in self._storage:
            del self._storage[file_path]
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cached file states and reset statistics."""
        self._storage.clear()
        self._hits = 0
        self._misses = 0
    
    def cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats (size, hits, misses, etc.)
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._storage),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_files": list(self._storage.keys()),
        }
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._storage)
    
    def __contains__(self, file_path: str) -> bool:
        """Check membership using 'in' operator."""
        return self.has(file_path)
    
    def __repr__(self) -> str:
        """String representation of cache state."""
        return f"FileStateCache(size={len(self._storage)}/{self.maxsize})"


