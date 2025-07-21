"""
Embedding Caching System for PSI Engine

This module provides efficient caching mechanisms for token embeddings to improve
performance and reduce redundant calculations. Features include:
- LRU cache for real-time embedding storage
- Persistent cache with disk storage
- FAISS integration for efficient similarity search
- Memory management and cache invalidation strategies
- Performance metrics and optimization
"""

import asyncio
import hashlib
import json
import pickle
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from mcp_security.utils.logging import SecurityLogger, get_logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

@dataclass
class CacheEntry:
    """Single cache entry for storing embedding data"""
    
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Update access information when entry is created"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def update_access(self) -> None:
        """Update access tracking information"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.timestamp
    
    @property
    def size_bytes(self) -> int:
        """Estimate memory size of this entry"""
        embedding_size = self.embedding.nbytes if self.embedding is not None else 0
        metadata_size = len(str(self.metadata)) if self.metadata else 0
        return embedding_size + metadata_size + 64  # Base overhead

@dataclass
class CacheStats:
    """Cache performance statistics"""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entries_count: int = 0
    average_lookup_time_ms: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_hit_rate(self) -> None:
        """Update calculated hit rate"""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "entries_count": self.entries_count,
            "average_lookup_time_ms": self.average_lookup_time_ms,
            "hit_rate": self.hit_rate,
            "memory_usage_mb": self.memory_usage_mb
        }

class LRUEmbeddingCache:
    """
    Thread-safe LRU cache for embedding storage with memory management
    """
    
    def __init__(self, max_size: int = 10000, max_memory_mb: float = 512.0):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Performance tracking
        self._lookup_times: List[float] = []
        
    def _generate_key(self, text: str, model_name: str = "default") -> str:
        """Generate cache key from text and model name"""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds limits (must be called with lock)"""
        # Check size limit
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats.evictions += 1
        
        # Check memory limit
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        while current_memory > self.max_memory_bytes and self._cache:
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            removed_entry = self._cache[oldest_key]
            current_memory -= removed_entry.size_bytes
            self._remove_entry(oldest_key)
            self._stats.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats (must be called with lock)"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entries_count -= 1
    
    def _update_stats(self) -> None:
        """Update performance statistics (must be called with lock)"""
        self._stats.update_hit_rate()
        self._stats.memory_usage_mb = self._stats.size_bytes / (1024 * 1024)
        
        if self._lookup_times:
            self._stats.average_lookup_time_ms = sum(self._lookup_times) / len(self._lookup_times)
            # Keep only recent lookup times (last 1000)
            if len(self._lookup_times) > 1000:
                self._lookup_times = self._lookup_times[-1000:]
    
    def get(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            text: Input text
            model_name: Model identifier
            
        Returns:
            Cached embedding or None if not found
        """
        start_time = time.time()
        key = self._generate_key(text, model_name)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(key)
                entry.update_access()
                self._cache[key] = entry
                
                self._stats.hits += 1
                lookup_time = (time.time() - start_time) * 1000
                self._lookup_times.append(lookup_time)
                self._update_stats()
                
                return entry.embedding.copy()
            else:
                self._stats.misses += 1
                self._update_stats()
                return None
    
    def put(self, text: str, embedding: np.ndarray, model_name: str = "default", 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store embedding in cache
        
        Args:
            text: Input text
            embedding: Embedding vector
            model_name: Model identifier
            metadata: Optional metadata
        """
        key = self._generate_key(text, model_name)
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                embedding=embedding.copy(),
                timestamp=time.time(),
                metadata=metadata
            )
            
            # Update cache and stats
            self._cache[key] = entry
            self._stats.size_bytes += entry.size_bytes
            self._stats.entries_count += 1
            
            # Evict if necessary
            self._evict_if_needed()
            self._update_stats()
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self._lock:
            self._update_stats()
            return CacheStats(**self._stats.to_dict())
    
    def get_keys_by_frequency(self, limit: int = 100) -> List[Tuple[str, int]]:
        """Get most frequently accessed keys"""
        with self._lock:
            items = [(key, entry.access_count) for key, entry in self._cache.items()]
            return sorted(items, key=lambda x: x[1], reverse=True)[:limit]

class PersistentEmbeddingCache:
    """
    Persistent cache for storing embeddings on disk
    """
    
    def __init__(self, cache_dir: Path, max_files: int = 1000, 
                 max_size_mb: float = 1024.0):
        """
        Initialize persistent cache
        
        Args:
            cache_dir: Directory for cache files
            max_files: Maximum number of cache files
            max_size_mb: Maximum total size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Initialize metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load or initialize cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {"files": {}, "total_size": 0}
        else:
            self.metadata = {"files": {}, "total_size": 0}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_filename(self, text: str, model_name: str = "default") -> str:
        """Generate filename for cache entry"""
        key = hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()
        return f"{key}.pkl"
    
    def _cleanup_if_needed(self) -> None:
        """Remove old files if cache exceeds limits"""
        # Sort files by access time (least recently used first)
        file_list = [(f, meta["last_accessed"]) 
                    for f, meta in self.metadata["files"].items()]
        file_list.sort(key=lambda x: x[1])
        
        # Remove files if exceeding limits
        while (len(file_list) >= self.max_files or 
               self.metadata["total_size"] > self.max_size_bytes):
            if not file_list:
                break
                
            filename, _ = file_list.pop(0)
            self._remove_file(filename)
    
    def _remove_file(self, filename: str) -> None:
        """Remove cache file and update metadata"""
        file_path = self.cache_dir / filename
        if file_path.exists():
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                
                if filename in self.metadata["files"]:
                    del self.metadata["files"][filename]
                    self.metadata["total_size"] -= file_size
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {filename}: {e}")
    
    async def get(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """
        Get embedding from persistent cache
        
        Args:
            text: Input text
            model_name: Model identifier
            
        Returns:
            Cached embedding or None if not found
        """
        filename = self._generate_filename(text, model_name)
        file_path = self.cache_dir / filename
        
        if not file_path.exists() or filename not in self.metadata["files"]:
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            self.metadata["files"][filename]["last_accessed"] = time.time()
            self._save_metadata()
            
            return data["embedding"]
        except Exception as e:
            self.logger.warning(f"Failed to load cached embedding {filename}: {e}")
            self._remove_file(filename)
            return None
    
    async def put(self, text: str, embedding: np.ndarray, model_name: str = "default",
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store embedding in persistent cache
        
        Args:
            text: Input text
            embedding: Embedding vector
            model_name: Model identifier
            metadata: Optional metadata
        """
        filename = self._generate_filename(text, model_name)
        file_path = self.cache_dir / filename
        
        data = {
            "text": text,
            "model_name": model_name,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": metadata
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            file_size = file_path.stat().st_size
            
            # Update metadata
            if filename in self.metadata["files"]:
                old_size = self.metadata["files"][filename]["size"]
                self.metadata["total_size"] -= old_size
            
            self.metadata["files"][filename] = {
                "size": file_size,
                "created": time.time(),
                "last_accessed": time.time()
            }
            self.metadata["total_size"] += file_size
            
            self._cleanup_if_needed()
            self._save_metadata()
            
        except Exception as e:
            self.logger.error(f"Failed to save embedding to cache {filename}: {e}")
    
    def clear(self) -> None:
        """Clear all cached files"""
        try:
            for filename in list(self.metadata["files"].keys()):
                self._remove_file(filename)
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Failed to clear persistent cache: {e}")

class FAISSEmbeddingIndex:
    """
    FAISS-based similarity search index for embeddings
    """
    
    def __init__(self, dimension: int = 768, index_type: str = "IVF"):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ("IVF", "Flat", "HNSW")
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.text_mapping: Dict[int, str] = {}
        self.reverse_mapping: Dict[str, int] = {}
        self.next_id = 0
        
        self.logger: SecurityLogger = get_logger(__name__)
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the FAISS index"""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        elif self.index_type == "IVF":
            # Index with inverted files for faster search
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical NSW for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        self.logger.info(f"Initialized FAISS index: {self.index_type} (dim={self.dimension})")
    
    def add_embeddings(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index
        
        Args:
            texts: List of text strings
            embeddings: Corresponding embeddings (shape: [n, dimension])
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts must match number of embeddings")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_id = self.next_id
        self.index.add(embeddings)
        
        # Update mappings
        for i, text in enumerate(texts):
            current_id = start_id + i
            self.text_mapping[current_id] = text
            self.reverse_mapping[text] = current_id
        
        self.next_id += len(texts)
        
        # Train index if necessary (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            if self.index.ntotal >= 1000:  # Need minimum samples to train
                self.index.train(embeddings)
                self.logger.info("FAISS IVF index trained")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10, 
                      threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:
                text = self.text_mapping.get(idx, "")
                if text:
                    results.append((text, float(score)))
        
        return results
    
    def get_embedding_by_text(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for a specific text"""
        if text not in self.reverse_mapping:
            return None
        
        idx = self.reverse_mapping[text]
        # Note: FAISS doesn't directly support getting vectors by ID
        # This would require maintaining a separate storage
        return None
    
    def save_index(self, filepath: Path) -> None:
        """Save index to disk"""
        try:
            faiss.write_index(self.index, str(filepath))
            
            # Save mappings separately
            mappings = {
                "text_mapping": self.text_mapping,
                "reverse_mapping": self.reverse_mapping,
                "next_id": self.next_id,
                "dimension": self.dimension,
                "index_type": self.index_type
            }
            
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(mappings, f, indent=2)
                
            self.logger.info(f"FAISS index saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
    
    def load_index(self, filepath: Path) -> None:
        """Load index from disk"""
        try:
            self.index = faiss.read_index(str(filepath))
            
            # Load mappings
            with open(filepath.with_suffix('.json'), 'r') as f:
                mappings = json.load(f)
            
            self.text_mapping = {int(k): v for k, v in mappings["text_mapping"].items()}
            self.reverse_mapping = mappings["reverse_mapping"]
            self.next_id = mappings["next_id"]
            self.dimension = mappings["dimension"]
            self.index_type = mappings["index_type"]
            
            self.logger.info(f"FAISS index loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            self._initialize_index()

class EmbeddingCacheManager:
    """
    Unified manager for all embedding caching operations
    """
    
    def __init__(self, settings: Any):
        """
        Initialize cache manager
        
        Args:
            settings: PSI settings object
        """
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Initialize cache components
        cache_size = getattr(settings, 'embedding_cache_size', 10000)
        cache_memory_mb = getattr(settings, 'embedding_cache_memory_mb', 512.0)
        self.lru_cache = LRUEmbeddingCache(cache_size, cache_memory_mb)
        
        # Persistent cache
        cache_dir = getattr(settings, 'cache_dir', Path("cache"))
        self.persistent_cache = PersistentEmbeddingCache(
            cache_dir / "embeddings",
            max_files=getattr(settings, 'persistent_cache_files', 1000),
            max_size_mb=getattr(settings, 'persistent_cache_size_mb', 1024.0)
        )
        
        # FAISS index
        embedding_dim = getattr(settings, 'embedding_dimension', 768)
        index_type = getattr(settings, 'faiss_index_type', 'IVF')
        
        try:
            self.faiss_index = FAISSEmbeddingIndex(embedding_dim, index_type)
            self.faiss_available = True
        except ImportError:
            self.logger.warning("FAISS not available, similarity search disabled")
            self.faiss_index = None
            self.faiss_available = False
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.similarity_matches = 0
        
        self.logger.info("EmbeddingCacheManager initialized",
                        extra={
                            "lru_cache_size": cache_size,
                            "cache_memory_mb": cache_memory_mb,
                            "faiss_available": self.faiss_available,
                            "embedding_dimension": embedding_dim
                        })
    
    async def get_embedding(self, text: str, model_name: str = "default",
                           similarity_threshold: float = 0.85) -> Optional[np.ndarray]:
        """
        Get embedding with multi-level caching
        
        Args:
            text: Input text
            model_name: Model identifier
            similarity_threshold: Threshold for similarity matching
            
        Returns:
            Cached embedding or None if not found
        """
        self.total_requests += 1
        
        # Level 1: LRU cache (fastest)
        embedding = self.lru_cache.get(text, model_name)
        if embedding is not None:
            self.cache_hits += 1
            return embedding
        
        # Level 2: Persistent cache
        embedding = await self.persistent_cache.get(text, model_name)
        if embedding is not None:
            # Add to LRU cache for future access
            self.lru_cache.put(text, embedding, model_name)
            self.cache_hits += 1
            return embedding
        
        # Level 3: Similarity search (if available and text is long enough)
        if (self.faiss_available and len(text.split()) >= 3 and 
            self.faiss_index.index.ntotal > 0):
            
            # This would require computing embedding first, which defeats the purpose
            # So we skip similarity search in get_embedding
            pass
        
        return None
    
    async def put_embedding(self, text: str, embedding: np.ndarray, 
                           model_name: str = "default",
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store embedding in all cache levels
        
        Args:
            text: Input text
            embedding: Embedding vector
            model_name: Model identifier
            metadata: Optional metadata
        """
        # Store in LRU cache
        self.lru_cache.put(text, embedding, model_name, metadata)
        
        # Store in persistent cache (async)
        await self.persistent_cache.put(text, embedding, model_name, metadata)
        
        # Add to FAISS index for similarity search
        if self.faiss_available:
            try:
                self.faiss_index.add_embeddings([text], embedding.reshape(1, -1))
            except Exception as e:
                self.logger.warning(f"Failed to add embedding to FAISS index: {e}")
    
    def find_similar_texts(self, query_embedding: np.ndarray, k: int = 10,
                          threshold: float = 0.85) -> List[Tuple[str, float]]:
        """
        Find similar texts using FAISS index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            
        Returns:
            List of (text, similarity_score) tuples
        """
        if not self.faiss_available:
            return []
        
        try:
            return self.faiss_index.search_similar(query_embedding, k, threshold)
        except Exception as e:
            self.logger.warning(f"FAISS similarity search failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        lru_stats = self.lru_cache.get_stats()
        
        stats = {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "overall_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0,
            "similarity_matches": self.similarity_matches,
            "lru_cache": lru_stats.to_dict(),
            "faiss_available": self.faiss_available,
            "faiss_entries": self.faiss_index.index.ntotal if self.faiss_available else 0
        }
        
        return stats
    
    async def preload_common_embeddings(self, common_texts: List[str],
                                       embedding_function,
                                       model_name: str = "default") -> None:
        """
        Preload embeddings for common texts
        
        Args:
            common_texts: List of frequently used texts
            embedding_function: Function to compute embeddings
            model_name: Model identifier
        """
        self.logger.info(f"Preloading {len(common_texts)} common embeddings")
        
        for i, text in enumerate(common_texts):
            # Check if already cached
            existing = await self.get_embedding(text, model_name)
            if existing is not None:
                continue
            
            try:
                # Compute and cache embedding
                embedding = await embedding_function(text)
                await self.put_embedding(text, embedding, model_name)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Preloaded {i + 1}/{len(common_texts)} embeddings")
                    
            except Exception as e:
                self.logger.warning(f"Failed to preload embedding for '{text[:50]}...': {e}")
        
        self.logger.info("Embedding preloading completed")
    
    def clear_all_caches(self) -> None:
        """Clear all cache levels"""
        self.lru_cache.clear()
        self.persistent_cache.clear()
        
        if self.faiss_available:
            self.faiss_index._initialize_index()
        
        # Reset stats
        self.total_requests = 0
        self.cache_hits = 0
        self.similarity_matches = 0
        
        self.logger.info("All caches cleared")
    
    async def save_faiss_index(self, filepath: Optional[Path] = None) -> None:
        """Save FAISS index to disk"""
        if not self.faiss_available:
            return
        
        if filepath is None:
            cache_dir = getattr(self.settings, 'cache_dir', Path("cache"))
            filepath = cache_dir / "faiss_index.bin"
        
        self.faiss_index.save_index(filepath)
    
    async def load_faiss_index(self, filepath: Optional[Path] = None) -> None:
        """Load FAISS index from disk"""
        if not self.faiss_available:
            return
        
        if filepath is None:
            cache_dir = getattr(self.settings, 'cache_dir', Path("cache"))
            filepath = cache_dir / "faiss_index.bin"
        
        if filepath.exists():
            self.faiss_index.load_index(filepath)
        else:
            self.logger.warning(f"FAISS index file not found: {filepath}") 