#!/usr/bin/env python3
"""
Standalone PSI Cache Logic Test

This test verifies the core caching logic without heavy dependencies.
It implements simplified versions of the cache components for testing.
"""

import sys
import asyncio
import time
import tempfile
import shutil
import hashlib
import json
import pickle
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass

import numpy as np

@dataclass 
class SimpleCacheEntry:
    """Simplified cache entry for testing"""
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        self.last_accessed = time.time()
        self.access_count = 1
    
    def update_access(self):
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def size_bytes(self) -> int:
        return self.embedding.nbytes + 64  # Overhead

class SimpleLRUCache:
    """Simplified LRU cache implementation for testing"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: float = 10.0):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, SimpleCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, text: str, model_name: str = "default") -> str:
        """Generate cache key"""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """Get from cache"""
        key = self._generate_key(text, model_name)
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recent)
                entry = self._cache.pop(key)
                entry.update_access()
                self._cache[key] = entry
                self.hits += 1
                return entry.embedding.copy()
            else:
                self.misses += 1
                return None
    
    def put(self, text: str, embedding: np.ndarray, model_name: str = "default"):
        """Put in cache"""
        key = self._generate_key(text, model_name)
        
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._cache.pop(key)
            
            # Create entry
            entry = SimpleCacheEntry(
                embedding=embedding.copy(),
                timestamp=time.time()
            )
            
            # Add to cache
            self._cache[key] = entry
            
            # Evict if needed
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict entries if over limits"""
        # Size limit
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self.evictions += 1
        
        # Memory limit
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        while current_memory > self.max_memory_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            removed_entry = self._cache.pop(oldest_key)
            current_memory -= removed_entry.size_bytes
            self.evictions += 1
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            memory_usage = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "entries_count": len(self._cache),
                "hit_rate": hit_rate,
                "memory_usage_mb": memory_usage / (1024 * 1024)
            }

class SimplePersistentCache:
    """Simplified persistent cache for testing"""
    
    def __init__(self, cache_dir: Path, max_files: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {"files": {}}
        else:
            self.metadata = {"files": {}}
    
    def _save_metadata(self):
        """Save metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass
    
    def _generate_filename(self, text: str, model_name: str = "default") -> str:
        """Generate filename"""
        key = hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()
        return f"{key}.pkl"
    
    async def get(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """Get from persistent cache"""
        filename = self._generate_filename(text, model_name)
        file_path = self.cache_dir / filename
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            if filename in self.metadata["files"]:
                self.metadata["files"][filename]["last_accessed"] = time.time()
                self._save_metadata()
            
            return data["embedding"]
        except Exception:
            return None
    
    async def put(self, text: str, embedding: np.ndarray, model_name: str = "default"):
        """Put in persistent cache"""
        filename = self._generate_filename(text, model_name)
        file_path = self.cache_dir / filename
        
        data = {
            "text": text,
            "model_name": model_name,
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.metadata["files"][filename] = {
                "created": time.time(),
                "last_accessed": time.time()
            }
            
            # Cleanup if needed
            if len(self.metadata["files"]) > self.max_files:
                # Remove oldest file
                oldest_file = min(
                    self.metadata["files"].items(),
                    key=lambda x: x[1]["last_accessed"]
                )[0]
                
                old_path = self.cache_dir / oldest_file
                if old_path.exists():
                    old_path.unlink()
                del self.metadata["files"][oldest_file]
            
            self._save_metadata()
        except Exception:
            pass
    
    def clear(self):
        """Clear persistent cache"""
        for filename in list(self.metadata["files"].keys()):
            file_path = self.cache_dir / filename
            if file_path.exists():
                file_path.unlink()
        self.metadata = {"files": {}}
        self._save_metadata()

class CacheTestSuite:
    """Test suite for cache functionality"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_results: Dict[str, bool] = {}
    
    def generate_test_embedding(self, size: int = 128) -> np.ndarray:
        """Generate test embedding"""
        return np.random.rand(size).astype(np.float32)
    
    async def run_all_tests(self) -> bool:
        """Run all tests"""
        print("🧪 Standalone PSI Cache Test Suite")
        print("=" * 50)
        
        # Setup
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test LRU Cache
            print("\n=== Testing LRU Cache ===")
            await self.test_lru_cache()
            
            # Test Persistent Cache
            print("\n=== Testing Persistent Cache ===")
            await self.test_persistent_cache()
            
            # Test Performance
            print("\n=== Testing Performance ===")
            await self.test_performance()
            
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Results
        print("\n" + "=" * 50)
        print("🎯 Test Results:")
        all_passed = True
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")
            if not passed:
                all_passed = False
        
        print(f"\n🏁 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        return all_passed
    
    async def test_lru_cache(self):
        """Test LRU cache"""
        try:
            cache = SimpleLRUCache(max_size=5, max_memory_mb=1.0)
            
            # Basic operations
            test_text = "hello world"
            test_embedding = self.generate_test_embedding()
            
            cache.put(test_text, test_embedding)
            retrieved = cache.get(test_text)
            
            assert retrieved is not None, "Failed to retrieve"
            assert np.allclose(retrieved, test_embedding), "Embedding mismatch"
            print("✅ Basic put/get works")
            
            # Stats
            stats = cache.get_stats()
            assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
            assert stats["entries_count"] == 1, f"Expected 1 entry, got {stats['entries_count']}"
            print("✅ Statistics work")
            
            # Eviction
            for i in range(10):
                cache.put(f"test_{i}", self.generate_test_embedding())
            
            final_stats = cache.get_stats()
            assert final_stats["entries_count"] <= 5, f"Cache size exceeded: {final_stats['entries_count']}"
            assert final_stats["evictions"] > 0, "No evictions"
            print("✅ Eviction works")
            
            # Miss
            miss = cache.get("nonexistent")
            assert miss is None, "Should return None"
            print("✅ Miss handling works")
            
            self.test_results['lru_cache'] = True
            
        except Exception as e:
            print(f"❌ LRU cache test failed: {e}")
            self.test_results['lru_cache'] = False
    
    async def test_persistent_cache(self):
        """Test persistent cache"""
        try:
            cache_dir = self.temp_dir / "persistent_test"
            cache = SimplePersistentCache(cache_dir, max_files=10)
            
            # Basic operations
            test_text = "persistent test"
            test_embedding = self.generate_test_embedding()
            
            await cache.put(test_text, test_embedding)
            retrieved = await cache.get(test_text)
            
            assert retrieved is not None, "Failed to retrieve"
            assert np.allclose(retrieved, test_embedding), "Embedding mismatch"
            print("✅ Persistent put/get works")
            
            # Metadata
            assert cache.metadata_file.exists(), "Metadata file missing"
            print("✅ Metadata creation works")
            
            # Persistence across instances
            cache2 = SimplePersistentCache(cache_dir, max_files=10)
            retrieved2 = await cache2.get(test_text)
            assert retrieved2 is not None, "Failed cross-instance retrieval"
            assert np.allclose(retrieved2, test_embedding), "Cross-instance mismatch"
            print("✅ Cross-instance persistence works")
            
            # Miss
            miss = await cache.get("nonexistent")
            assert miss is None, "Should return None"
            print("✅ Persistent miss handling works")
            
            self.test_results['persistent_cache'] = True
            
        except Exception as e:
            print(f"❌ Persistent cache test failed: {e}")
            self.test_results['persistent_cache'] = False
    
    async def test_performance(self):
        """Test performance characteristics"""
        try:
            cache = SimpleLRUCache(max_size=100, max_memory_mb=10.0)
            
            # Generate test data
            test_data = [
                (f"prompt {i}", self.generate_test_embedding())
                for i in range(50)
            ]
            
            # Fill cache and measure time
            start_time = time.time()
            for text, embedding in test_data:
                cache.put(text, embedding)
            fill_time = (time.time() - start_time) * 1000
            
            print(f"✅ Cache fill time: {fill_time:.2f}ms for 50 entries")
            
            # Test retrieval performance
            hit_times = []
            for text, _ in test_data[:25]:
                hit_start = time.time()
                result = cache.get(text)
                hit_time = (time.time() - hit_start) * 1000
                hit_times.append(hit_time)
                assert result is not None, "Cache hit failed"
            
            avg_hit_time = sum(hit_times) / len(hit_times)
            print(f"✅ Average hit time: {avg_hit_time:.3f}ms")
            
            # Stats
            stats = cache.get_stats()
            print(f"✅ Hit rate: {stats['hit_rate']:.2%}")
            print(f"✅ Memory usage: {stats['memory_usage_mb']:.2f}MB")
            
            # Performance assertions
            assert avg_hit_time < 1.0, f"Cache too slow: {avg_hit_time:.3f}ms"
            assert fill_time < 100.0, f"Fill too slow: {fill_time:.2f}ms"
            
            self.test_results['performance'] = True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False

async def main():
    """Main test execution"""
    test_suite = CacheTestSuite()
    success = await test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 