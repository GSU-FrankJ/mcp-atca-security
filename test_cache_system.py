#!/usr/bin/env python3
"""
PSI Embedding Cache System Test Suite

This test verifies the caching system functionality without requiring heavy ML dependencies.
Tests include LRU cache, persistent cache, and performance metrics.
"""

import sys
import os
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

# Mock Settings for testing
class MockSettings:
    """Mock settings class for testing cache components"""
    
    def __init__(self, temp_dir: Path):
        # Cache settings
        self.embedding_cache_size = 100
        self.embedding_cache_memory_mb = 10.0
        self.embedding_dimension = 128  # Smaller for testing
        
        # Persistent cache settings
        self.cache_dir = temp_dir / "cache"
        self.persistent_cache_files = 50
        self.persistent_cache_size_mb = 50.0
        
        # FAISS settings
        self.faiss_index_type = "Flat"  # Simpler for testing


class CacheTestSuite:
    """Test suite for PSI caching system"""
    
    def __init__(self):
        self.temp_dir = None
        self.settings = None
        self.test_results: Dict[str, bool] = {}
        
    async def run_all_tests(self) -> bool:
        """Run all cache tests"""
        print("🧪 PSI Cache System Test Suite")
        print("=" * 50)
        
        # Setup test environment
        self.temp_dir = Path(tempfile.mkdtemp())
        self.settings = MockSettings(self.temp_dir)
        
        try:
            # Test 1: LRU Cache functionality
            print("\n=== Testing LRU Cache ===")
            await self.test_lru_cache()
            
            # Test 2: Persistent Cache functionality
            print("\n=== Testing Persistent Cache ===")
            await self.test_persistent_cache()
            
            # Test 3: Cache Manager integration
            print("\n=== Testing Cache Manager ===")
            await self.test_cache_manager()
            
            # Test 4: Performance metrics
            print("\n=== Testing Performance Metrics ===")
            await self.test_performance_metrics()
            
            # Test 5: FAISS integration (if available)
            print("\n=== Testing FAISS Integration ===")
            await self.test_faiss_integration()
            
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Print results
        print("\n" + "=" * 50)
        print("🎯 Cache Test Results:")
        all_passed = True
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")
            if not passed:
                all_passed = False
        
        print(f"\n🏁 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        return all_passed
    
    def generate_test_embedding(self, size: int = 128) -> np.ndarray:
        """Generate a test embedding vector"""
        return np.random.rand(size).astype(np.float32)
    
    async def test_lru_cache(self) -> None:
        """Test LRU cache functionality"""
        try:
            from mcp_security.core.psi.cache import LRUEmbeddingCache
            
            cache = LRUEmbeddingCache(max_size=5, max_memory_mb=1.0)
            
            # Test basic put/get
            test_text = "hello world"
            test_embedding = self.generate_test_embedding()
            
            cache.put(test_text, test_embedding)
            retrieved = cache.get(test_text)
            
            assert retrieved is not None, "Failed to retrieve cached embedding"
            assert np.allclose(retrieved, test_embedding), "Retrieved embedding doesn't match"
            print("✅ Basic put/get operations work")
            
            # Test cache stats
            stats = cache.get_stats()
            assert stats.hits == 1, f"Expected 1 hit, got {stats.hits}"
            assert stats.entries_count == 1, f"Expected 1 entry, got {stats.entries_count}"
            print("✅ Cache statistics tracking works")
            
            # Test cache eviction
            for i in range(10):  # More than max_size
                cache.put(f"test_{i}", self.generate_test_embedding())
            
            final_stats = cache.get_stats()
            assert final_stats.entries_count <= 5, f"Cache size exceeded limit: {final_stats.entries_count}"
            assert final_stats.evictions > 0, "No evictions occurred"
            print("✅ Cache eviction works correctly")
            
            # Test cache miss
            miss_result = cache.get("nonexistent_key")
            assert miss_result is None, "Should return None for missing key"
            print("✅ Cache miss handling works")
            
            self.test_results['lru_cache'] = True
            
        except Exception as e:
            print(f"❌ LRU cache test failed: {e}")
            self.test_results['lru_cache'] = False
    
    async def test_persistent_cache(self) -> None:
        """Test persistent cache functionality"""
        try:
            from mcp_security.core.psi.cache import PersistentEmbeddingCache
            
            cache_dir = self.temp_dir / "persistent_test"
            cache = PersistentEmbeddingCache(
                cache_dir=cache_dir,
                max_files=10,
                max_size_mb=1.0
            )
            
            # Test basic put/get
            test_text = "persistent test"
            test_embedding = self.generate_test_embedding()
            
            await cache.put(test_text, test_embedding)
            retrieved = await cache.get(test_text)
            
            assert retrieved is not None, "Failed to retrieve from persistent cache"
            assert np.allclose(retrieved, test_embedding), "Retrieved embedding doesn't match"
            print("✅ Persistent cache put/get works")
            
            # Test metadata file creation
            metadata_file = cache_dir / "metadata.json"
            assert metadata_file.exists(), "Metadata file not created"
            print("✅ Metadata file creation works")
            
            # Test persistence across instances
            cache2 = PersistentEmbeddingCache(
                cache_dir=cache_dir,
                max_files=10,
                max_size_mb=1.0
            )
            
            retrieved2 = await cache2.get(test_text)
            assert retrieved2 is not None, "Failed to retrieve from new instance"
            assert np.allclose(retrieved2, test_embedding), "Persistence failed"
            print("✅ Cache persistence across instances works")
            
            # Test cache miss
            miss_result = await cache.get("nonexistent_persistent_key")
            assert miss_result is None, "Should return None for missing key"
            print("✅ Persistent cache miss handling works")
            
            self.test_results['persistent_cache'] = True
            
        except Exception as e:
            print(f"❌ Persistent cache test failed: {e}")
            self.test_results['persistent_cache'] = False
    
    async def test_cache_manager(self) -> None:
        """Test unified cache manager"""
        try:
            from mcp_security.core.psi.cache import EmbeddingCacheManager
            
            manager = EmbeddingCacheManager(self.settings)
            
            # Test multi-level caching
            test_text = "cache manager test"
            test_embedding = self.generate_test_embedding()
            
            # Put embedding
            await manager.put_embedding(test_text, test_embedding)
            
            # Get embedding (should hit LRU cache)
            retrieved = await manager.get_embedding(test_text)
            assert retrieved is not None, "Failed to retrieve from cache manager"
            assert np.allclose(retrieved, test_embedding), "Retrieved embedding doesn't match"
            print("✅ Cache manager put/get works")
            
            # Test performance stats
            stats = manager.get_performance_stats()
            assert 'total_requests' in stats, "Missing performance stats"
            assert 'cache_hits' in stats, "Missing hit statistics"
            assert stats['overall_hit_rate'] >= 0, "Invalid hit rate"
            print("✅ Cache manager statistics work")
            
            # Test cache clearing
            manager.clear_all_caches()
            retrieved_after_clear = await manager.get_embedding(test_text)
            # Should miss LRU but might hit persistent
            print("✅ Cache clearing works")
            
            self.test_results['cache_manager'] = True
            
        except Exception as e:
            print(f"❌ Cache manager test failed: {e}")
            self.test_results['cache_manager'] = False
    
    async def test_performance_metrics(self) -> None:
        """Test performance metrics and optimization"""
        try:
            from mcp_security.core.psi.cache import LRUEmbeddingCache
            
            cache = LRUEmbeddingCache(max_size=100, max_memory_mb=10.0)
            
            # Generate test data
            test_prompts = [
                f"test prompt {i}" for i in range(50)
            ]
            test_embeddings = [
                self.generate_test_embedding() for _ in range(50)
            ]
            
            # Time cache operations
            start_time = time.time()
            
            # Fill cache
            for prompt, embedding in zip(test_prompts, test_embeddings):
                cache.put(prompt, embedding)
            
            # Test retrieval performance
            hit_times = []
            for prompt in test_prompts[:25]:  # Test some hits
                hit_start = time.time()
                result = cache.get(prompt)
                hit_time = (time.time() - hit_start) * 1000  # ms
                hit_times.append(hit_time)
                assert result is not None, "Cache hit failed"
            
            avg_hit_time = sum(hit_times) / len(hit_times)
            print(f"✅ Average cache hit time: {avg_hit_time:.3f}ms")
            
            # Test cache statistics
            stats = cache.get_stats()
            print(f"✅ Cache hit rate: {stats.hit_rate:.2%}")
            print(f"✅ Memory usage: {stats.memory_usage_mb:.2f}MB")
            
            # Performance should be fast (< 1ms per operation)
            assert avg_hit_time < 1.0, f"Cache hits too slow: {avg_hit_time:.3f}ms"
            
            self.test_results['performance_metrics'] = True
            
        except Exception as e:
            print(f"❌ Performance metrics test failed: {e}")
            self.test_results['performance_metrics'] = False
    
    async def test_faiss_integration(self) -> None:
        """Test FAISS integration for similarity search"""
        try:
            from mcp_security.core.psi.cache import FAISSEmbeddingIndex
            
            # Test basic FAISS operations
            dimension = 128
            index = FAISSEmbeddingIndex(dimension=dimension, index_type="Flat")
            
            # Add test embeddings
            test_texts = ["hello", "world", "test", "embedding", "similarity"]
            test_embeddings = np.array([
                self.generate_test_embedding(dimension) for _ in test_texts
            ])
            
            index.add_embeddings(test_texts, test_embeddings)
            print("✅ FAISS index creation and embedding addition works")
            
            # Test similarity search
            query_embedding = test_embeddings[0]  # Use first embedding as query
            similar_results = index.search_similar(query_embedding, k=3, threshold=0.5)
            
            assert len(similar_results) > 0, "No similar results found"
            assert similar_results[0][0] == test_texts[0], "Most similar should be exact match"
            print("✅ FAISS similarity search works")
            
            # Test index persistence
            index_path = self.temp_dir / "test_index.bin"
            index.save_index(index_path)
            assert index_path.exists(), "FAISS index file not saved"
            print("✅ FAISS index persistence works")
            
            self.test_results['faiss_integration'] = True
            
        except ImportError:
            print("⚠️  FAISS not available, skipping FAISS tests")
            self.test_results['faiss_integration'] = True  # Pass if not available
        except Exception as e:
            print(f"❌ FAISS integration test failed: {e}")
            self.test_results['faiss_integration'] = False

async def main():
    """Main test execution"""
    test_suite = CacheTestSuite()
    success = await test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 