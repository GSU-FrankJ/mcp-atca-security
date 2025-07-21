"""
Modular Architecture and Performance Optimization for PSI Engine

This module provides a modular, high-performance architecture including:
- Plugin-based embedding model architecture
- Efficient batching for multiple prompts
- Parallel processing for independent analysis steps
- Hardware acceleration support
- Performance monitoring and profiling
- Sub-200ms processing optimization
"""

import asyncio
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Protocol
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import inspect
import importlib
import sys
from contextlib import asynccontextmanager
import weakref

import numpy as np
from functools import wraps, lru_cache
import threading

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings

class EmbeddingModelPlugin(Protocol):
    """Protocol for embedding model plugins"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        ...
    
    async def encode_async(self, texts: List[str]) -> np.ndarray:
        """Async encode texts to embeddings"""
        ...
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        ...
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        ...

class AnalysisPlugin(Protocol):
    """Protocol for analysis plugins"""
    
    async def analyze(self, prompt: str, embeddings: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Analyze prompt with embeddings"""
        ...
    
    @property
    def plugin_name(self) -> str:
        """Get plugin name"""
        ...

@dataclass
class ProcessingConfig:
    """Configuration for processing optimization"""
    
    # Batch processing
    max_batch_size: int = 32
    batch_timeout_ms: float = 50.0
    enable_batching: bool = True
    
    # Parallel processing
    max_workers: int = min(mp.cpu_count(), 8)
    use_process_pool: bool = False
    enable_async: bool = True
    
    # Performance targets
    target_processing_time_ms: float = 200.0
    target_throughput_rps: float = 100.0
    
    # Hardware acceleration
    enable_gpu: bool = True
    gpu_batch_size: int = 64
    enable_quantization: bool = True
    
    # Caching
    enable_result_caching: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600
    
    # Monitoring
    enable_profiling: bool = True
    profiling_sample_rate: float = 0.1
    enable_metrics: bool = True

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    
    # Processing times
    total_processing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    analysis_time_ms: float = 0.0
    overhead_time_ms: float = 0.0
    
    # Throughput
    requests_processed: int = 0
    requests_per_second: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_size_used: int = 0
    
    # Batch efficiency
    avg_batch_size: float = 0.0
    batch_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'processing': {
                'total_time_ms': self.total_processing_time_ms,
                'embedding_time_ms': self.embedding_time_ms,
                'analysis_time_ms': self.analysis_time_ms,
                'overhead_time_ms': self.overhead_time_ms
            },
            'throughput': {
                'requests_processed': self.requests_processed,
                'requests_per_second': self.requests_per_second
            },
            'resources': {
                'cpu_usage_percent': self.cpu_usage_percent,
                'memory_usage_mb': self.memory_usage_mb,
                'gpu_usage_percent': self.gpu_usage_percent
            },
            'cache': {
                'hit_rate': self.cache_hit_rate,
                'size_used': self.cache_size_used
            },
            'batch': {
                'avg_size': self.avg_batch_size,
                'utilization': self.batch_utilization
            }
        }

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize performance profiler"""
        self.config = config
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=10000)
        self.current_metrics = PerformanceMetrics()
        
        # Profiling data
        self.profiling_data: Dict[str, List[float]] = defaultdict(list)
        self.profiling_enabled = config.enable_profiling
        
        # Performance monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        if config.enable_metrics:
            self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_performance(self) -> None:
        """Monitor system performance in background"""
        try:
            import psutil
            process = psutil.Process()
            
            while self.monitoring_active:
                try:
                    # CPU and memory usage
                    self.current_metrics.cpu_usage_percent = process.cpu_percent()
                    self.current_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    
                    # GPU usage (if available)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            self.current_metrics.gpu_usage_percent = gpus[0].load * 100
                    except ImportError:
                        pass
                    
                    time.sleep(1.0)  # Update every second
                    
                except Exception as e:
                    self.logger.debug(f"Performance monitoring error: {e}")
                    
        except ImportError:
            self.logger.warning("psutil not available for performance monitoring")
    
    @asynccontextmanager
    async def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        if not self.profiling_enabled or np.random.random() > self.config.profiling_sample_rate:
            yield
            return
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.profiling_data[operation_name].append(duration_ms)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'current_metrics': self.current_metrics.to_dict(),
            'profiling_data': {}
        }
        
        # Add profiling statistics
        for operation, times in self.profiling_data.items():
            if times:
                summary['profiling_data'][operation] = {
                    'count': len(times),
                    'avg_ms': np.mean(times),
                    'median_ms': np.median(times),
                    'p95_ms': np.percentile(times, 95),
                    'p99_ms': np.percentile(times, 99),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times)
                }
        
        return summary

class BatchProcessor:
    """Efficient batch processing for multiple prompts"""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize batch processor"""
        self.config = config
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Batch management
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.batch_condition = asyncio.Condition()
        
        # Processing state
        self.processing_active = False
        self.batch_task: Optional[asyncio.Task] = None
        
        if config.enable_batching:
            self.start_batch_processing()
    
    def start_batch_processing(self) -> None:
        """Start batch processing task"""
        if self.batch_task is None or self.batch_task.done():
            self.processing_active = True
            self.batch_task = asyncio.create_task(self._batch_processing_loop())
    
    async def stop_batch_processing(self) -> None:
        """Stop batch processing"""
        self.processing_active = False
        if self.batch_task:
            await self.batch_task
    
    async def _batch_processing_loop(self) -> None:
        """Main batch processing loop"""
        while self.processing_active:
            try:
                async with self.batch_condition:
                    # Wait for requests or timeout
                    try:
                        await asyncio.wait_for(
                            self.batch_condition.wait_for(
                                lambda: len(self.pending_requests) >= self.config.max_batch_size
                            ),
                            timeout=self.config.batch_timeout_ms / 1000.0
                        )
                    except asyncio.TimeoutError:
                        pass  # Process whatever we have
                
                # Process current batch
                if self.pending_requests:
                    await self._process_batch()
                    
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self) -> None:
        """Process a batch of requests"""
        async with self.batch_lock:
            if not self.pending_requests:
                return
            
            # Extract batch
            batch_size = min(len(self.pending_requests), self.config.max_batch_size)
            batch = [self.pending_requests.popleft() for _ in range(batch_size)]
        
        # Process batch
        try:
            await self._execute_batch(batch)
        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}")
            # Set errors for all requests in batch
            for request in batch:
                if not request['future'].done():
                    request['future'].set_exception(e)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Execute a batch of requests"""
        # Group by processing function
        function_groups = defaultdict(list)
        for request in batch:
            function_groups[request['function']].append(request)
        
        # Process each function group
        for func, requests in function_groups.items():
            try:
                # Extract inputs
                inputs = [req['input'] for req in requests]
                
                # Execute batch function
                if asyncio.iscoroutinefunction(func):
                    results = await func(inputs)
                else:
                    results = func(inputs)
                
                # Set results
                for request, result in zip(requests, results):
                    if not request['future'].done():
                        request['future'].set_result(result)
                        
            except Exception as e:
                # Set error for all requests in this group
                for request in requests:
                    if not request['future'].done():
                        request['future'].set_exception(e)
    
    async def submit_batch_request(
        self, 
        function: Callable, 
        input_data: Any
    ) -> Any:
        """Submit a request for batch processing"""
        if not self.config.enable_batching:
            # Process immediately
            if asyncio.iscoroutinefunction(function):
                return await function([input_data])
            else:
                return function([input_data])
        
        # Create future for result
        future = asyncio.Future()
        request = {
            'function': function,
            'input': input_data,
            'future': future,
            'timestamp': time.time()
        }
        
        # Add to pending requests
        async with self.batch_condition:
            self.pending_requests.append(request)
            self.batch_condition.notify()
        
        # Wait for result
        return await future

class PluginManager:
    """Manager for modular plugins"""
    
    def __init__(self):
        """Initialize plugin manager"""
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Plugin registries
        self.embedding_plugins: Dict[str, EmbeddingModelPlugin] = {}
        self.analysis_plugins: Dict[str, AnalysisPlugin] = {}
        
        # Plugin metadata
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Active plugins
        self.active_embedding_plugin: Optional[str] = None
        self.active_analysis_plugins: List[str] = []
    
    def register_embedding_plugin(
        self, 
        name: str, 
        plugin: EmbeddingModelPlugin,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an embedding model plugin"""
        self.embedding_plugins[name] = plugin
        self.plugin_metadata[name] = metadata or {}
        
        self.logger.info(
            "Embedding plugin registered",
            plugin_name=name,
            model_name=plugin.model_name,
            embedding_dim=plugin.embedding_dimension
        )
    
    def register_analysis_plugin(
        self, 
        name: str, 
        plugin: AnalysisPlugin,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an analysis plugin"""
        self.analysis_plugins[name] = plugin
        self.plugin_metadata[name] = metadata or {}
        
        self.logger.info(
            "Analysis plugin registered",
            plugin_name=name
        )
    
    def set_active_embedding_plugin(self, name: str) -> None:
        """Set active embedding plugin"""
        if name not in self.embedding_plugins:
            raise ValueError(f"Embedding plugin '{name}' not registered")
        
        old_plugin = self.active_embedding_plugin
        self.active_embedding_plugin = name
        
        self.logger.info(
            "Active embedding plugin changed",
            old_plugin=old_plugin,
            new_plugin=name
        )
    
    def add_analysis_plugin(self, name: str) -> None:
        """Add analysis plugin to active list"""
        if name not in self.analysis_plugins:
            raise ValueError(f"Analysis plugin '{name}' not registered")
        
        if name not in self.active_analysis_plugins:
            self.active_analysis_plugins.append(name)
            
            self.logger.info(
                "Analysis plugin activated",
                plugin_name=name,
                active_count=len(self.active_analysis_plugins)
            )
    
    def remove_analysis_plugin(self, name: str) -> None:
        """Remove analysis plugin from active list"""
        if name in self.active_analysis_plugins:
            self.active_analysis_plugins.remove(name)
            
            self.logger.info(
                "Analysis plugin deactivated",
                plugin_name=name,
                active_count=len(self.active_analysis_plugins)
            )
    
    def get_active_embedding_plugin(self) -> Optional[EmbeddingModelPlugin]:
        """Get active embedding plugin"""
        if self.active_embedding_plugin:
            return self.embedding_plugins[self.active_embedding_plugin]
        return None
    
    def get_active_analysis_plugins(self) -> List[AnalysisPlugin]:
        """Get active analysis plugins"""
        return [
            self.analysis_plugins[name] 
            for name in self.active_analysis_plugins
            if name in self.analysis_plugins
        ]
    
    def load_plugin_from_module(self, module_path: str, plugin_class: str) -> None:
        """Dynamically load plugin from module"""
        try:
            module = importlib.import_module(module_path)
            plugin_cls = getattr(module, plugin_class)
            plugin_instance = plugin_cls()
            
            # Determine plugin type and register
            if hasattr(plugin_instance, 'encode'):
                self.register_embedding_plugin(
                    plugin_instance.model_name, 
                    plugin_instance
                )
            elif hasattr(plugin_instance, 'analyze'):
                self.register_analysis_plugin(
                    plugin_instance.plugin_name, 
                    plugin_instance
                )
            else:
                raise ValueError(f"Invalid plugin type: {plugin_class}")
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin {module_path}.{plugin_class}: {e}")
            raise

class ModularPSIProcessor:
    """Modular, high-performance PSI processor"""
    
    def __init__(self, config: ProcessingConfig, settings: Settings):
        """Initialize modular PSI processor"""
        self.config = config
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Core components
        self.profiler = PerformanceProfiler(config)
        self.batch_processor = BatchProcessor(config)
        self.plugin_manager = PluginManager()
        
        # Processing pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Result cache
        self.result_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
        
        self._initialize_processing_pools()
        
        self.logger.info(
            "ModularPSIProcessor initialized",
            max_batch_size=config.max_batch_size,
            max_workers=config.max_workers,
            target_time_ms=config.target_processing_time_ms,
            enable_batching=config.enable_batching,
            enable_caching=config.enable_result_caching
        )
    
    def _initialize_processing_pools(self) -> None:
        """Initialize thread and process pools"""
        if self.config.enable_async:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="psi_worker"
            )
        
        if self.config.use_process_pool:
            self.process_pool = ProcessPoolExecutor(
                max_workers=min(self.config.max_workers, mp.cpu_count())
            )
    
    async def process_prompt(
        self, 
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single prompt through the modular pipeline"""
        
        async with self.profiler.profile_operation("total_processing"):
            start_time = time.perf_counter()
            
            try:
                # Check cache first
                if self.config.enable_result_caching:
                    cached_result = self._get_cached_result(prompt)
                    if cached_result is not None:
                        return cached_result
                
                # Get embeddings
                async with self.profiler.profile_operation("embedding_generation"):
                    embeddings = await self._generate_embeddings([prompt])
                    prompt_embedding = embeddings[0] if embeddings is not None else None
                
                if prompt_embedding is None:
                    raise ValueError("Failed to generate embeddings")
                
                # Run analysis plugins in parallel
                async with self.profiler.profile_operation("analysis_processing"):
                    analysis_results = await self._run_analysis_plugins(
                        prompt, prompt_embedding, **kwargs
                    )
                
                # Combine results
                result = {
                    'prompt': prompt,
                    'embeddings': prompt_embedding.tolist() if prompt_embedding is not None else None,
                    'analysis_results': analysis_results,
                    'processing_time_ms': (time.perf_counter() - start_time) * 1000,
                    'metadata': {
                        'processor_version': '1.0',
                        'active_plugins': {
                            'embedding': self.plugin_manager.active_embedding_plugin,
                            'analysis': self.plugin_manager.active_analysis_plugins
                        }
                    }
                }
                
                # Cache result
                if self.config.enable_result_caching:
                    self._cache_result(prompt, result)
                
                # Update metrics
                self.request_count += 1
                processing_time = (time.perf_counter() - start_time) * 1000
                self.profiler.current_metrics.total_processing_time_ms = processing_time
                self.profiler.current_metrics.requests_processed = self.request_count
                
                # Calculate throughput
                elapsed_time = time.time() - self.start_time
                self.profiler.current_metrics.requests_per_second = self.request_count / elapsed_time
                
                return result
                
            except Exception as e:
                self.logger.error(
                    "Prompt processing failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    prompt_length=len(prompt)
                )
                raise
    
    async def process_batch(
        self, 
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of prompts efficiently"""
        
        async with self.profiler.profile_operation("batch_processing"):
            start_time = time.perf_counter()
            
            try:
                self.logger.debug(
                    "Processing batch",
                    batch_size=len(prompts),
                    enable_batching=self.config.enable_batching
                )
                
                # Check cache for all prompts
                cached_results = {}
                uncached_prompts = []
                uncached_indices = []
                
                if self.config.enable_result_caching:
                    for i, prompt in enumerate(prompts):
                        cached_result = self._get_cached_result(prompt)
                        if cached_result is not None:
                            cached_results[i] = cached_result
                        else:
                            uncached_prompts.append(prompt)
                            uncached_indices.append(i)
                else:
                    uncached_prompts = prompts
                    uncached_indices = list(range(len(prompts)))
                
                # Process uncached prompts
                uncached_results = []
                if uncached_prompts:
                    # Generate embeddings in batch
                    async with self.profiler.profile_operation("batch_embedding_generation"):
                        embeddings = await self._generate_embeddings(uncached_prompts)
                    
                    if embeddings is None:
                        raise ValueError("Failed to generate batch embeddings")
                    
                    # Process each prompt with its embedding
                    async with self.profiler.profile_operation("batch_analysis_processing"):
                        if self.config.enable_async and self.thread_pool:
                            # Parallel processing
                            tasks = []
                            for prompt, embedding in zip(uncached_prompts, embeddings):
                                task = asyncio.create_task(
                                    self._process_single_with_embedding(prompt, embedding, **kwargs)
                                )
                                tasks.append(task)
                            
                            uncached_results = await asyncio.gather(*tasks)
                        else:
                            # Sequential processing
                            for prompt, embedding in zip(uncached_prompts, embeddings):
                                result = await self._process_single_with_embedding(prompt, embedding, **kwargs)
                                uncached_results.append(result)
                
                # Combine cached and uncached results
                all_results = [None] * len(prompts)
                
                # Place cached results
                for i, result in cached_results.items():
                    all_results[i] = result
                
                # Place uncached results
                for i, result in zip(uncached_indices, uncached_results):
                    all_results[i] = result
                    
                    # Cache new results
                    if self.config.enable_result_caching:
                        self._cache_result(prompts[i], result)
                
                # Update batch metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                self.profiler.current_metrics.avg_batch_size = len(prompts)
                self.profiler.current_metrics.batch_utilization = len(uncached_prompts) / len(prompts)
                
                self.logger.debug(
                    "Batch processing completed",
                    batch_size=len(prompts),
                    cached_count=len(cached_results),
                    processed_count=len(uncached_prompts),
                    processing_time_ms=processing_time
                )
                
                return all_results
                
            except Exception as e:
                self.logger.error(
                    "Batch processing failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    batch_size=len(prompts)
                )
                raise
    
    async def _generate_embeddings(self, prompts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using active embedding plugin"""
        
        embedding_plugin = self.plugin_manager.get_active_embedding_plugin()
        if embedding_plugin is None:
            self.logger.warning("No active embedding plugin")
            return None
        
        try:
            if hasattr(embedding_plugin, 'encode_async'):
                embeddings = await embedding_plugin.encode_async(prompts)
            else:
                # Run in thread pool to avoid blocking
                if self.thread_pool:
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        self.thread_pool, 
                        embedding_plugin.encode, 
                        prompts
                    )
                else:
                    embeddings = embedding_plugin.encode(prompts)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
    
    async def _run_analysis_plugins(
        self, 
        prompt: str, 
        embedding: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Run all active analysis plugins"""
        
        analysis_plugins = self.plugin_manager.get_active_analysis_plugins()
        if not analysis_plugins:
            return {}
        
        results = {}
        
        if self.config.enable_async and len(analysis_plugins) > 1:
            # Run plugins in parallel
            tasks = []
            for plugin in analysis_plugins:
                task = asyncio.create_task(
                    plugin.analyze(prompt, embedding, **kwargs)
                )
                tasks.append((plugin.plugin_name, task))
            
            for plugin_name, task in tasks:
                try:
                    result = await task
                    results[plugin_name] = result
                except Exception as e:
                    self.logger.error(f"Analysis plugin {plugin_name} failed: {e}")
                    results[plugin_name] = {'error': str(e)}
        else:
            # Run plugins sequentially
            for plugin in analysis_plugins:
                try:
                    result = await plugin.analyze(prompt, embedding, **kwargs)
                    results[plugin.plugin_name] = result
                except Exception as e:
                    self.logger.error(f"Analysis plugin {plugin.plugin_name} failed: {e}")
                    results[plugin.plugin_name] = {'error': str(e)}
        
        return results
    
    async def _process_single_with_embedding(
        self, 
        prompt: str, 
        embedding: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Process single prompt with pre-computed embedding"""
        
        start_time = time.perf_counter()
        
        # Run analysis plugins
        analysis_results = await self._run_analysis_plugins(prompt, embedding, **kwargs)
        
        result = {
            'prompt': prompt,
            'embeddings': embedding.tolist(),
            'analysis_results': analysis_results,
            'processing_time_ms': (time.perf_counter() - start_time) * 1000,
            'metadata': {
                'processor_version': '1.0',
                'active_plugins': {
                    'embedding': self.plugin_manager.active_embedding_plugin,
                    'analysis': self.plugin_manager.active_analysis_plugins
                }
            }
        }
        
        return result
    
    def _get_cached_result(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached result for prompt"""
        if not self.config.enable_result_caching:
            return None
        
        cache_key = hash(prompt)
        
        with self.cache_lock:
            if cache_key in self.result_cache:
                result, timestamp = self.result_cache[cache_key]
                
                # Check TTL
                if time.time() - timestamp < self.config.cache_ttl_seconds:
                    self.profiler.current_metrics.cache_hit_rate = (
                        self.profiler.current_metrics.cache_hit_rate * 0.9 + 0.1
                    )
                    return result
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]
            
            # Cache miss
            self.profiler.current_metrics.cache_hit_rate *= 0.9
            return None
    
    def _cache_result(self, prompt: str, result: Dict[str, Any]) -> None:
        """Cache result for prompt"""
        if not self.config.enable_result_caching:
            return
        
        cache_key = hash(prompt)
        
        with self.cache_lock:
            # Check cache size limit
            if len(self.result_cache) >= self.config.cache_size:
                # Remove oldest entry (simple LRU)
                oldest_key = min(
                    self.result_cache.keys(),
                    key=lambda k: self.result_cache[k][1]
                )
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = (result, time.time())
            self.profiler.current_metrics.cache_size_used = len(self.result_cache)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = self.profiler.get_performance_summary()
        
        # Add processor-specific metrics
        metrics['processor'] = {
            'request_count': self.request_count,
            'uptime_seconds': time.time() - self.start_time,
            'cache_entries': len(self.result_cache),
            'active_plugins': {
                'embedding': self.plugin_manager.active_embedding_plugin,
                'analysis_count': len(self.plugin_manager.active_analysis_plugins)
            }
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown processor and cleanup resources"""
        self.logger.info("Shutting down ModularPSIProcessor")
        
        # Stop monitoring
        self.profiler.stop_monitoring()
        
        # Stop batch processing
        await self.batch_processor.stop_batch_processing()
        
        # Shutdown thread pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        self.logger.info("ModularPSIProcessor shutdown completed") 