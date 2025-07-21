"""
Prompt Semantics Inspection (PSI) Engine for MCP+ATCA Security Defense System.

This module provides comprehensive prompt analysis including:
- Token-level embedding analysis with semantic anomaly detection
- Advanced adversarial attack detection (ESA, CAP) using state-of-the-art ML techniques
- Multi-granularity detection with dynamic threshold adjustment
- Real-time performance optimization for sub-200ms processing
- Comprehensive evaluation framework with FNR reduction focus
- Efficient embedding caching system with multi-level storage
- Enhanced ensemble methods with uncertainty quantification
"""

from .engine import PSIEngine, PSIResult
from .analyzers import TokenEmbeddingAnalyzer, TokenAnalysisResult, PromptAnalysisResult
from .detectors import AdversarialDetector, AdversarialDetectionResult
from .attacks import EmbeddingShiftAttack, ContrastiveAdversarialPrompting
from .data import DatasetManager, PromptLabeler, Dataset, PromptSample
from .evaluation import EvaluationFramework, PerformanceBenchmark, EvaluationResult, BenchmarkResult
from .cache import (
    EmbeddingCacheManager,
    LRUEmbeddingCache,
    PersistentEmbeddingCache,
    FAISSEmbeddingIndex,
    CacheEntry,
    CacheStats
)

# Configurable anomaly detection system (Task 5.4)
try:
    from .anomaly_config import (
        ConfigurableAnomalyDetector,
        ThresholdAdapter,
        AnomalyConfigAPI,
        SecurityLevel,
        DetectionSignal,
        ThresholdConfig,
        FusionConfig,
        AnomalyDetectionConfig,
        DetectionResult
    )
    ANOMALY_CONFIG_AVAILABLE = True
except ImportError:
    ANOMALY_CONFIG_AVAILABLE = False

# Modular architecture and performance optimization (Task 5.5)
try:
    from .modular_architecture import (
        ModularPSIProcessor,
        PluginManager,
        BatchProcessor,
        PerformanceProfiler,
        EmbeddingModelPlugin,
        AnalysisPlugin,
        ProcessingConfig,
        PerformanceMetrics
    )
    MODULAR_ARCHITECTURE_AVAILABLE = True
except ImportError:
    MODULAR_ARCHITECTURE_AVAILABLE = False

# Enhanced detection components
try:
    from .enhanced_detectors import (
        EnhancedAdversarialDetector,
        EnhancedDetectionResult,
        DetectionFeatures,
        DynamicThresholds,
        AttackType
    )
    ENHANCED_DETECTORS_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTORS_AVAILABLE = False

# Advanced training components
try:
    from .adversarial_trainer import (
        AdversarialTrainer,
        TrainingConfig,
        TrainingResult,
        AdversarialDetectorNet,
        FocalLoss,
        FNRPenalizedLoss
    )
    ADVERSARIAL_TRAINER_AVAILABLE = True
except ImportError:
    ADVERSARIAL_TRAINER_AVAILABLE = False

__all__ = [
    # Core engine
    "PSIEngine",
    "PSIResult",

    # Analysis components
    "TokenEmbeddingAnalyzer",
    "TokenAnalysisResult",
    "PromptAnalysisResult",

    # Detection systems
    "AdversarialDetector",
    "AdversarialDetectionResult",

    # Attack simulation
    "EmbeddingShiftAttack",
    "ContrastiveAdversarialPrompting",

    # Data management
    "DatasetManager",
    "PromptLabeler",
    "Dataset",
    "PromptSample",

    # Evaluation tools
    "EvaluationFramework",
    "PerformanceBenchmark",
    "EvaluationResult",
    "BenchmarkResult",

    # Caching system
    "EmbeddingCacheManager",
    "LRUEmbeddingCache",
    "PersistentEmbeddingCache",
    "FAISSEmbeddingIndex",
    "CacheEntry",
    "CacheStats",
]

# Add configurable anomaly detection components if available
if ANOMALY_CONFIG_AVAILABLE:
    __all__.extend([
        "ConfigurableAnomalyDetector",
        "ThresholdAdapter",
        "AnomalyConfigAPI",
        "SecurityLevel",
        "DetectionSignal",
        "ThresholdConfig",
        "FusionConfig",
        "AnomalyDetectionConfig",
        "DetectionResult"
    ])

# Add modular architecture components if available
if MODULAR_ARCHITECTURE_AVAILABLE:
    __all__.extend([
        "ModularPSIProcessor",
        "PluginManager",
        "BatchProcessor",
        "PerformanceProfiler",
        "EmbeddingModelPlugin",
        "AnalysisPlugin",
        "ProcessingConfig",
        "PerformanceMetrics"
    ])

# Add enhanced components if available
if ENHANCED_DETECTORS_AVAILABLE:
    __all__.extend([
        "EnhancedAdversarialDetector",
        "EnhancedDetectionResult", 
        "DetectionFeatures",
        "DynamicThresholds",
        "AttackType"
    ])

if ADVERSARIAL_TRAINER_AVAILABLE:
    __all__.extend([
        "AdversarialTrainer",
        "TrainingConfig",
        "TrainingResult",
        "AdversarialDetectorNet",
        "FocalLoss",
        "FNRPenalizedLoss"
    ])

# Capability flags
__all__.extend([
    "ENHANCED_DETECTORS_AVAILABLE",
    "ADVERSARIAL_TRAINER_AVAILABLE",
    "ANOMALY_CONFIG_AVAILABLE",
    "MODULAR_ARCHITECTURE_AVAILABLE"
]) 