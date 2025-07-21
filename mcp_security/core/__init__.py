"""
Core security analysis modules for MCP+ATCA Security Defense System.

This package provides the main security engines:
- PSI (Prompt Semantics Inspection): Token-level embedding analysis and adversarial detection
- TIADS (Tool Invocation Anomaly Detection System): Tool call pattern analysis
- PES (Prompt Evolution Scanner): Multi-turn conversation analysis  
- PIFF (Prompt Injection Firewall & Filter): Real-time prompt filtering

Each engine can operate independently or as part of an integrated defense system.
"""

from .psi import (
    PSIEngine,
    PSIResult,
    TokenEmbeddingAnalyzer,
    TokenAnalysisResult,
    PromptAnalysisResult,
    AdversarialDetector,
    AdversarialDetectionResult,
    EmbeddingShiftAttack,
    ContrastiveAdversarialPrompting,
    DatasetManager,
    PromptLabeler,
    Dataset,
    PromptSample,
    EvaluationFramework,
    PerformanceBenchmark,
    EvaluationResult,
    BenchmarkResult,
    EmbeddingCacheManager,
    LRUEmbeddingCache,
    PersistentEmbeddingCache,
    FAISSEmbeddingIndex,
    CacheEntry,
    CacheStats
)

# Placeholder imports for other security modules
from .tiads import TIADSEngine
from .pes import PESEngine
from .piff import PIFFEngine

__all__ = [
    # PSI Engine components
    "PSIEngine",
    "PSIResult",
    "TokenEmbeddingAnalyzer",
    "TokenAnalysisResult",
    "PromptAnalysisResult",
    "AdversarialDetector",
    "AdversarialDetectionResult",
    "EmbeddingShiftAttack",
    "ContrastiveAdversarialPrompting",
    "DatasetManager",
    "PromptLabeler",
    "Dataset",
    "PromptSample",
    "EvaluationFramework",
    "PerformanceBenchmark",
    "EvaluationResult",
    "BenchmarkResult",
    "EmbeddingCacheManager",
    "LRUEmbeddingCache",
    "PersistentEmbeddingCache",
    "FAISSEmbeddingIndex",
    "CacheEntry",
    "CacheStats",

    # Other security engines (placeholders)
    "TIADSEngine",
    "PESEngine",
    "PIFFEngine",
] 