"""
Configurable Anomaly Detection System for PSI Engine

This module provides a flexible and adaptive anomaly detection framework including:
- Configurable thresholds with percentile-based adaptation
- Multi-signal fusion for improved accuracy
- Context-aware threshold adjustment
- Security engineer configuration API
- Real-time performance monitoring
- Sensitivity level management
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics
import math

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings

class SecurityLevel(Enum):
    """Security sensitivity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionSignal(Enum):
    """Types of detection signals"""
    EMBEDDING_DISTANCE = "embedding_distance"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOKEN_FREQUENCY = "token_frequency"
    CONTEXT_COHERENCE = "context_coherence"
    GRADIENT_MAGNITUDE = "gradient_magnitude"
    STATISTICAL_OUTLIER = "statistical_outlier"
    LINGUISTIC_PATTERN = "linguistic_pattern"
    ENSEMBLE_CONSENSUS = "ensemble_consensus"

@dataclass
class ThresholdConfig:
    """Configuration for detection thresholds"""
    
    # Basic threshold settings
    base_threshold: float = 0.5
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    
    # Adaptive settings
    adaptation_rate: float = 0.1
    adaptation_window: int = 1000
    percentile_target: float = 95.0
    sensitivity_multiplier: float = 1.0
    
    # Context-aware settings
    context_weight: float = 0.3
    temporal_decay: float = 0.95
    confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class FusionConfig:
    """Configuration for multi-signal fusion"""
    
    # Signal weights
    signal_weights: Dict[DetectionSignal, float] = field(default_factory=lambda: {
        DetectionSignal.EMBEDDING_DISTANCE: 0.25,
        DetectionSignal.SEMANTIC_SIMILARITY: 0.20,
        DetectionSignal.TOKEN_FREQUENCY: 0.15,
        DetectionSignal.CONTEXT_COHERENCE: 0.15,
        DetectionSignal.GRADIENT_MAGNITUDE: 0.10,
        DetectionSignal.STATISTICAL_OUTLIER: 0.10,
        DetectionSignal.LINGUISTIC_PATTERN: 0.05
    })
    
    # Fusion methods
    fusion_method: str = "weighted_average"  # "weighted_average", "majority_vote", "evidence_fusion"
    consensus_threshold: float = 0.6
    uncertainty_penalty: float = 0.2
    
    # Quality control
    min_signals_required: int = 3
    signal_reliability_weights: Dict[DetectionSignal, float] = field(default_factory=lambda: {
        signal: 1.0 for signal in DetectionSignal
    })
    
    def normalize_weights(self) -> None:
        """Normalize signal weights to sum to 1.0"""
        total_weight = sum(self.signal_weights.values())
        if total_weight > 0:
            for signal in self.signal_weights:
                self.signal_weights[signal] /= total_weight

@dataclass
class AnomalyDetectionConfig:
    """Complete anomaly detection configuration"""
    
    # Security level
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    # Threshold configurations by signal type
    thresholds: Dict[DetectionSignal, ThresholdConfig] = field(default_factory=dict)
    
    # Fusion configuration
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    
    # Performance settings
    max_processing_time_ms: float = 200.0
    enable_caching: bool = True
    batch_processing: bool = True
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_window: int = 10000
    performance_alerting: bool = True
    
    def __post_init__(self):
        """Initialize default thresholds for all signals"""
        if not self.thresholds:
            for signal in DetectionSignal:
                self.thresholds[signal] = ThresholdConfig()
        
        # Apply security level adjustments
        self._apply_security_level_adjustments()
        
        # Normalize fusion weights
        self.fusion_config.normalize_weights()
    
    def _apply_security_level_adjustments(self) -> None:
        """Apply security level-specific threshold adjustments"""
        level_multipliers = {
            SecurityLevel.LOW: 0.7,
            SecurityLevel.MEDIUM: 1.0,
            SecurityLevel.HIGH: 1.3,
            SecurityLevel.CRITICAL: 1.6
        }
        
        multiplier = level_multipliers[self.security_level]
        
        for threshold_config in self.thresholds.values():
            threshold_config.sensitivity_multiplier = multiplier
            threshold_config.base_threshold = min(
                threshold_config.base_threshold * multiplier,
                threshold_config.max_threshold
            )

@dataclass
class DetectionResult:
    """Result from anomaly detection"""
    
    is_anomalous: bool
    confidence: float
    overall_score: float
    signal_scores: Dict[DetectionSignal, float]
    signal_contributions: Dict[DetectionSignal, float]
    threshold_used: float
    context_factors: Dict[str, Any]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConfigurableAnomalyDetector:
    """
    Configurable anomaly detection system with adaptive thresholds.
    
    Features:
    - Multi-signal fusion with configurable weights
    - Adaptive threshold adjustment based on performance
    - Context-aware detection with temporal considerations
    - Security level-based sensitivity management
    - Real-time performance monitoring
    - Configuration API for security engineers
    """
    
    def __init__(self, config: AnomalyDetectionConfig, settings: Settings):
        """
        Initialize the configurable anomaly detector.
        
        Args:
            config: Anomaly detection configuration
            settings: System settings
        """
        self.config = config
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Performance tracking
        self.detection_history: deque = deque(maxlen=config.monitoring_window)
        self.performance_metrics: Dict[str, deque] = {
            'processing_time': deque(maxlen=1000),
            'accuracy': deque(maxlen=100),
            'precision': deque(maxlen=100),
            'recall': deque(maxlen=100),
            'f1_score': deque(maxlen=100)
        }
        
        # Adaptive components
        self.signal_scalers: Dict[DetectionSignal, StandardScaler] = {}
        self.threshold_adapters: Dict[DetectionSignal, 'ThresholdAdapter'] = {}
        
        # Context tracking
        self.context_history: deque = deque(maxlen=5000)
        self.signal_reliability: Dict[DetectionSignal, float] = {}
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(
            "ConfigurableAnomalyDetector initialized",
            security_level=config.security_level.value,
            num_signals=len(config.thresholds),
            fusion_method=config.fusion_config.fusion_method,
            max_processing_time=config.max_processing_time_ms
        )
    
    def _initialize_components(self) -> None:
        """Initialize detection components"""
        
        # Initialize signal scalers
        for signal in DetectionSignal:
            self.signal_scalers[signal] = StandardScaler()
        
        # Initialize threshold adapters
        for signal, threshold_config in self.config.thresholds.items():
            self.threshold_adapters[signal] = ThresholdAdapter(threshold_config)
        
        # Initialize signal reliability
        for signal in DetectionSignal:
            self.signal_reliability[signal] = 1.0
    
    async def detect_anomalies(
        self, 
        prompt: str,
        signal_values: Dict[DetectionSignal, float],
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect anomalies using multi-signal fusion.
        
        Args:
            prompt: Input prompt to analyze
            signal_values: Dictionary of signal values
            context: Optional context information
            
        Returns:
            DetectionResult with comprehensive analysis
        """
        start_time = time.perf_counter()
        
        try:
            self.logger.debug(
                "Starting configurable anomaly detection",
                prompt_length=len(prompt),
                num_signals=len(signal_values),
                security_level=self.config.security_level.value
            )
            
            # Validate and preprocess signals
            processed_signals = await self._preprocess_signals(signal_values)
            
            # Apply context-aware adjustments
            context_factors = await self._analyze_context(prompt, context)
            adjusted_signals = await self._apply_context_adjustments(
                processed_signals, context_factors
            )
            
            # Get adaptive thresholds
            adaptive_thresholds = await self._get_adaptive_thresholds(context_factors)
            
            # Calculate signal contributions
            signal_contributions = await self._calculate_signal_contributions(
                adjusted_signals, adaptive_thresholds
            )
            
            # Perform multi-signal fusion
            fusion_result = await self._fuse_signals(
                adjusted_signals, signal_contributions, adaptive_thresholds
            )
            
            # Determine final anomaly decision
            is_anomalous, confidence = await self._make_anomaly_decision(
                fusion_result, context_factors
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.performance_metrics['processing_time'].append(processing_time)
            
            # Create result
            result = DetectionResult(
                is_anomalous=is_anomalous,
                confidence=confidence,
                overall_score=fusion_result['overall_score'],
                signal_scores=adjusted_signals,
                signal_contributions=signal_contributions,
                threshold_used=fusion_result['threshold_used'],
                context_factors=context_factors,
                processing_time_ms=processing_time,
                metadata={
                    'security_level': self.config.security_level.value,
                    'fusion_method': self.config.fusion_config.fusion_method,
                    'num_signals_used': len(adjusted_signals)
                }
            )
            
            # Update detection history
            self.detection_history.append({
                'timestamp': time.time(),
                'prompt_length': len(prompt),
                'is_anomalous': is_anomalous,
                'confidence': confidence,
                'processing_time': processing_time,
                'signal_scores': adjusted_signals.copy()
            })
            
            # Log security event if anomaly detected
            if is_anomalous:
                self.logger.security_event(
                    event_type="configurable_anomaly_detected",
                    message=f"Anomaly detected with {confidence:.2f} confidence",
                    severity=self._get_severity_level(confidence),
                    resource="configurable_anomaly_detector",
                    action="anomaly_detection",
                    prompt_length=len(prompt),
                    confidence=confidence,
                    security_level=self.config.security_level.value,
                    processing_time_ms=processing_time,
                    signal_contributions=signal_contributions
                )
            
            return result
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.error(
                "Configurable anomaly detection failed",
                error_type=type(e).__name__,
                error_message=str(e),
                prompt_length=len(prompt),
                processing_time_ms=processing_time
            )
            
            # Return safe default
            return DetectionResult(
                is_anomalous=False,
                confidence=0.0,
                overall_score=0.0,
                signal_scores={},
                signal_contributions={},
                threshold_used=0.5,
                context_factors={},
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
    
    async def _preprocess_signals(
        self, 
        signal_values: Dict[DetectionSignal, float]
    ) -> Dict[DetectionSignal, float]:
        """Preprocess and normalize signal values"""
        
        processed = {}
        
        for signal, value in signal_values.items():
            try:
                # Clip to valid range
                clipped_value = np.clip(value, 0.0, 1.0)
                
                # Apply signal-specific scaling if available
                if signal in self.signal_scalers:
                    scaler = self.signal_scalers[signal]
                    if hasattr(scaler, 'scale_'):
                        # Use fitted scaler
                        scaled_value = scaler.transform([[clipped_value]])[0][0]
                        processed[signal] = np.clip(scaled_value, 0.0, 1.0)
                    else:
                        processed[signal] = clipped_value
                else:
                    processed[signal] = clipped_value
                    
            except Exception as e:
                self.logger.warning(f"Signal preprocessing failed for {signal}: {e}")
                processed[signal] = 0.0
        
        return processed
    
    async def _analyze_context(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze context factors that may affect detection"""
        
        context_factors = {
            'prompt_length': len(prompt),
            'word_count': len(prompt.split()),
            'has_context': context is not None,
            'time_of_day': time.strftime('%H:%M:%S'),
            'recent_anomaly_rate': self._calculate_recent_anomaly_rate()
        }
        
        # Analyze prompt characteristics
        context_factors.update({
            'avg_word_length': np.mean([len(word) for word in prompt.split()]) if prompt.split() else 0,
            'punctuation_density': sum(1 for c in prompt if c in '.,!?;:') / len(prompt) if prompt else 0,
            'uppercase_ratio': sum(1 for c in prompt if c.isupper()) / len(prompt) if prompt else 0,
            'special_char_ratio': sum(1 for c in prompt if not c.isalnum() and not c.isspace()) / len(prompt) if prompt else 0
        })
        
        # Add user-provided context
        if context:
            context_factors.update({
                'user_context': context,
                'context_keys': list(context.keys()) if isinstance(context, dict) else []
            })
        
        return context_factors
    
    def _calculate_recent_anomaly_rate(self) -> float:
        """Calculate recent anomaly detection rate"""
        if len(self.detection_history) < 10:
            return 0.0
        
        recent_detections = list(self.detection_history)[-100:]  # Last 100 detections
        anomaly_count = sum(1 for d in recent_detections if d['is_anomalous'])
        
        return anomaly_count / len(recent_detections)
    
    async def _apply_context_adjustments(
        self,
        signal_values: Dict[DetectionSignal, float],
        context_factors: Dict[str, Any]
    ) -> Dict[DetectionSignal, float]:
        """Apply context-aware adjustments to signal values"""
        
        adjusted = signal_values.copy()
        
        # Length-based adjustments
        length_factor = min(context_factors['prompt_length'] / 100, 1.5)  # Longer prompts get slight boost
        
        # Recent anomaly rate adjustments
        anomaly_rate = context_factors['recent_anomaly_rate']
        rate_factor = 1.0 + (anomaly_rate - 0.1) * 0.2  # Adjust based on recent pattern
        
        # Time-based adjustments (simplified)
        hour = int(context_factors['time_of_day'].split(':')[0])
        time_factor = 1.1 if hour < 6 or hour > 22 else 1.0  # Higher sensitivity at night
        
        # Apply adjustments
        for signal in adjusted:
            context_weight = self.config.thresholds[signal].context_weight
            
            # Weighted adjustment
            adjustment = (
                length_factor * 0.3 +
                rate_factor * 0.4 +
                time_factor * 0.3
            ) * context_weight
            
            adjusted[signal] = adjusted[signal] * adjustment
            adjusted[signal] = np.clip(adjusted[signal], 0.0, 1.0)
        
        return adjusted
    
    async def _get_adaptive_thresholds(
        self, 
        context_factors: Dict[str, Any]
    ) -> Dict[DetectionSignal, float]:
        """Get adaptive thresholds for each signal"""
        
        adaptive_thresholds = {}
        
        for signal, adapter in self.threshold_adapters.items():
            base_threshold = self.config.thresholds[signal].base_threshold
            
            # Apply adaptive adjustment
            adaptive_threshold = adapter.get_adaptive_threshold()
            
            # Apply context factors
            context_adjustment = self._calculate_context_threshold_adjustment(
                signal, context_factors
            )
            
            final_threshold = adaptive_threshold * context_adjustment
            
            # Ensure within bounds
            min_thresh = self.config.thresholds[signal].min_threshold
            max_thresh = self.config.thresholds[signal].max_threshold
            
            adaptive_thresholds[signal] = np.clip(final_threshold, min_thresh, max_thresh)
        
        return adaptive_thresholds
    
    def _calculate_context_threshold_adjustment(
        self,
        signal: DetectionSignal,
        context_factors: Dict[str, Any]
    ) -> float:
        """Calculate context-based threshold adjustment"""
        
        # Base adjustment
        adjustment = 1.0
        
        # Security level adjustment
        if self.config.security_level == SecurityLevel.CRITICAL:
            adjustment *= 0.8  # Lower thresholds for critical
        elif self.config.security_level == SecurityLevel.LOW:
            adjustment *= 1.2  # Higher thresholds for low
        
        # Signal reliability adjustment
        reliability = self.signal_reliability.get(signal, 1.0)
        adjustment *= (0.8 + 0.4 * reliability)  # Less reliable signals get higher thresholds
        
        return adjustment
    
    async def _calculate_signal_contributions(
        self,
        signal_values: Dict[DetectionSignal, float],
        thresholds: Dict[DetectionSignal, float]
    ) -> Dict[DetectionSignal, float]:
        """Calculate how much each signal contributes to the final decision"""
        
        contributions = {}
        
        for signal, value in signal_values.items():
            threshold = thresholds.get(signal, 0.5)
            
            # Calculate contribution based on how much the signal exceeds its threshold
            if value > threshold:
                excess = value - threshold
                max_excess = 1.0 - threshold
                contribution = excess / max_excess if max_excess > 0 else 0.0
            else:
                contribution = 0.0
            
            # Weight by signal importance and reliability
            signal_weight = self.config.fusion_config.signal_weights.get(signal, 0.1)
            reliability = self.signal_reliability.get(signal, 1.0)
            
            contributions[signal] = contribution * signal_weight * reliability
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for signal in contributions:
                contributions[signal] /= total_contribution
        
        return contributions
    
    async def _fuse_signals(
        self,
        signal_values: Dict[DetectionSignal, float],
        contributions: Dict[DetectionSignal, float],
        thresholds: Dict[DetectionSignal, float]
    ) -> Dict[str, Any]:
        """Fuse multiple detection signals"""
        
        fusion_method = self.config.fusion_config.fusion_method
        
        if fusion_method == "weighted_average":
            return await self._weighted_average_fusion(signal_values, contributions, thresholds)
        elif fusion_method == "majority_vote":
            return await self._majority_vote_fusion(signal_values, thresholds)
        elif fusion_method == "evidence_fusion":
            return await self._evidence_fusion(signal_values, contributions, thresholds)
        else:
            # Default to weighted average
            return await self._weighted_average_fusion(signal_values, contributions, thresholds)
    
    async def _weighted_average_fusion(
        self,
        signal_values: Dict[DetectionSignal, float],
        contributions: Dict[DetectionSignal, float],
        thresholds: Dict[DetectionSignal, float]
    ) -> Dict[str, Any]:
        """Weighted average fusion method"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for signal, value in signal_values.items():
            weight = self.config.fusion_config.signal_weights.get(signal, 0.1)
            reliability = self.signal_reliability.get(signal, 1.0)
            
            effective_weight = weight * reliability
            total_score += value * effective_weight
            total_weight += effective_weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate dynamic threshold
        weighted_threshold = sum(
            thresholds.get(signal, 0.5) * self.config.fusion_config.signal_weights.get(signal, 0.1)
            for signal in signal_values
        )
        
        return {
            'overall_score': overall_score,
            'threshold_used': weighted_threshold,
            'fusion_confidence': min(total_weight, 1.0)
        }
    
    async def _majority_vote_fusion(
        self,
        signal_values: Dict[DetectionSignal, float],
        thresholds: Dict[DetectionSignal, float]
    ) -> Dict[str, Any]:
        """Majority vote fusion method"""
        
        votes = 0
        total_signals = len(signal_values)
        
        for signal, value in signal_values.items():
            threshold = thresholds.get(signal, 0.5)
            if value > threshold:
                votes += 1
        
        vote_ratio = votes / total_signals if total_signals > 0 else 0.0
        consensus_threshold = self.config.fusion_config.consensus_threshold
        
        return {
            'overall_score': vote_ratio,
            'threshold_used': consensus_threshold,
            'fusion_confidence': abs(vote_ratio - 0.5) * 2  # Confidence based on how decisive the vote is
        }
    
    async def _evidence_fusion(
        self,
        signal_values: Dict[DetectionSignal, float],
        contributions: Dict[DetectionSignal, float],
        thresholds: Dict[DetectionSignal, float]
    ) -> Dict[str, Any]:
        """Evidence-based fusion using Dempster-Shafer theory (simplified)"""
        
        # Simplified evidence fusion
        evidence_for = 0.0
        evidence_against = 0.0
        uncertainty = 0.0
        
        for signal, value in signal_values.items():
            threshold = thresholds.get(signal, 0.5)
            reliability = self.signal_reliability.get(signal, 1.0)
            
            if value > threshold:
                evidence_for += (value - threshold) * reliability
            else:
                evidence_against += (threshold - value) * reliability
            
            uncertainty += (1.0 - reliability) * 0.1
        
        total_evidence = evidence_for + evidence_against + uncertainty
        
        if total_evidence > 0:
            normalized_evidence_for = evidence_for / total_evidence
            confidence = 1.0 - (uncertainty / total_evidence)
        else:
            normalized_evidence_for = 0.0
            confidence = 0.0
        
        return {
            'overall_score': normalized_evidence_for,
            'threshold_used': 0.5,  # Evidence fusion uses fixed threshold
            'fusion_confidence': confidence
        }
    
    async def _make_anomaly_decision(
        self,
        fusion_result: Dict[str, Any],
        context_factors: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Make final anomaly decision based on fusion result"""
        
        overall_score = fusion_result['overall_score']
        threshold = fusion_result['threshold_used']
        fusion_confidence = fusion_result['fusion_confidence']
        
        # Apply confidence threshold
        confidence_threshold = self.config.fusion_config.consensus_threshold
        
        if fusion_confidence < confidence_threshold:
            # Apply uncertainty penalty
            penalty = self.config.fusion_config.uncertainty_penalty
            effective_score = overall_score * (1.0 - penalty)
        else:
            effective_score = overall_score
        
        # Make decision
        is_anomalous = effective_score > threshold
        
        # Calculate final confidence
        if is_anomalous:
            confidence = min((effective_score - threshold) / (1.0 - threshold), 1.0) * fusion_confidence
        else:
            confidence = min((threshold - effective_score) / threshold, 1.0) * fusion_confidence
        
        return is_anomalous, confidence
    
    def _get_severity_level(self, confidence: float) -> str:
        """Get severity level based on confidence"""
        if confidence > 0.9:
            return "CRITICAL"
        elif confidence > 0.7:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def update_configuration(self, new_config: AnomalyDetectionConfig) -> None:
        """Update detector configuration"""
        old_level = self.config.security_level
        self.config = new_config
        
        # Reinitialize if security level changed
        if new_config.security_level != old_level:
            self._initialize_components()
        
        # Update threshold adapters
        for signal, threshold_config in new_config.thresholds.items():
            if signal in self.threshold_adapters:
                self.threshold_adapters[signal].update_config(threshold_config)
            else:
                self.threshold_adapters[signal] = ThresholdAdapter(threshold_config)
        
        self.logger.info(
            "Configuration updated",
            old_security_level=old_level.value,
            new_security_level=new_config.security_level.value,
            fusion_method=new_config.fusion_config.fusion_method
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            'detection_count': len(self.detection_history),
            'config': {
                'security_level': self.config.security_level.value,
                'fusion_method': self.config.fusion_config.fusion_method,
                'num_signals': len(self.config.thresholds)
            }
        }
        
        # Add performance metrics
        for metric_name, values in self.performance_metrics.items():
            if values:
                stats[f'avg_{metric_name}'] = np.mean(values)
                stats[f'median_{metric_name}'] = np.median(values)
                stats[f'95th_percentile_{metric_name}'] = np.percentile(values, 95)
        
        # Add recent anomaly rate
        stats['recent_anomaly_rate'] = self._calculate_recent_anomaly_rate()
        
        # Add threshold statistics
        stats['current_thresholds'] = {
            signal.value: adapter.get_adaptive_threshold()
            for signal, adapter in self.threshold_adapters.items()
        }
        
        return stats


class ThresholdAdapter:
    """Adaptive threshold management for individual signals"""
    
    def __init__(self, config: ThresholdConfig):
        """Initialize threshold adapter"""
        self.config = config
        self.current_threshold = config.base_threshold
        self.performance_history: deque = deque(maxlen=config.adaptation_window)
        
    def update_config(self, new_config: ThresholdConfig) -> None:
        """Update threshold configuration"""
        self.config = new_config
        self.current_threshold = max(
            min(self.current_threshold, new_config.max_threshold),
            new_config.min_threshold
        )
    
    def get_adaptive_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.current_threshold
    
    def update_performance(self, score: float, label: bool) -> None:
        """Update performance history for threshold adaptation"""
        self.performance_history.append({'score': score, 'label': label})
        
        # Adapt threshold if we have enough data
        if len(self.performance_history) >= 50:
            self._adapt_threshold()
    
    def _adapt_threshold(self) -> None:
        """Adapt threshold based on recent performance"""
        if len(self.performance_history) < 10:
            return
        
        # Get recent performance data
        scores = [h['score'] for h in self.performance_history]
        labels = [h['label'] for h in self.performance_history]
        
        positive_scores = [s for s, l in zip(scores, labels) if l]
        negative_scores = [s for s, l in zip(scores, labels) if not l]
        
        if not positive_scores or not negative_scores:
            return
        
        # Calculate optimal threshold using percentiles
        pos_percentile = np.percentile(positive_scores, self.config.percentile_target)
        neg_percentile = np.percentile(negative_scores, 100 - self.config.percentile_target)
        
        optimal_threshold = (pos_percentile + neg_percentile) / 2
        
        # Apply adaptation rate
        threshold_change = (optimal_threshold - self.current_threshold) * self.config.adaptation_rate
        new_threshold = self.current_threshold + threshold_change
        
        # Apply bounds and sensitivity multiplier
        new_threshold *= self.config.sensitivity_multiplier
        new_threshold = np.clip(new_threshold, self.config.min_threshold, self.config.max_threshold)
        
        self.current_threshold = new_threshold


class AnomalyConfigAPI:
    """Configuration API for security engineers"""
    
    def __init__(self, detector: ConfigurableAnomalyDetector):
        """Initialize configuration API"""
        self.detector = detector
        self.logger: SecurityLogger = get_logger(__name__)
    
    def set_security_level(self, level: SecurityLevel) -> Dict[str, Any]:
        """Set security sensitivity level"""
        old_level = self.detector.config.security_level
        self.detector.config.security_level = level
        self.detector.config._apply_security_level_adjustments()
        
        self.logger.info(
            "Security level changed",
            old_level=old_level.value,
            new_level=level.value
        )
        
        return {
            'status': 'success',
            'old_level': old_level.value,
            'new_level': level.value,
            'updated_thresholds': {
                signal.value: config.base_threshold
                for signal, config in self.detector.config.thresholds.items()
            }
        }
    
    def update_signal_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Update signal fusion weights"""
        try:
            # Convert string keys to DetectionSignal enums
            signal_weights = {}
            for signal_name, weight in weights.items():
                try:
                    signal = DetectionSignal(signal_name)
                    signal_weights[signal] = float(weight)
                except ValueError:
                    return {'status': 'error', 'message': f'Invalid signal: {signal_name}'}
            
            # Update weights
            self.detector.config.fusion_config.signal_weights.update(signal_weights)
            self.detector.config.fusion_config.normalize_weights()
            
            self.logger.info("Signal weights updated", new_weights=weights)
            
            return {
                'status': 'success',
                'updated_weights': {
                    signal.value: weight
                    for signal, weight in self.detector.config.fusion_config.signal_weights.items()
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def configure_threshold(
        self,
        signal: str,
        threshold_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure threshold for a specific signal"""
        try:
            signal_enum = DetectionSignal(signal)
            config = ThresholdConfig.from_dict(threshold_config)
            
            self.detector.config.thresholds[signal_enum] = config
            self.detector.threshold_adapters[signal_enum] = ThresholdAdapter(config)
            
            self.logger.info(
                "Threshold configured",
                signal=signal,
                config=threshold_config
            )
            
            return {'status': 'success', 'signal': signal, 'config': threshold_config}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def set_fusion_method(self, method: str) -> Dict[str, Any]:
        """Set signal fusion method"""
        valid_methods = ["weighted_average", "majority_vote", "evidence_fusion"]
        
        if method not in valid_methods:
            return {
                'status': 'error',
                'message': f'Invalid fusion method. Valid options: {valid_methods}'
            }
        
        old_method = self.detector.config.fusion_config.fusion_method
        self.detector.config.fusion_config.fusion_method = method
        
        self.logger.info(
            "Fusion method changed",
            old_method=old_method,
            new_method=method
        )
        
        return {'status': 'success', 'old_method': old_method, 'new_method': method}
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        config = self.detector.config
        
        return {
            'security_level': config.security_level.value,
            'fusion_method': config.fusion_config.fusion_method,
            'signal_weights': {
                signal.value: weight
                for signal, weight in config.fusion_config.signal_weights.items()
            },
            'thresholds': {
                signal.value: threshold_config.to_dict()
                for signal, threshold_config in config.thresholds.items()
            },
            'performance_settings': {
                'max_processing_time_ms': config.max_processing_time_ms,
                'enable_caching': config.enable_caching,
                'batch_processing': config.batch_processing
            }
        }
    
    def export_config(self, filepath: Path) -> Dict[str, Any]:
        """Export current configuration to file"""
        try:
            config_dict = self.get_current_config()
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info("Configuration exported", filepath=str(filepath))
            
            return {'status': 'success', 'filepath': str(filepath)}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def import_config(self, filepath: Path) -> Dict[str, Any]:
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Apply configuration
            if 'security_level' in config_dict:
                self.set_security_level(SecurityLevel(config_dict['security_level']))
            
            if 'fusion_method' in config_dict:
                self.set_fusion_method(config_dict['fusion_method'])
            
            if 'signal_weights' in config_dict:
                self.update_signal_weights(config_dict['signal_weights'])
            
            if 'thresholds' in config_dict:
                for signal, threshold_config in config_dict['thresholds'].items():
                    self.configure_threshold(signal, threshold_config)
            
            self.logger.info("Configuration imported", filepath=str(filepath))
            
            return {'status': 'success', 'filepath': str(filepath)}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)} 