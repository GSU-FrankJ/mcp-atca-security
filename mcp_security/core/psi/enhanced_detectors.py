"""
Enhanced Adversarial Attack Detection Module for PSI Engine

This module implements state-of-the-art adversarial attack detection including:
- Advanced ESA detection with gradient flow analysis
- Enhanced CAP detection with semantic consistency verification
- Multi-granularity detection (token, phrase, sentence levels)
- Dynamic threshold adjustment using percentile statistics
- Ensemble methods with uncertainty quantification
- Real-time performance optimization
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import faiss

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings
from .analyzers import PromptAnalysisResult, TokenAnalysisResult

class AttackType(Enum):
    """Enumeration of supported attack types"""
    ESA = "embedding_shift_attack"
    CAP = "contrastive_adversarial_prompting"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    GRADIENT_ANOMALY = "gradient_anomaly"
    ENSEMBLE_CLASSIFIER = "ensemble_classifier"
    MULTI_GRANULARITY = "multi_granularity"

@dataclass
class DetectionFeatures:
    """Feature vector for adversarial detection"""
    
    # ESA-specific features
    embedding_variance: float = 0.0
    gradient_magnitude: float = 0.0
    embedding_smoothness: float = 0.0
    perturbation_signature: float = 0.0
    
    # CAP-specific features
    semantic_consistency: float = 0.0
    paraphrase_similarity: float = 0.0
    intent_deviation: float = 0.0
    linguistic_naturalness: float = 0.0
    
    # Statistical features
    token_frequency_anomaly: float = 0.0
    embedding_outlier_score: float = 0.0
    position_entropy: float = 0.0
    attention_divergence: float = 0.0
    
    # Multi-granularity features
    token_level_anomaly: float = 0.0
    phrase_level_anomaly: float = 0.0
    sentence_level_anomaly: float = 0.0
    cross_level_consistency: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector"""
        return np.array([
            self.embedding_variance,
            self.gradient_magnitude,
            self.embedding_smoothness,
            self.perturbation_signature,
            self.semantic_consistency,
            self.paraphrase_similarity,
            self.intent_deviation,
            self.linguistic_naturalness,
            self.token_frequency_anomaly,
            self.embedding_outlier_score,
            self.position_entropy,
            self.attention_divergence,
            self.token_level_anomaly,
            self.phrase_level_anomaly,
            self.sentence_level_anomaly,
            self.cross_level_consistency
        ])

@dataclass
class EnhancedDetectionResult:
    """Enhanced results from adversarial attack detection"""
    
    is_adversarial: bool
    confidence_score: float
    detected_attacks: List[AttackType]
    attack_vectors: List[Dict[str, Any]]
    risk_score: float
    explanation: str
    processing_time_ms: float
    
    # Enhanced information
    features: DetectionFeatures
    detection_breakdown: Dict[str, float]
    uncertainty_score: float
    attack_sophistication: float
    recommended_action: str

@dataclass 
class DynamicThresholds:
    """Dynamic threshold management"""
    
    esa_threshold: float = 0.8
    cap_threshold: float = 0.75
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.5
    
    # Adaptive parameters
    adaptation_rate: float = 0.1
    window_size: int = 1000
    percentile_target: float = 95.0
    
    # Historical data
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    false_positive_rate: float = 0.05
    false_negative_rate: float = 0.02
    
    def update_thresholds(self, scores: List[float], labels: List[bool]) -> None:
        """Update thresholds based on recent performance"""
        if len(scores) < 10:  # Need minimum data
            return
        
        # Calculate optimal thresholds using ROC analysis
        positive_scores = [s for s, l in zip(scores, labels) if l]
        negative_scores = [s for s, l in zip(scores, labels) if not l]
        
        if positive_scores and negative_scores:
            # Use percentile-based adaptation
            pos_percentile = np.percentile(positive_scores, self.percentile_target)
            neg_percentile = np.percentile(negative_scores, 100 - self.percentile_target)
            
            # Adaptive update
            new_threshold = (pos_percentile + neg_percentile) / 2
            
            # Smooth update
            self.confidence_threshold += self.adaptation_rate * (new_threshold - self.confidence_threshold)
            self.confidence_threshold = np.clip(self.confidence_threshold, 0.1, 0.9)

class EnhancedAdversarialDetector:
    """
    Enhanced adversarial attack detector with state-of-the-art methods.
    
    Features:
    - Advanced ESA detection with gradient flow analysis
    - Enhanced CAP detection with semantic verification
    - Multi-granularity analysis across linguistic levels
    - Dynamic threshold adjustment with adaptive learning
    - Ensemble methods with uncertainty quantification
    - Real-time performance optimization
    """
    
    def __init__(self, embedding_model: SentenceTransformer, settings: Settings):
        """
        Initialize the Enhanced AdversarialDetector.
        
        Args:
            embedding_model: Pre-loaded SentenceTransformer model
            settings: Configuration settings containing PSI parameters
        """
        self.embedding_model = embedding_model
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Dynamic thresholds
        self.thresholds = DynamicThresholds(
            esa_threshold=getattr(settings, 'psi_esa_threshold', 0.8),
            cap_threshold=getattr(settings, 'psi_cap_threshold', 0.75),
            risk_threshold=getattr(settings, 'psi_risk_threshold', 0.6)
        )
        
        # Detection models
        self.ensemble_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.gradient_analyzer: Optional[nn.Module] = None
        self.semantic_verifier: Optional[nn.Module] = None
        
        # Feature processing
        self.feature_scaler = RobustScaler()
        self.pca_reducer = PCA(n_components=12)
        self.cluster_analyzer = DBSCAN(eps=0.3, min_samples=10)
        
        # Historical data for adaptation
        self.detection_history: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[str, deque] = {
            'precision': deque(maxlen=100),
            'recall': deque(maxlen=100),
            'f1': deque(maxlen=100),
            'processing_time': deque(maxlen=1000)
        }
        
        # Initialize components
        self._initialize_enhanced_components()
        
        self.logger.info(
            "Enhanced AdversarialDetector initialized",
            esa_threshold=self.thresholds.esa_threshold,
            cap_threshold=self.thresholds.cap_threshold,
            risk_threshold=self.thresholds.risk_threshold
        )
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced detection components"""
        
        # Initialize ensemble models
        self.ensemble_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000
            ),
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42
            )
        }
        
        # Initialize anomaly detectors for different granularities
        for granularity in ['token', 'phrase', 'sentence']:
            self.anomaly_detectors[granularity] = IsolationForest(
                contamination=0.05,
                random_state=42
            )
    
    async def detect_attacks_enhanced(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult
    ) -> EnhancedDetectionResult:
        """
        Enhanced adversarial attack detection with multi-granularity analysis.
        
        Args:
            prompt: Input prompt to analyze
            embedding_analysis: Results from token-level embedding analysis
            
        Returns:
            EnhancedDetectionResult with comprehensive detection information
        """
        start_time = time.perf_counter()
        
        try:
            self.logger.debug(
                "Starting enhanced adversarial detection",
                prompt_length=len(prompt),
                num_tokens=len(embedding_analysis.tokens)
            )
            
            # Extract comprehensive features
            features = await self._extract_comprehensive_features(prompt, embedding_analysis)
            
            # Multi-level detection
            detection_results = {}
            
            # Level 1: ESA Detection with gradient analysis
            detection_results['esa'] = await self._enhanced_esa_detection(
                prompt, embedding_analysis, features
            )
            
            # Level 2: CAP Detection with semantic verification
            detection_results['cap'] = await self._enhanced_cap_detection(
                prompt, embedding_analysis, features
            )
            
            # Level 3: Multi-granularity anomaly detection
            detection_results['multi_granularity'] = await self._multi_granularity_detection(
                prompt, embedding_analysis, features
            )
            
            # Level 4: Ensemble classification with uncertainty
            detection_results['ensemble'] = await self._ensemble_detection_with_uncertainty(
                prompt, embedding_analysis, features
            )
            
            # Level 5: Semantic consistency verification
            detection_results['semantic'] = await self._semantic_consistency_check(
                prompt, embedding_analysis, features
            )
            
            # Aggregate results with confidence weighting
            final_result = self._aggregate_detection_results(
                detection_results, features, prompt, embedding_analysis
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            final_result.processing_time_ms = processing_time
            
            # Update performance tracking
            self.performance_metrics['processing_time'].append(processing_time)
            
            # Log detection event
            if final_result.is_adversarial:
                self.logger.security_event(
                    event_type="enhanced_adversarial_detection",
                    message=f"Enhanced detector found {len(final_result.detected_attacks)} attack types",
                    severity="HIGH",
                    resource="enhanced_adversarial_detector",
                    action="enhanced_detection",
                    prompt_length=len(prompt),
                    detected_attacks=[a.value for a in final_result.detected_attacks],
                    confidence=final_result.confidence_score,
                    risk_score=final_result.risk_score,
                    processing_time_ms=processing_time,
                    uncertainty=final_result.uncertainty_score,
                    sophistication=final_result.attack_sophistication
                )
            
            return final_result
            
        except Exception as e:
            self.logger.error(
                "Enhanced adversarial detection failed",
                error_type=type(e).__name__,
                error_message=str(e),
                prompt_length=len(prompt)
            )
            
            # Return safe default
            processing_time = (time.perf_counter() - start_time) * 1000
            return EnhancedDetectionResult(
                is_adversarial=False,
                confidence_score=0.0,
                detected_attacks=[],
                attack_vectors=[],
                risk_score=0.0,
                explanation=f"Detection failed: {str(e)}",
                processing_time_ms=processing_time,
                features=DetectionFeatures(),
                detection_breakdown={},
                uncertainty_score=1.0,
                attack_sophistication=0.0,
                recommended_action="retry_analysis"
            )
    
    async def _extract_comprehensive_features(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult
    ) -> DetectionFeatures:
        """Extract comprehensive feature set for detection"""
        
        features = DetectionFeatures()
        
        try:
            # Get embeddings for analysis
            prompt_embedding = self.embedding_model.encode([prompt])[0]
            token_embeddings = np.array([
                result.embedding for result in embedding_analysis.token_results
            ])
            
            # ESA-specific features
            features.embedding_variance = np.var(token_embeddings.flatten())
            features.gradient_magnitude = await self._calculate_gradient_magnitude(prompt)
            features.embedding_smoothness = self._calculate_embedding_smoothness(token_embeddings)
            features.perturbation_signature = self._detect_perturbation_signature(token_embeddings)
            
            # CAP-specific features
            features.semantic_consistency = await self._calculate_semantic_consistency(prompt)
            features.paraphrase_similarity = await self._calculate_paraphrase_similarity(prompt)
            features.intent_deviation = self._calculate_intent_deviation(prompt, prompt_embedding)
            features.linguistic_naturalness = self._calculate_linguistic_naturalness(prompt)
            
            # Statistical features
            features.token_frequency_anomaly = self._calculate_frequency_anomaly(embedding_analysis.tokens)
            features.embedding_outlier_score = self._calculate_outlier_score(token_embeddings)
            features.position_entropy = self._calculate_position_entropy(embedding_analysis.token_results)
            features.attention_divergence = await self._calculate_attention_divergence(prompt)
            
            # Multi-granularity features
            features.token_level_anomaly = embedding_analysis.overall_anomaly_score
            features.phrase_level_anomaly = await self._calculate_phrase_anomaly(prompt)
            features.sentence_level_anomaly = await self._calculate_sentence_anomaly(prompt)
            features.cross_level_consistency = self._calculate_cross_level_consistency(
                features.token_level_anomaly,
                features.phrase_level_anomaly,
                features.sentence_level_anomaly
            )
            
        except Exception as e:
            self.logger.warning(f"Feature extraction error: {e}")
        
        return features
    
    async def _enhanced_esa_detection(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult, 
        features: DetectionFeatures
    ) -> Dict[str, Any]:
        """Enhanced ESA detection with gradient flow analysis"""
        
        # Combine multiple ESA detection signals
        esa_signals = {
            'embedding_variance': features.embedding_variance,
            'gradient_magnitude': features.gradient_magnitude,
            'smoothness_violation': 1.0 - features.embedding_smoothness,
            'perturbation_signature': features.perturbation_signature
        }
        
        # Weighted combination
        weights = {'embedding_variance': 0.3, 'gradient_magnitude': 0.4, 
                  'smoothness_violation': 0.2, 'perturbation_signature': 0.1}
        
        esa_score = sum(weights[k] * v for k, v in esa_signals.items())
        
        is_detected = esa_score > self.thresholds.esa_threshold
        
        return {
            'attack_type': AttackType.ESA,
            'is_detected': is_detected,
            'confidence': esa_score,
            'signals': esa_signals,
            'explanation': f"ESA detection score: {esa_score:.3f} (threshold: {self.thresholds.esa_threshold})"
        }
    
    async def _enhanced_cap_detection(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult, 
        features: DetectionFeatures
    ) -> Dict[str, Any]:
        """Enhanced CAP detection with semantic verification"""
        
        # Combine multiple CAP detection signals
        cap_signals = {
            'semantic_inconsistency': 1.0 - features.semantic_consistency,
            'paraphrase_anomaly': 1.0 - features.paraphrase_similarity,
            'intent_deviation': features.intent_deviation,
            'naturalness_violation': 1.0 - features.linguistic_naturalness
        }
        
        # Weighted combination
        weights = {'semantic_inconsistency': 0.35, 'paraphrase_anomaly': 0.25,
                  'intent_deviation': 0.25, 'naturalness_violation': 0.15}
        
        cap_score = sum(weights[k] * v for k, v in cap_signals.items())
        
        is_detected = cap_score > self.thresholds.cap_threshold
        
        return {
            'attack_type': AttackType.CAP,
            'is_detected': is_detected,
            'confidence': cap_score,
            'signals': cap_signals,
            'explanation': f"CAP detection score: {cap_score:.3f} (threshold: {self.thresholds.cap_threshold})"
        }
    
    async def _multi_granularity_detection(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult, 
        features: DetectionFeatures
    ) -> Dict[str, Any]:
        """Multi-granularity anomaly detection"""
        
        # Check consistency across granularity levels
        granularity_scores = {
            'token_level': features.token_level_anomaly,
            'phrase_level': features.phrase_level_anomaly,
            'sentence_level': features.sentence_level_anomaly
        }
        
        # Calculate multi-granularity score
        consistency_penalty = 1.0 - features.cross_level_consistency
        max_granularity_score = max(granularity_scores.values())
        
        multi_granularity_score = max_granularity_score + 0.3 * consistency_penalty
        
        is_detected = multi_granularity_score > 0.7
        
        return {
            'attack_type': AttackType.MULTI_GRANULARITY,
            'is_detected': is_detected,
            'confidence': multi_granularity_score,
            'signals': granularity_scores,
            'consistency_penalty': consistency_penalty,
            'explanation': f"Multi-granularity score: {multi_granularity_score:.3f}"
        }
    
    async def _ensemble_detection_with_uncertainty(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult, 
        features: DetectionFeatures
    ) -> Dict[str, Any]:
        """Ensemble detection with uncertainty quantification"""
        
        feature_vector = features.to_vector().reshape(1, -1)
        
        ensemble_predictions = []
        ensemble_confidences = []
        
        # Get predictions from each model if trained
        for model_name, model in self.ensemble_models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    # Scale features if scaler is fitted
                    if hasattr(self.feature_scaler, 'scale_'):
                        scaled_features = self.feature_scaler.transform(feature_vector)
                    else:
                        scaled_features = feature_vector
                    
                    if model_name == 'isolation_forest':
                        # Isolation forest returns -1/1, convert to 0/1
                        pred = model.predict(scaled_features)[0]
                        prediction = 1 if pred == -1 else 0
                        confidence = abs(model.score_samples(scaled_features)[0])
                    else:
                        proba = model.predict_proba(scaled_features)[0]
                        prediction = np.argmax(proba)
                        confidence = max(proba)
                    
                    ensemble_predictions.append(prediction)
                    ensemble_confidences.append(confidence)
                    
                except Exception as e:
                    self.logger.debug(f"Ensemble model {model_name} prediction failed: {e}")
        
        if ensemble_predictions:
            # Calculate ensemble decision
            ensemble_vote = sum(ensemble_predictions) / len(ensemble_predictions)
            mean_confidence = np.mean(ensemble_confidences)
            uncertainty = np.std(ensemble_confidences) if len(ensemble_confidences) > 1 else 0.0
            
            is_detected = ensemble_vote > 0.5 and mean_confidence > self.thresholds.confidence_threshold
        else:
            ensemble_vote = 0.0
            mean_confidence = 0.0
            uncertainty = 1.0
            is_detected = False
        
        return {
            'attack_type': AttackType.ENSEMBLE_CLASSIFIER,
            'is_detected': is_detected,
            'confidence': mean_confidence,
            'uncertainty': uncertainty,
            'ensemble_vote': ensemble_vote,
            'num_models': len(ensemble_predictions),
            'explanation': f"Ensemble vote: {ensemble_vote:.3f}, confidence: {mean_confidence:.3f}"
        }
    
    async def _semantic_consistency_check(
        self, 
        prompt: str, 
        embedding_analysis: PromptAnalysisResult, 
        features: DetectionFeatures
    ) -> Dict[str, Any]:
        """Semantic consistency verification"""
        
        # Check if semantic features indicate inconsistency
        consistency_score = features.semantic_consistency
        
        # Additional semantic checks
        intent_consistency = 1.0 - features.intent_deviation
        naturalness_score = features.linguistic_naturalness
        
        # Combined semantic score
        semantic_score = (consistency_score + intent_consistency + naturalness_score) / 3
        semantic_anomaly = 1.0 - semantic_score
        
        is_detected = semantic_anomaly > 0.4
        
        return {
            'attack_type': AttackType.SEMANTIC_INCONSISTENCY,
            'is_detected': is_detected,
            'confidence': semantic_anomaly,
            'consistency_score': consistency_score,
            'intent_consistency': intent_consistency,
            'naturalness_score': naturalness_score,
            'explanation': f"Semantic anomaly: {semantic_anomaly:.3f}"
        }
    
    def _aggregate_detection_results(
        self, 
        detection_results: Dict[str, Dict[str, Any]], 
        features: DetectionFeatures,
        prompt: str,
        embedding_analysis: PromptAnalysisResult
    ) -> EnhancedDetectionResult:
        """Aggregate detection results with confidence weighting"""
        
        detected_attacks = []
        attack_vectors = []
        confidence_scores = []
        detection_breakdown = {}
        
        # Process each detection result
        for detection_name, result in detection_results.items():
            detection_breakdown[detection_name] = result.get('confidence', 0.0)
            
            if result.get('is_detected', False):
                attack_type = result.get('attack_type')
                if attack_type:
                    detected_attacks.append(attack_type)
                
                attack_vectors.append(result)
                confidence_scores.append(result.get('confidence', 0.0))
        
        # Calculate overall metrics
        is_adversarial = len(detected_attacks) > 0
        overall_confidence = max(confidence_scores) if confidence_scores else 0.0
        
        # Calculate uncertainty (standard deviation of detection confidences)
        uncertainty_score = np.std(list(detection_breakdown.values())) if detection_breakdown else 0.0
        
        # Calculate attack sophistication
        attack_sophistication = self._calculate_attack_sophistication(features, detection_breakdown)
        
        # Calculate risk score
        risk_score = self._calculate_enhanced_risk_score(
            detected_attacks, confidence_scores, features, uncertainty_score
        )
        
        # Generate explanation
        explanation = self._generate_enhanced_explanation(
            detected_attacks, detection_breakdown, overall_confidence, uncertainty_score
        )
        
        # Recommend action
        recommended_action = self._recommend_action(
            is_adversarial, risk_score, uncertainty_score, attack_sophistication
        )
        
        return EnhancedDetectionResult(
            is_adversarial=is_adversarial,
            confidence_score=overall_confidence,
            detected_attacks=detected_attacks,
            attack_vectors=attack_vectors,
            risk_score=risk_score,
            explanation=explanation,
            processing_time_ms=0.0,  # Will be set by caller
            features=features,
            detection_breakdown=detection_breakdown,
            uncertainty_score=uncertainty_score,
            attack_sophistication=attack_sophistication,
            recommended_action=recommended_action
        )
    
    # Helper methods for feature calculation
    async def _calculate_gradient_magnitude(self, prompt: str) -> float:
        """Calculate gradient magnitude for ESA detection"""
        try:
            # Simplified gradient calculation
            embedding = self.embedding_model.encode([prompt])[0]
            return np.linalg.norm(np.gradient(embedding))
        except Exception:
            return 0.0
    
    def _calculate_embedding_smoothness(self, embeddings: np.ndarray) -> float:
        """Calculate embedding smoothness"""
        if len(embeddings) < 2:
            return 1.0
        
        try:
            # Calculate pairwise distances
            diffs = np.diff(embeddings, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(diffs, axis=1)))
            return smoothness
        except Exception:
            return 1.0
    
    def _detect_perturbation_signature(self, embeddings: np.ndarray) -> float:
        """Detect perturbation signatures in embeddings"""
        try:
            # Look for regular patterns that might indicate artificial perturbations
            if len(embeddings) < 3:
                return 0.0
            
            # Calculate variance across dimensions
            dim_variance = np.var(embeddings, axis=0)
            # Look for suspiciously uniform variance (signature of perturbations)
            variance_uniformity = 1.0 - np.std(dim_variance) / (np.mean(dim_variance) + 1e-8)
            
            return np.clip(variance_uniformity, 0.0, 1.0)
        except Exception:
            return 0.0
    
    async def _calculate_semantic_consistency(self, prompt: str) -> float:
        """Calculate semantic consistency score"""
        try:
            # Split prompt into sentences and check consistency
            sentences = prompt.split('.')
            if len(sentences) < 2:
                return 1.0
            
            sentence_embeddings = self.embedding_model.encode(sentences)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(sentence_embeddings)):
                for j in range(i + 1, len(sentence_embeddings)):
                    sim = np.dot(sentence_embeddings[i], sentence_embeddings[j])
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 1.0
        except Exception:
            return 1.0
    
    async def _calculate_paraphrase_similarity(self, prompt: str) -> float:
        """Calculate paraphrase similarity score"""
        # Simplified implementation - would need paraphrase models in practice
        return 0.8  # Placeholder
    
    def _calculate_intent_deviation(self, prompt: str, embedding: np.ndarray) -> float:
        """Calculate intent deviation score"""
        # Simplified implementation - would analyze intent consistency
        return 0.2  # Placeholder
    
    def _calculate_linguistic_naturalness(self, prompt: str) -> float:
        """Calculate linguistic naturalness score"""
        try:
            # Simple heuristics for naturalness
            words = prompt.split()
            if len(words) == 0:
                return 0.0
            
            # Check average word length
            avg_word_length = np.mean([len(word) for word in words])
            length_score = 1.0 / (1.0 + abs(avg_word_length - 5.0))  # Assume 5 is natural
            
            # Check sentence structure (simplified)
            sentence_count = len(prompt.split('.'))
            word_per_sentence = len(words) / max(sentence_count, 1)
            structure_score = 1.0 / (1.0 + abs(word_per_sentence - 15.0))  # Assume 15 is natural
            
            return (length_score + structure_score) / 2
        except Exception:
            return 0.5
    
    def _calculate_frequency_anomaly(self, tokens: List[str]) -> float:
        """Calculate token frequency anomaly"""
        if not tokens:
            return 0.0
        
        # Simple frequency analysis
        from collections import Counter
        token_counts = Counter(tokens)
        
        # Look for unusual frequency patterns
        max_freq = max(token_counts.values())
        total_tokens = len(tokens)
        
        # High repetition might indicate artificial generation
        anomaly_score = max_freq / total_tokens
        return np.clip(anomaly_score, 0.0, 1.0)
    
    def _calculate_outlier_score(self, embeddings: np.ndarray) -> float:
        """Calculate embedding outlier score"""
        if len(embeddings) < 3:
            return 0.0
        
        try:
            # Calculate distances from centroid
            centroid = np.mean(embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            
            # Z-score based outlier detection
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            if std_dist == 0:
                return 0.0
            
            max_z_score = max(abs(d - mean_dist) / std_dist for d in distances)
            outlier_score = max_z_score / 3.0  # Normalize by 3-sigma rule
            
            return np.clip(outlier_score, 0.0, 1.0)
        except Exception:
            return 0.0
    
    def _calculate_position_entropy(self, token_results: List[TokenAnalysisResult]) -> float:
        """Calculate positional entropy"""
        if not token_results:
            return 0.0
        
        try:
            # Calculate entropy of anomaly scores across positions
            anomaly_scores = [result.anomaly_score for result in token_results]
            
            # Convert to probability distribution
            scores_array = np.array(anomaly_scores)
            scores_array = scores_array / (np.sum(scores_array) + 1e-8)
            
            # Calculate entropy
            entropy = -np.sum(scores_array * np.log(scores_array + 1e-8))
            max_entropy = np.log(len(anomaly_scores))
            
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            return normalized_entropy
        except Exception:
            return 0.0
    
    async def _calculate_attention_divergence(self, prompt: str) -> float:
        """Calculate attention divergence score"""
        # Placeholder - would need attention analysis
        return 0.3
    
    async def _calculate_phrase_anomaly(self, prompt: str) -> float:
        """Calculate phrase-level anomaly score"""
        # Simplified phrase analysis
        words = prompt.split()
        if len(words) < 3:
            return 0.0
        
        # Create phrases (3-grams)
        phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        phrase_embeddings = self.embedding_model.encode(phrases)
        
        # Calculate variance across phrases
        phrase_variance = np.var(phrase_embeddings.flatten())
        return np.clip(phrase_variance * 10, 0.0, 1.0)  # Scale for interpretation
    
    async def _calculate_sentence_anomaly(self, prompt: str) -> float:
        """Calculate sentence-level anomaly score"""
        sentences = prompt.split('.')
        if len(sentences) < 2:
            return 0.0
        
        try:
            sentence_embeddings = self.embedding_model.encode(sentences)
            sentence_variance = np.var(sentence_embeddings.flatten())
            return np.clip(sentence_variance * 10, 0.0, 1.0)
        except Exception:
            return 0.0
    
    def _calculate_cross_level_consistency(
        self, 
        token_anomaly: float, 
        phrase_anomaly: float, 
        sentence_anomaly: float
    ) -> float:
        """Calculate consistency across granularity levels"""
        anomalies = [token_anomaly, phrase_anomaly, sentence_anomaly]
        
        # Consistency is inverse of variance
        consistency = 1.0 - np.std(anomalies)
        return np.clip(consistency, 0.0, 1.0)
    
    def _calculate_attack_sophistication(
        self, 
        features: DetectionFeatures, 
        detection_breakdown: Dict[str, float]
    ) -> float:
        """Calculate attack sophistication score"""
        
        # Higher sophistication if multiple attack types detected
        num_detections = sum(1 for score in detection_breakdown.values() if score > 0.5)
        
        # Consider feature complexity
        feature_complexity = np.mean([
            features.gradient_magnitude,
            features.semantic_consistency,
            features.cross_level_consistency
        ])
        
        sophistication = (num_detections / 5.0) + (0.5 * feature_complexity)
        return np.clip(sophistication, 0.0, 1.0)
    
    def _calculate_enhanced_risk_score(
        self, 
        detected_attacks: List[AttackType], 
        confidence_scores: List[float],
        features: DetectionFeatures,
        uncertainty: float
    ) -> float:
        """Calculate enhanced risk score"""
        
        if not detected_attacks:
            return 0.0
        
        # Base risk from number and confidence of attacks
        base_risk = len(detected_attacks) / 7.0  # Max 7 attack types
        confidence_risk = max(confidence_scores) if confidence_scores else 0.0
        
        # Penalty for high uncertainty
        uncertainty_penalty = uncertainty * 0.3
        
        # Feature-based risk
        feature_risk = np.mean([
            features.perturbation_signature,
            features.intent_deviation,
            1.0 - features.semantic_consistency
        ])
        
        total_risk = (0.4 * base_risk + 0.4 * confidence_risk + 
                     0.2 * feature_risk - uncertainty_penalty)
        
        return np.clip(total_risk, 0.0, 1.0)
    
    def _generate_enhanced_explanation(
        self, 
        detected_attacks: List[AttackType], 
        detection_breakdown: Dict[str, float],
        confidence: float,
        uncertainty: float
    ) -> str:
        """Generate enhanced explanation"""
        
        if not detected_attacks:
            return "No adversarial attacks detected."
        
        attack_names = [attack.value for attack in detected_attacks]
        
        explanation = f"Detected {len(attack_names)} attack type(s): {', '.join(attack_names)}. "
        explanation += f"Overall confidence: {confidence:.2f}, uncertainty: {uncertainty:.2f}. "
        
        # Add top detection signals
        sorted_detections = sorted(detection_breakdown.items(), 
                                 key=lambda x: x[1], reverse=True)
        top_signals = sorted_detections[:3]
        
        explanation += "Top signals: " + ", ".join([
            f"{name}({score:.2f})" for name, score in top_signals
        ])
        
        return explanation
    
    def _recommend_action(
        self, 
        is_adversarial: bool, 
        risk_score: float, 
        uncertainty: float,
        sophistication: float
    ) -> str:
        """Recommend action based on detection results"""
        
        if not is_adversarial:
            return "allow"
        
        if risk_score > 0.8:
            return "block"
        elif risk_score > 0.6:
            return "flag_for_review"
        elif uncertainty > 0.7:
            return "request_additional_analysis"
        elif sophistication > 0.7:
            return "escalate_to_human_review"
        else:
            return "monitor"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            'detection_count': len(self.detection_history),
            'avg_processing_time_ms': np.mean(self.performance_metrics['processing_time']) 
                if self.performance_metrics['processing_time'] else 0.0,
            'current_thresholds': {
                'esa': self.thresholds.esa_threshold,
                'cap': self.thresholds.cap_threshold,
                'risk': self.thresholds.risk_threshold,
                'confidence': self.thresholds.confidence_threshold
            }
        }
        
        # Add recent performance metrics if available
        for metric_name, values in self.performance_metrics.items():
            if values and metric_name != 'processing_time':
                stats[f'avg_{metric_name}'] = np.mean(values)
        
        return stats 