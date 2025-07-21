"""
Token-Level Embedding Analysis Module for PSI Engine

This module implements token-level embedding analysis with sliding window
semantic shift detection and cosine similarity calculations for prompt anomaly detection.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings


@dataclass
class TokenAnalysisResult:
    """Results from token-level embedding analysis."""
    token: str
    token_index: int
    embedding: np.ndarray
    similarity_scores: Dict[str, float]  # Similarity to reference embeddings
    anomaly_score: float  # Aggregated anomaly score (0-1)
    is_anomalous: bool  # Whether this token is flagged as anomalous
    context_window: List[str]  # Surrounding tokens for context


@dataclass
class PromptAnalysisResult:
    """Complete analysis results for a prompt."""
    prompt: str
    tokens: List[str]
    token_results: List[TokenAnalysisResult]
    overall_anomaly_score: float
    semantic_shifts: List[Dict[str, Any]]  # Detected semantic shifts
    anomalies: List[Dict[str, Any]]  # Detected anomalies with metadata
    processing_time_ms: float
    
    
class TokenEmbeddingAnalyzer:
    """
    Advanced token-level embedding analyzer for semantic anomaly detection.
    
    Features:
    - Token-level embedding extraction using SentenceTransformers
    - Sliding window semantic shift detection
    - Cosine similarity calculations with reference patterns
    - Configurable anomaly thresholds
    - Performance-optimized batched processing
    """
    
    def __init__(
        self, 
        embedding_model: SentenceTransformer, 
        tokenizer: AutoTokenizer, 
        settings: Settings
    ):
        """
        Initialize the TokenEmbeddingAnalyzer.
        
        Args:
            embedding_model: Pre-loaded SentenceTransformer model
            tokenizer: Pre-loaded tokenizer for the model
            settings: Configuration settings containing PSI parameters
        """
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Configuration parameters
        self.window_size = getattr(settings, 'psi_window_size', 5)
        self.similarity_threshold = getattr(settings, 'psi_similarity_threshold', 0.7)
        self.anomaly_threshold = getattr(settings, 'psi_anomaly_threshold', 0.6)
        self.max_tokens = getattr(settings, 'psi_max_tokens', 512)
        
        # Reference embeddings cache (will be populated)
        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_tokens: List[str] = []
        self.faiss_index: Optional[faiss.Index] = None
        
        # Performance tracking
        self._analysis_times: List[float] = []
        
        self.logger.info(
            "TokenEmbeddingAnalyzer initialized",
            window_size=self.window_size,
            similarity_threshold=self.similarity_threshold,
            anomaly_threshold=self.anomaly_threshold,
            max_tokens=self.max_tokens
        )
    
    async def analyze_tokens(self, prompt: str) -> PromptAnalysisResult:
        """
        Perform comprehensive token-level analysis of a prompt.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            PromptAnalysisResult containing detailed analysis results
        """
        start_time = time.perf_counter()
        
        try:
            self.logger.debug(
                "Starting token-level analysis",
                prompt_length=len(prompt)
            )
            
            # Step 1: Tokenize the prompt
            tokens = await self._tokenize_prompt(prompt)
            
            # Step 2: Generate embeddings for each token in context
            token_embeddings = await self._generate_token_embeddings(prompt, tokens)
            
            # Step 3: Analyze each token for anomalies
            token_results = await self._analyze_individual_tokens(
                tokens, token_embeddings
            )
            
            # Step 4: Detect semantic shifts using sliding window
            semantic_shifts = await self._detect_semantic_shifts(
                tokens, token_embeddings
            )
            
            # Step 5: Aggregate anomaly information
            anomalies = self._aggregate_anomalies(token_results, semantic_shifts)
            overall_score = self._calculate_overall_anomaly_score(token_results)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self._analysis_times.append(processing_time)
            
            result = PromptAnalysisResult(
                prompt=prompt,
                tokens=tokens,
                token_results=token_results,
                overall_anomaly_score=overall_score,
                semantic_shifts=semantic_shifts,
                anomalies=anomalies,
                processing_time_ms=processing_time
            )
            
            self.logger.debug(
                "Token-level analysis completed",
                num_tokens=len(tokens),
                overall_anomaly_score=overall_score,
                num_anomalies=len(anomalies),
                processing_time_ms=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            
            self.logger.error(
                "Token-level analysis failed",
                error_type=type(e).__name__,
                error_message=str(e),
                prompt_length=len(prompt),
                processing_time_ms=processing_time
            )
            
            # Return safe default result
            return PromptAnalysisResult(
                prompt=prompt,
                tokens=[],
                token_results=[],
                overall_anomaly_score=1.0,  # High anomaly score on error
                semantic_shifts=[],
                anomalies=[{
                    'type': 'analysis_error',
                    'message': f"Analysis failed: {str(e)}",
                    'severity': 'critical'
                }],
                processing_time_ms=processing_time
            )
    
    async def _tokenize_prompt(self, prompt: str) -> List[str]:
        """
        Tokenize prompt using the configured tokenizer.
        
        Args:
            prompt: Input prompt to tokenize
            
        Returns:
            List of tokens
        """
        try:
            # Use the tokenizer to get tokens
            encoded = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                max_length=self.max_tokens,
                truncation=True
            )
            
            # Decode back to get string tokens
            tokens = [
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in encoded
            ]
            
            # Filter out empty tokens and special tokens for analysis
            tokens = [token.strip() for token in tokens if token.strip()]
            
            return tokens
            
        except Exception as e:
            self.logger.error(
                "Tokenization failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            # Fallback to simple whitespace tokenization
            return prompt.split()
    
    async def _generate_token_embeddings(
        self, 
        prompt: str, 
        tokens: List[str]
    ) -> np.ndarray:
        """
        Generate contextual embeddings for tokens.
        
        Args:
            prompt: Original prompt for context
            tokens: List of tokens to embed
            
        Returns:
            Array of embeddings with shape (num_tokens, embedding_dim)
        """
        try:
            # Generate sentence embedding for the full prompt
            # This provides contextual information for each token
            full_embedding = self.embedding_model.encode(
                prompt,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Generate embeddings for individual tokens with context
            token_embeddings = []
            
            for i, token in enumerate(tokens):
                # Create context window around the token
                context_start = max(0, i - self.window_size)
                context_end = min(len(tokens), i + self.window_size + 1)
                context_tokens = tokens[context_start:context_end]
                context_text = " ".join(context_tokens)
                
                # Generate embedding for token in context
                token_embedding = self.embedding_model.encode(
                    context_text,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Convert to numpy for consistency
                if isinstance(token_embedding, torch.Tensor):
                    token_embedding = token_embedding.cpu().numpy()
                
                token_embeddings.append(token_embedding)
            
            # Stack embeddings into array
            embeddings_array = np.stack(token_embeddings)
            
            self.logger.debug(
                "Generated token embeddings",
                num_tokens=len(tokens),
                embedding_shape=embeddings_array.shape
            )
            
            return embeddings_array
            
        except Exception as e:
            self.logger.error(
                "Token embedding generation failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            # Return zero embeddings as fallback
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            return np.zeros((len(tokens), embedding_dim))
    
    async def _analyze_individual_tokens(
        self,
        tokens: List[str],
        embeddings: np.ndarray
    ) -> List[TokenAnalysisResult]:
        """
        Analyze each token for anomalies using similarity metrics.
        
        Args:
            tokens: List of tokens
            embeddings: Token embeddings array
            
        Returns:
            List of TokenAnalysisResult objects
        """
        results = []
        
        for i, (token, embedding) in enumerate(zip(tokens, embeddings)):
            try:
                # Calculate similarity scores with reference embeddings
                similarity_scores = await self._calculate_similarity_scores(embedding)
                
                # Calculate anomaly score based on similarities
                anomaly_score = self._calculate_token_anomaly_score(similarity_scores)
                
                # Determine if token is anomalous
                is_anomalous = anomaly_score > self.anomaly_threshold
                
                # Get context window
                context_start = max(0, i - 2)
                context_end = min(len(tokens), i + 3)
                context_window = tokens[context_start:context_end]
                
                result = TokenAnalysisResult(
                    token=token,
                    token_index=i,
                    embedding=embedding,
                    similarity_scores=similarity_scores,
                    anomaly_score=anomaly_score,
                    is_anomalous=is_anomalous,
                    context_window=context_window
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to analyze individual token",
                    token=token,
                    token_index=i,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                # Create safe default result
                results.append(TokenAnalysisResult(
                    token=token,
                    token_index=i,
                    embedding=embedding,
                    similarity_scores={},
                    anomaly_score=1.0,  # High anomaly score on error
                    is_anomalous=True,
                    context_window=[]
                ))
        
        return results
    
    async def _calculate_similarity_scores(
        self, 
        embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate similarity scores between token embedding and reference patterns.
        
        Args:
            embedding: Token embedding vector
            
        Returns:
            Dictionary of similarity scores
        """
        similarity_scores = {}
        
        try:
            # If we have reference embeddings, calculate similarities
            if self.reference_embeddings is not None:
                # Calculate cosine similarities with all reference embeddings
                similarities = cosine_similarity(
                    embedding.reshape(1, -1),
                    self.reference_embeddings
                )[0]
                
                # Get top-k most similar references
                top_k = min(10, len(similarities))
                top_indices = np.argsort(similarities)[-top_k:]
                
                similarity_scores['max_similarity'] = float(np.max(similarities))
                similarity_scores['mean_similarity'] = float(np.mean(similarities))
                similarity_scores['min_similarity'] = float(np.min(similarities))
                similarity_scores['std_similarity'] = float(np.std(similarities))
                
                # Add top similar tokens if available
                if self.reference_tokens:
                    top_similar_tokens = [
                        self.reference_tokens[i] for i in top_indices 
                        if i < len(self.reference_tokens)
                    ]
                    similarity_scores['top_similar_tokens'] = top_similar_tokens[:5]
            
            else:
                # No reference embeddings available - use default scores
                similarity_scores = {
                    'max_similarity': 0.5,
                    'mean_similarity': 0.5,
                    'min_similarity': 0.5,
                    'std_similarity': 0.0
                }
                
        except Exception as e:
            self.logger.warning(
                "Similarity calculation failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            # Return default similarity scores on error
            similarity_scores = {
                'max_similarity': 0.0,
                'mean_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0
            }
        
        return similarity_scores
    
    def _calculate_token_anomaly_score(
        self, 
        similarity_scores: Dict[str, float]
    ) -> float:
        """
        Calculate anomaly score for a token based on similarity metrics.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            
        Returns:
            Anomaly score between 0 and 1 (higher = more anomalous)
        """
        try:
            max_sim = similarity_scores.get('max_similarity', 0.5)
            mean_sim = similarity_scores.get('mean_similarity', 0.5)
            std_sim = similarity_scores.get('std_similarity', 0.0)
            
            # Calculate anomaly score (inverse of similarity)
            # Low similarity = high anomaly
            # High standard deviation = unusual pattern = higher anomaly
            
            base_anomaly = 1.0 - max_sim  # Invert max similarity
            variance_penalty = std_sim * 0.5  # Add penalty for high variance
            
            anomaly_score = min(1.0, base_anomaly + variance_penalty)
            
            return anomaly_score
            
        except Exception:
            # Return high anomaly score on calculation error
            return 1.0
    
    async def _detect_semantic_shifts(
        self,
        tokens: List[str],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect semantic shifts using sliding window analysis.
        
        Args:
            tokens: List of tokens
            embeddings: Token embeddings array
            
        Returns:
            List of detected semantic shifts with metadata
        """
        semantic_shifts = []
        
        try:
            if len(embeddings) < 2:
                return semantic_shifts
            
            # Calculate pairwise similarities between adjacent tokens
            for i in range(len(embeddings) - 1):
                current_embedding = embeddings[i]
                next_embedding = embeddings[i + 1]
                
                # Calculate cosine similarity between adjacent tokens
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1),
                    next_embedding.reshape(1, -1)
                )[0][0]
                
                # Detect significant semantic shift (low similarity)
                if similarity < self.similarity_threshold:
                    semantic_shift = {
                        'position': i,
                        'tokens': [tokens[i], tokens[i + 1]],
                        'similarity': float(similarity),
                        'shift_magnitude': 1.0 - similarity,
                        'type': 'adjacent_token_shift',
                        'severity': self._classify_shift_severity(1.0 - similarity)
                    }
                    semantic_shifts.append(semantic_shift)
            
            # Window-based semantic shift detection
            window_shifts = await self._detect_window_shifts(tokens, embeddings)
            semantic_shifts.extend(window_shifts)
            
            self.logger.debug(
                "Semantic shift detection completed",
                num_shifts=len(semantic_shifts)
            )
            
        except Exception as e:
            self.logger.error(
                "Semantic shift detection failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
        
        return semantic_shifts
    
    async def _detect_window_shifts(
        self,
        tokens: List[str],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect semantic shifts using sliding window analysis.
        
        Args:
            tokens: List of tokens
            embeddings: Token embeddings array
            
        Returns:
            List of detected window-based semantic shifts
        """
        window_shifts = []
        
        try:
            if len(embeddings) < self.window_size * 2:
                return window_shifts
            
            # Sliding window analysis
            for i in range(len(embeddings) - self.window_size):
                # Get current window and next window
                current_window = embeddings[i:i + self.window_size]
                next_window = embeddings[i + 1:i + 1 + self.window_size]
                
                # Calculate average embeddings for each window
                current_avg = np.mean(current_window, axis=0)
                next_avg = np.mean(next_window, axis=0)
                
                # Calculate similarity between window averages
                window_similarity = cosine_similarity(
                    current_avg.reshape(1, -1),
                    next_avg.reshape(1, -1)
                )[0][0]
                
                # Detect window-based semantic shift
                if window_similarity < self.similarity_threshold:
                    window_shift = {
                        'position': i,
                        'window_size': self.window_size,
                        'tokens': tokens[i:i + self.window_size * 2],
                        'similarity': float(window_similarity),
                        'shift_magnitude': 1.0 - window_similarity,
                        'type': 'window_based_shift',
                        'severity': self._classify_shift_severity(1.0 - window_similarity)
                    }
                    window_shifts.append(window_shift)
            
        except Exception as e:
            self.logger.warning(
                "Window-based shift detection failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
        
        return window_shifts
    
    def _classify_shift_severity(self, shift_magnitude: float) -> str:
        """
        Classify the severity of a semantic shift.
        
        Args:
            shift_magnitude: Magnitude of the shift (0-1)
            
        Returns:
            Severity classification string
        """
        if shift_magnitude > 0.8:
            return 'critical'
        elif shift_magnitude > 0.6:
            return 'high'
        elif shift_magnitude > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _aggregate_anomalies(
        self,
        token_results: List[TokenAnalysisResult],
        semantic_shifts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate all detected anomalies into a unified list.
        
        Args:
            token_results: Individual token analysis results
            semantic_shifts: Detected semantic shifts
            
        Returns:
            List of anomalies with metadata
        """
        anomalies = []
        
        # Add token-level anomalies
        for result in token_results:
            if result.is_anomalous:
                anomaly = {
                    'type': 'token_anomaly',
                    'token': result.token,
                    'position': result.token_index,
                    'anomaly_score': result.anomaly_score,
                    'context': result.context_window,
                    'similarity_scores': result.similarity_scores,
                    'severity': self._classify_anomaly_severity(result.anomaly_score)
                }
                anomalies.append(anomaly)
        
        # Add semantic shift anomalies
        for shift in semantic_shifts:
            anomaly = {
                'type': 'semantic_shift',
                'position': shift['position'],
                'tokens': shift['tokens'],
                'shift_magnitude': shift['shift_magnitude'],
                'shift_type': shift['type'],
                'severity': shift['severity']
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_anomaly_severity(self, anomaly_score: float) -> str:
        """
        Classify anomaly severity based on score.
        
        Args:
            anomaly_score: Anomaly score (0-1)
            
        Returns:
            Severity classification string
        """
        if anomaly_score > 0.9:
            return 'critical'
        elif anomaly_score > 0.7:
            return 'high'
        elif anomaly_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_overall_anomaly_score(
        self, 
        token_results: List[TokenAnalysisResult]
    ) -> float:
        """
        Calculate overall anomaly score for the entire prompt.
        
        Args:
            token_results: Individual token analysis results
            
        Returns:
            Overall anomaly score (0-1)
        """
        if not token_results:
            return 0.0
        
        # Calculate weighted average of token anomaly scores
        total_score = sum(result.anomaly_score for result in token_results)
        avg_score = total_score / len(token_results)
        
        # Apply penalty for number of anomalous tokens
        anomalous_count = sum(1 for result in token_results if result.is_anomalous)
        anomalous_ratio = anomalous_count / len(token_results)
        
        # Combine average score with anomalous token ratio
        overall_score = (avg_score * 0.7) + (anomalous_ratio * 0.3)
        
        return min(1.0, overall_score)
    
    async def load_reference_embeddings(
        self, 
        reference_prompts: List[str]
    ) -> None:
        """
        Load and build reference embeddings from normal prompts.
        
        Args:
            reference_prompts: List of normal/safe prompts for reference
        """
        try:
            self.logger.info(
                "Building reference embeddings database",
                num_prompts=len(reference_prompts)
            )
            
            # Generate embeddings for reference prompts
            reference_embeddings = []
            reference_tokens = []
            
            for prompt in reference_prompts:
                tokens = await self._tokenize_prompt(prompt)
                embeddings = await self._generate_token_embeddings(prompt, tokens)
                
                reference_embeddings.extend(embeddings)
                reference_tokens.extend(tokens)
            
            # Convert to numpy array
            self.reference_embeddings = np.array(reference_embeddings)
            self.reference_tokens = reference_tokens
            
            # Build FAISS index for efficient similarity search
            if len(self.reference_embeddings) > 0:
                embedding_dim = self.reference_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                
                # Normalize embeddings for cosine similarity
                normalized_embeddings = self.reference_embeddings / np.linalg.norm(
                    self.reference_embeddings, axis=1, keepdims=True
                )
                
                self.faiss_index.add(normalized_embeddings.astype('float32'))
            
            self.logger.info(
                "Reference embeddings database built successfully",
                num_embeddings=len(self.reference_embeddings),
                embedding_dim=self.reference_embeddings.shape[1] if len(self.reference_embeddings) > 0 else 0
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load reference embeddings",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the analyzer.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self._analysis_times:
            return {}
        
        return {
            'avg_analysis_time_ms': np.mean(self._analysis_times),
            'p95_analysis_time_ms': np.percentile(self._analysis_times, 95),
            'p99_analysis_time_ms': np.percentile(self._analysis_times, 99),
            'min_analysis_time_ms': np.min(self._analysis_times),
            'max_analysis_time_ms': np.max(self._analysis_times),
            'total_analyses': len(self._analysis_times)
        } 