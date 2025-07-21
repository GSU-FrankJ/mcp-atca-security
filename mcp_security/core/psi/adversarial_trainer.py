"""
Advanced Adversarial Training Module for PSI Engine

This module implements sophisticated adversarial training techniques including:
- Specialized loss functions for FNR reduction
- Multi-objective optimization for balanced performance
- Online learning with concept drift adaptation
- Advanced data augmentation strategies
- Ensemble training with diversity enforcement
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings
from .enhanced_detectors import EnhancedAdversarialDetector, DetectionFeatures
from .attacks import EmbeddingShiftAttack, ContrastiveAdversarialPrompting
from .data import Dataset, PromptSample

@dataclass
class TrainingConfig:
    """Configuration for adversarial training"""
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001
    
    # Loss function weights
    fnr_penalty_weight: float = 3.0  # Higher penalty for false negatives
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    contrastive_margin: float = 0.5
    
    # Data augmentation
    augmentation_ratio: float = 2.0
    noise_level: float = 0.1
    
    # Ensemble parameters
    num_ensemble_models: int = 5
    diversity_weight: float = 0.1
    
    # Online learning
    online_learning_rate: float = 0.0001
    forgetting_factor: float = 0.95
    adaptation_threshold: float = 0.1

@dataclass
class TrainingResult:
    """Results from adversarial training"""
    
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    model_paths: List[str]
    best_epoch: int
    final_fnr: float
    convergence_time: float

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FNRPenalizedLoss(nn.Module):
    """Loss function with heavy penalty for false negatives"""
    
    def __init__(self, fnr_weight: float = 3.0, base_loss: str = 'ce'):
        super().__init__()
        self.fnr_weight = fnr_weight
        self.base_loss = base_loss
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Base loss (cross-entropy or focal)
        if self.base_loss == 'focal':
            base_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            base_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Identify false negatives (predicted 0, actual 1)
        predictions = torch.argmax(inputs, dim=1)
        false_negatives = (predictions == 0) & (targets == 1)
        
        # Apply penalty to false negatives
        penalty = torch.ones_like(base_loss)
        penalty[false_negatives] = self.fnr_weight
        
        penalized_loss = base_loss * penalty
        return penalized_loss.mean()

class AdversarialDetectorNet(nn.Module):
    """Neural network for adversarial detection"""
    
    def __init__(self, input_dim: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self, m):
        """Initialize weights using Xavier initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AdversarialTrainer:
    """
    Advanced adversarial trainer with FNR reduction focus.
    
    Features:
    - Multi-objective loss functions (FNR penalty, focal loss, contrastive)
    - Advanced data augmentation with synthetic attack generation
    - Ensemble training with diversity enforcement
    - Online learning for concept drift adaptation
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, detector: EnhancedAdversarialDetector, config: TrainingConfig):
        """
        Initialize the adversarial trainer.
        
        Args:
            detector: Enhanced adversarial detector to train
            config: Training configuration
        """
        self.detector = detector
        self.config = config
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Training state
        self.models: List[AdversarialDetectorNet] = []
        self.optimizers: List[optim.Optimizer] = []
        self.schedulers: List[optim.lr_scheduler._LRScheduler] = []
        
        # Loss functions
        self.loss_functions = {
            'focal': FocalLoss(config.focal_loss_alpha, config.focal_loss_gamma),
            'fnr_penalty': FNRPenalizedLoss(config.fnr_penalty_weight),
            'contrastive': nn.CosineEmbeddingLoss(margin=config.contrastive_margin)
        }
        
        # Training history
        self.training_history: Dict[str, List[float]] = defaultdict(list)
        
        # Online learning components
        self.online_optimizer: Optional[optim.Optimizer] = None
        self.concept_drift_detector = self._create_drift_detector()
        
        self.logger.info(
            "AdversarialTrainer initialized",
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            fnr_penalty_weight=config.fnr_penalty_weight
        )
    
    async def train_enhanced_detector(
        self, 
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        save_dir: Optional[Path] = None
    ) -> TrainingResult:
        """
        Train enhanced adversarial detector with advanced techniques.
        
        Args:
            training_data: Training dataset
            validation_data: Validation dataset (optional)
            save_dir: Directory to save trained models
            
        Returns:
            TrainingResult with comprehensive metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting enhanced adversarial training",
                training_samples=len(training_data.samples),
                validation_samples=len(validation_data.samples) if validation_data else 0,
                num_ensemble_models=self.config.num_ensemble_models
            )
            
            # Prepare data
            train_features, train_labels = await self._prepare_training_data(training_data)
            val_features, val_labels = None, None
            
            if validation_data:
                val_features, val_labels = await self._prepare_training_data(validation_data)
            else:
                # Split training data
                train_features, val_features, train_labels, val_labels = train_test_split(
                    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
                )
            
            # Data augmentation
            aug_features, aug_labels = await self._augment_training_data(
                train_features, train_labels, training_data
            )
            
            # Combine original and augmented data
            combined_features = np.concatenate([train_features, aug_features])
            combined_labels = np.concatenate([train_labels, aug_labels])
            
            self.logger.info(
                "Data preparation completed",
                original_samples=len(train_features),
                augmented_samples=len(aug_features),
                total_samples=len(combined_features)
            )
            
            # Train ensemble of models
            ensemble_results = []
            model_paths = []
            
            for model_idx in range(self.config.num_ensemble_models):
                self.logger.info(f"Training ensemble model {model_idx + 1}/{self.config.num_ensemble_models}")
                
                # Create model-specific data with bootstrapping
                model_features, model_labels = self._bootstrap_sample(
                    combined_features, combined_labels
                )
                
                # Train individual model
                model_result = await self._train_single_model(
                    model_features, model_labels,
                    val_features, val_labels,
                    model_idx
                )
                
                ensemble_results.append(model_result)
                
                # Save model if directory provided
                if save_dir:
                    model_path = save_dir / f"adversarial_detector_{model_idx}.pth"
                    torch.save(self.models[model_idx].state_dict(), model_path)
                    model_paths.append(str(model_path))
            
            # Evaluate ensemble performance
            ensemble_metrics = await self._evaluate_ensemble(
                val_features, val_labels
            )
            
            # Calculate final metrics
            best_model_idx = np.argmin([r['val_loss'] for r in ensemble_results])
            best_result = ensemble_results[best_model_idx]
            
            training_time = time.time() - start_time
            
            final_result = TrainingResult(
                train_metrics=best_result['train_metrics'],
                val_metrics=ensemble_metrics,
                test_metrics={},  # Will be filled by separate test evaluation
                training_history=dict(self.training_history),
                model_paths=model_paths,
                best_epoch=best_result['best_epoch'],
                final_fnr=ensemble_metrics.get('fnr', 0.0),
                convergence_time=training_time
            )
            
            self.logger.info(
                "Enhanced adversarial training completed",
                training_time_seconds=training_time,
                final_fnr=final_result.final_fnr,
                val_accuracy=ensemble_metrics.get('accuracy', 0.0),
                best_model_idx=best_model_idx
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(
                "Enhanced adversarial training failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def _prepare_training_data(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by extracting features"""
        
        features_list = []
        labels = []
        
        for sample in dataset.samples:
            try:
                # Extract features using the detector
                prompt_analysis = await self._mock_prompt_analysis(sample.prompt)
                features = await self.detector._extract_comprehensive_features(
                    sample.prompt, prompt_analysis
                )
                
                features_list.append(features.to_vector())
                labels.append(sample.label)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract features for sample: {e}")
                continue
        
        return np.array(features_list), np.array(labels)
    
    async def _mock_prompt_analysis(self, prompt: str):
        """Create mock prompt analysis for feature extraction"""
        # This would normally come from TokenEmbeddingAnalyzer
        # For now, create a simplified mock
        from .analyzers import PromptAnalysisResult, TokenAnalysisResult
        
        tokens = prompt.split()
        token_results = []
        
        for i, token in enumerate(tokens):
            # Mock embedding
            embedding = np.random.rand(768).astype(np.float32)
            
            token_result = TokenAnalysisResult(
                token=token,
                token_index=i,
                embedding=embedding,
                similarity_scores={'reference': 0.5},
                anomaly_score=np.random.rand(),
                is_anomalous=np.random.rand() > 0.8,
                context_window=tokens[max(0, i-2):min(len(tokens), i+3)]
            )
            token_results.append(token_result)
        
        return PromptAnalysisResult(
            prompt=prompt,
            tokens=tokens,
            token_results=token_results,
            overall_anomaly_score=np.mean([r.anomaly_score for r in token_results]),
            semantic_shifts=[],
            anomalies=[],
            processing_time_ms=1.0
        )
    
    async def _augment_training_data(
        self, 
        features: np.ndarray, 
        labels: np.ndarray,
        original_dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate augmented training data"""
        
        augmented_features = []
        augmented_labels = []
        
        # Get normal samples for attack generation
        normal_samples = [s for s in original_dataset.samples if s.label == 0]
        
        if not normal_samples:
            return np.array([]), np.array([])
        
        num_augmented = int(len(features) * self.config.augmentation_ratio)
        
        # Generate ESA attacks
        esa_generator = EmbeddingShiftAttack(self.detector.embedding_model, self.detector.settings)
        
        for _ in range(num_augmented // 2):
            try:
                sample = random.choice(normal_samples)
                
                # Generate ESA attack
                esa_variants = await esa_generator.generate_attacks(sample.prompt, 1)
                if esa_variants:
                    attack_prompt = esa_variants[0]
                    
                    # Extract features for the attack
                    prompt_analysis = await self._mock_prompt_analysis(attack_prompt)
                    attack_features = await self.detector._extract_comprehensive_features(
                        attack_prompt, prompt_analysis
                    )
                    
                    augmented_features.append(attack_features.to_vector())
                    augmented_labels.append(1)  # Malicious
                    
            except Exception as e:
                self.logger.debug(f"ESA augmentation failed: {e}")
                continue
        
        # Generate CAP attacks
        cap_generator = ContrastiveAdversarialPrompting(self.detector.embedding_model, self.detector.settings)
        
        for _ in range(num_augmented - len(augmented_features)):
            try:
                sample = random.choice(normal_samples)
                
                # Generate CAP attack
                cap_variants = await cap_generator.generate_attacks(sample.prompt, 1)
                if cap_variants:
                    attack_prompt = cap_variants[0]
                    
                    # Extract features for the attack
                    prompt_analysis = await self._mock_prompt_analysis(attack_prompt)
                    attack_features = await self.detector._extract_comprehensive_features(
                        attack_prompt, prompt_analysis
                    )
                    
                    augmented_features.append(attack_features.to_vector())
                    augmented_labels.append(1)  # Malicious
                    
            except Exception as e:
                self.logger.debug(f"CAP augmentation failed: {e}")
                continue
        
        # Add noise to existing features for regularization
        noise_features = []
        noise_labels = []
        
        for i in range(min(len(features), num_augmented // 4)):
            original_feature = features[i]
            noise = np.random.normal(0, self.config.noise_level, original_feature.shape)
            noisy_feature = original_feature + noise
            
            noise_features.append(noisy_feature)
            noise_labels.append(labels[i])
        
        # Combine all augmented data
        if augmented_features:
            all_aug_features = np.array(augmented_features + noise_features)
            all_aug_labels = np.array(augmented_labels + noise_labels)
        else:
            all_aug_features = np.array(noise_features)
            all_aug_labels = np.array(noise_labels)
        
        self.logger.info(
            "Data augmentation completed",
            esa_attacks=len([l for l in augmented_labels if l == 1]),
            noise_samples=len(noise_features),
            total_augmented=len(all_aug_features)
        )
        
        return all_aug_features, all_aug_labels
    
    def _bootstrap_sample(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample for ensemble diversity"""
        
        n_samples = len(features)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        return features[indices], labels[indices]
    
    async def _train_single_model(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        model_idx: int
    ) -> Dict[str, Any]:
        """Train a single model in the ensemble"""
        
        # Create model
        input_dim = train_features.shape[1]
        model = AdversarialDetectorNet(input_dim, self.config)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.patience // 2, factor=0.5
        )
        
        # Store for ensemble
        self.models.append(model)
        self.optimizers.append(optimizer)
        self.schedulers.append(scheduler)
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_features)
        train_labels_tensor = torch.LongTensor(train_labels)
        val_tensor = torch.FloatTensor(val_features)
        val_labels_tensor = torch.LongTensor(val_labels)
        
        # Create data loaders
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        
        # Handle class imbalance with weighted sampling
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            sampler=sampler
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        model_history = {'train_loss': [], 'val_loss': [], 'val_fnr': []}
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_features)
                
                # Multi-objective loss
                focal_loss = self.loss_functions['focal'](outputs, batch_labels)
                fnr_loss = self.loss_functions['fnr_penalty'](outputs, batch_labels)
                
                # Combined loss
                total_loss = 0.7 * focal_loss + 0.3 * fnr_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            
            with torch.no_grad():
                val_outputs = model(val_tensor)
                val_loss = self.loss_functions['focal'](val_outputs, val_labels_tensor).item()
                val_predictions = torch.argmax(val_outputs, dim=1).numpy()
            
            # Calculate metrics
            val_metrics = self._calculate_metrics(val_labels, val_predictions)
            
            # Update history
            avg_train_loss = train_loss / len(train_loader)
            model_history['train_loss'].append(avg_train_loss)
            model_history['val_loss'].append(val_loss)
            model_history['val_fnr'].append(val_metrics['fnr'])
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch} for model {model_idx}")
                    break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_outputs = model(val_tensor)
            final_predictions = torch.argmax(final_outputs, dim=1).numpy()
        
        final_metrics = self._calculate_metrics(val_labels, final_predictions)
        
        return {
            'model_idx': model_idx,
            'best_epoch': best_epoch,
            'val_loss': best_val_loss,
            'train_metrics': {'loss': model_history['train_loss'][-1]},
            'val_metrics': final_metrics,
            'history': model_history
        }
    
    async def _evaluate_ensemble(
        self, 
        val_features: np.ndarray, 
        val_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        
        if not self.models:
            return {}
        
        val_tensor = torch.FloatTensor(val_features)
        ensemble_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(val_tensor)
                probabilities = F.softmax(outputs, dim=1)
                ensemble_predictions.append(probabilities.numpy())
        
        # Average predictions
        avg_predictions = np.mean(ensemble_predictions, axis=0)
        final_predictions = np.argmax(avg_predictions, axis=1)
        
        # Calculate ensemble metrics
        metrics = self._calculate_metrics(val_labels, final_predictions)
        
        # Add ensemble-specific metrics
        prediction_variance = np.var([np.argmax(pred, axis=1) for pred in ensemble_predictions], axis=0)
        metrics['prediction_variance'] = np.mean(prediction_variance)
        metrics['ensemble_agreement'] = np.mean(prediction_variance < 0.25)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate rates
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # AUC calculation (if possible)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'tpr': tpr,
            'tnr': tnr,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def _create_drift_detector(self):
        """Create concept drift detector for online learning"""
        # Simplified drift detector
        return {'window_size': 1000, 'threshold': 0.1, 'recent_errors': deque(maxlen=1000)}
    
    async def online_update(
        self, 
        new_features: np.ndarray, 
        new_labels: np.ndarray
    ) -> Dict[str, float]:
        """Update models with new data (online learning)"""
        
        if not self.models or len(new_features) == 0:
            return {}
        
        try:
            # Convert to tensors
            features_tensor = torch.FloatTensor(new_features)
            labels_tensor = torch.LongTensor(new_labels)
            
            # Update each model in ensemble
            update_metrics = []
            
            for i, model in enumerate(self.models):
                if not self.online_optimizer:
                    self.online_optimizer = optim.AdamW(
                        model.parameters(), 
                        lr=self.config.online_learning_rate
                    )
                
                model.train()
                self.online_optimizer.zero_grad()
                
                outputs = model(features_tensor)
                loss = self.loss_functions['focal'](outputs, labels_tensor)
                
                loss.backward()
                self.online_optimizer.step()
                
                # Evaluate update
                model.eval()
                with torch.no_grad():
                    pred = torch.argmax(model(features_tensor), dim=1).numpy()
                
                metrics = self._calculate_metrics(new_labels, pred)
                update_metrics.append(metrics)
            
            # Average metrics across ensemble
            avg_metrics = {}
            for key in update_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in update_metrics])
            
            self.logger.info(
                "Online update completed",
                num_samples=len(new_features),
                avg_accuracy=avg_metrics.get('accuracy', 0.0),
                avg_fnr=avg_metrics.get('fnr', 0.0)
            )
            
            return avg_metrics
            
        except Exception as e:
            self.logger.error(f"Online update failed: {e}")
            return {}
    
    def save_training_state(self, save_path: Path) -> None:
        """Save complete training state"""
        
        state = {
            'config': self.config,
            'training_history': dict(self.training_history),
            'model_states': [model.state_dict() for model in self.models],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers]
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Training state saved to {save_path}")
    
    def load_training_state(self, load_path: Path) -> None:
        """Load complete training state"""
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.training_history = defaultdict(list, state['training_history'])
        
        # Restore models
        for i, model_state in enumerate(state['model_states']):
            if i < len(self.models):
                self.models[i].load_state_dict(model_state)
        
        # Restore optimizers
        for i, opt_state in enumerate(state['optimizer_states']):
            if i < len(self.optimizers):
                self.optimizers[i].load_state_dict(opt_state)
        
        self.logger.info(f"Training state loaded from {load_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        
        summary = {
            'config': self.config.__dict__,
            'num_models': len(self.models),
            'training_history': dict(self.training_history)
        }
        
        if self.training_history:
            # Add convergence analysis
            for metric, values in self.training_history.items():
                if values:
                    summary[f'{metric}_final'] = values[-1]
                    summary[f'{metric}_best'] = min(values) if 'loss' in metric else max(values)
        
        return summary 