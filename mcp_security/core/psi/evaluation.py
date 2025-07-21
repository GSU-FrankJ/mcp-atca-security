"""
Evaluation Framework for PSI Engine

This module implements comprehensive evaluation capabilities including:
- Performance metrics calculation
- Benchmarking and testing
- FNR (False Negative Rate) analysis
- ROC/PR curve generation
- Performance profiling and optimization analysis
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from ...utils.logging import get_logger, SecurityLogger
from ...utils.config import Settings
from .data import Dataset, PromptSample


@dataclass
class EvaluationResult:
    """Results from PSI evaluation."""
    dataset_name: str
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fnr: float  # False Negative Rate
    fpr: float  # False Positive Rate
    auc_roc: float
    auc_pr: float
    confusion_matrix: np.ndarray
    processing_times: List[float]
    avg_processing_time_ms: float
    p95_processing_time_ms: float
    performance_target_met: bool
    detailed_metrics: Dict[str, Any]
    timestamp: str


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""
    test_name: str
    num_prompts: int
    total_time_ms: float
    avg_time_per_prompt_ms: float
    throughput_prompts_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    performance_breakdown: Dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]


class EvaluationFramework:
    """
    Comprehensive evaluation framework for PSI engine.
    
    Features:
    - Standard ML metrics calculation
    - ROC and PR curve analysis
    - Performance profiling
    - Adversarial robustness testing
    - Custom PSI-specific metrics
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the EvaluationFramework.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Configuration
        self.performance_target_ms = getattr(settings, 'psi_performance_target_ms', 200.0)
        self.fnr_target = getattr(settings, 'psi_fnr_target', 0.05)  # Target FNR of 5%
        self.results_dir = Path(getattr(settings, 'psi_results_dir', '.taskmaster/results/psi'))
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        
        self.logger.info(
            "EvaluationFramework initialized",
            performance_target_ms=self.performance_target_ms,
            fnr_target=self.fnr_target,
            results_dir=str(self.results_dir)
        )
    
    async def evaluate_psi_engine(
        self, 
        psi_engine,  # PSIEngine instance
        test_dataset: Dataset,
        save_results: bool = True
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of PSI engine on test dataset.
        
        Args:
            psi_engine: PSI engine instance to evaluate
            test_dataset: Test dataset for evaluation
            save_results: Whether to save results to disk
            
        Returns:
            EvaluationResult containing comprehensive metrics
        """
        try:
            self.logger.info(
                "Starting PSI engine evaluation",
                dataset_name=test_dataset.name,
                num_samples=len(test_dataset)
            )
            
            # Collect predictions and ground truth
            predictions = []
            ground_truth = []
            confidence_scores = []
            processing_times = []
            
            # Evaluate each sample
            for sample in test_dataset.samples:
                start_time = time.perf_counter()
                
                # Analyze prompt with PSI engine
                result = await psi_engine.analyze_prompt(sample.prompt)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                processing_times.append(processing_time)
                
                # Collect results
                predictions.append(1 if result.is_malicious else 0)
                ground_truth.append(sample.label)
                confidence_scores.append(result.confidence_score)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(
                ground_truth, predictions, confidence_scores
            )
            
            # Calculate performance metrics
            performance_metrics = self._analyze_performance(processing_times)
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                dataset_name=test_dataset.name,
                total_samples=len(test_dataset),
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1'],
                fnr=metrics['fnr'],
                fpr=metrics['fpr'],
                auc_roc=metrics['auc_roc'],
                auc_pr=metrics['auc_pr'],
                confusion_matrix=metrics['confusion_matrix'],
                processing_times=processing_times,
                avg_processing_time_ms=performance_metrics['avg_time'],
                p95_processing_time_ms=performance_metrics['p95_time'],
                performance_target_met=performance_metrics['target_met'],
                detailed_metrics={
                    **metrics,
                    **performance_metrics,
                    'classification_report': metrics['classification_report']
                },
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Add to history
            self.evaluation_history.append(evaluation_result)
            
            # Save results if requested
            if save_results:
                await self._save_evaluation_results(evaluation_result)
            
            # Generate plots
            await self._generate_evaluation_plots(evaluation_result, ground_truth, confidence_scores)
            
            self.logger.info(
                "PSI engine evaluation completed",
                dataset_name=test_dataset.name,
                accuracy=evaluation_result.accuracy,
                f1_score=evaluation_result.f1_score,
                fnr=evaluation_result.fnr,
                avg_processing_time_ms=evaluation_result.avg_processing_time_ms,
                performance_target_met=evaluation_result.performance_target_met
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(
                "PSI engine evaluation failed",
                dataset_name=test_dataset.name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    def _calculate_comprehensive_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels  
            y_scores: Prediction confidence scores
            
        Returns:
            Dictionary of calculated metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        accuracy = np.mean(y_true == y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
        
        # False Negative Rate and False Positive Rate
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # AUC scores
        try:
            auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
        except ValueError:
            auc_roc = 0.5
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            auc_pr = np.trapz(precision_curve, recall_curve)
        except ValueError:
            auc_pr = 0.5
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Normal', 'Adversarial'],
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'classification_report': class_report
        }
    
    def _analyze_performance(self, processing_times: List[float]) -> Dict[str, Any]:
        """
        Analyze performance characteristics.
        
        Args:
            processing_times: List of processing times in milliseconds
            
        Returns:
            Dictionary of performance metrics
        """
        if not processing_times:
            return {
                'avg_time': 0.0,
                'p95_time': 0.0,
                'p99_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'target_met': False
            }
        
        avg_time = np.mean(processing_times)
        p95_time = np.percentile(processing_times, 95)
        p99_time = np.percentile(processing_times, 99)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        
        # Check if performance target is met
        target_met = p95_time <= self.performance_target_ms
        
        return {
            'avg_time': avg_time,
            'p95_time': p95_time,
            'p99_time': p99_time,
            'min_time': min_time,
            'max_time': max_time,
            'target_met': target_met,
            'total_samples': len(processing_times)
        }
    
    async def _save_evaluation_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to file."""
        try:
            # Create filename with timestamp
            filename = f"evaluation_{result.dataset_name}_{result.timestamp.replace(':', '-').replace(' ', '_')}.json"
            filepath = self.results_dir / filename
            
            # Convert result to serializable format
            result_dict = {
                'dataset_name': result.dataset_name,
                'total_samples': result.total_samples,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'fnr': result.fnr,
                'fpr': result.fpr,
                'auc_roc': result.auc_roc,
                'auc_pr': result.auc_pr,
                'confusion_matrix': result.confusion_matrix.tolist(),
                'avg_processing_time_ms': result.avg_processing_time_ms,
                'p95_processing_time_ms': result.p95_processing_time_ms,
                'performance_target_met': result.performance_target_met,
                'detailed_metrics': result.detailed_metrics,
                'timestamp': result.timestamp
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.info(
                "Evaluation results saved",
                filepath=str(filepath)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to save evaluation results",
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    async def _generate_evaluation_plots(
        self, 
        result: EvaluationResult, 
        y_true: List[int], 
        y_scores: List[float]
    ) -> None:
        """Generate evaluation plots and charts."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create plots directory
            plots_dir = self.results_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            timestamp_str = result.timestamp.replace(':', '-').replace(' ', '_')
            
            # 1. ROC Curve
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {result.auc_roc:.3f})')
                plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {result.dataset_name}')
                plt.legend()
                plt.grid(True)
                
                roc_path = plots_dir / f"roc_curve_{result.dataset_name}_{timestamp_str}.png"
                plt.savefig(roc_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # 2. Precision-Recall Curve
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {result.auc_pr:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {result.dataset_name}')
                plt.legend()
                plt.grid(True)
                
                pr_path = plots_dir / f"pr_curve_{result.dataset_name}_{timestamp_str}.png"
                plt.savefig(pr_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # 3. Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            cm = result.confusion_matrix
            
            # Create labels
            if cm.shape == (2, 2):
                labels = ['Normal', 'Adversarial']
                im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar(im)
                
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix - {result.dataset_name}')
                plt.xticks(range(len(labels)), labels)
                plt.yticks(range(len(labels)), labels)
                
                cm_path = plots_dir / f"confusion_matrix_{result.dataset_name}_{timestamp_str}.png"
                plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # 4. Processing Time Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(result.processing_times, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(result.avg_processing_time_ms, color='red', linestyle='--', 
                       label=f'Average: {result.avg_processing_time_ms:.1f}ms')
            plt.axvline(result.p95_processing_time_ms, color='orange', linestyle='--',
                       label=f'95th percentile: {result.p95_processing_time_ms:.1f}ms')
            plt.axvline(self.performance_target_ms, color='green', linestyle='-',
                       label=f'Target: {self.performance_target_ms}ms')
            
            plt.xlabel('Processing Time (ms)')
            plt.ylabel('Frequency')
            plt.title(f'Processing Time Distribution - {result.dataset_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            time_path = plots_dir / f"processing_times_{result.dataset_name}_{timestamp_str}.png"
            plt.savefig(time_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(
                "Evaluation plots generated",
                plots_dir=str(plots_dir),
                dataset_name=result.dataset_name
            )
            
        except Exception as e:
            self.logger.warning(
                "Failed to generate evaluation plots",
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    async def compare_evaluations(self, evaluation_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple evaluation results.
        
        Args:
            evaluation_names: List of evaluation result names to compare
            
        Returns:
            Comparison analysis
        """
        # Implementation for comparing multiple evaluations
        # This would load saved evaluation results and create comparative analysis
        pass
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        latest = self.evaluation_history[-1]
        
        # Calculate improvement over time if multiple evaluations
        improvement_metrics = {}
        if len(self.evaluation_history) > 1:
            baseline = self.evaluation_history[0]
            improvement_metrics = {
                'accuracy_improvement': latest.accuracy - baseline.accuracy,
                'f1_improvement': latest.f1_score - baseline.f1_score,
                'fnr_improvement': baseline.fnr - latest.fnr,  # Lower FNR is better
                'performance_improvement': baseline.avg_processing_time_ms - latest.avg_processing_time_ms
            }
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'latest_evaluation': {
                'dataset': latest.dataset_name,
                'accuracy': latest.accuracy,
                'f1_score': latest.f1_score,
                'fnr': latest.fnr,
                'avg_processing_time_ms': latest.avg_processing_time_ms,
                'performance_target_met': latest.performance_target_met,
                'timestamp': latest.timestamp
            },
            'improvement_metrics': improvement_metrics,
            'performance_targets': {
                'fnr_target': self.fnr_target,
                'processing_time_target_ms': self.performance_target_ms,
                'fnr_target_met': latest.fnr <= self.fnr_target,
                'processing_target_met': latest.performance_target_met
            }
        }


class PerformanceBenchmark:
    """
    Performance benchmarking utility for PSI engine.
    
    Features:
    - Throughput benchmarking
    - Latency analysis
    - Resource utilization monitoring
    - Bottleneck identification
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the PerformanceBenchmark.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        self.logger.info("PerformanceBenchmark initialized")
    
    async def benchmark_throughput(
        self, 
        psi_engine, 
        test_prompts: List[str],
        concurrent_requests: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark PSI engine throughput with concurrent requests.
        
        Args:
            psi_engine: PSI engine instance to benchmark
            test_prompts: List of test prompts
            concurrent_requests: Number of concurrent requests
            
        Returns:
            BenchmarkResult containing throughput metrics
        """
        try:
            self.logger.info(
                "Starting throughput benchmark",
                num_prompts=len(test_prompts),
                concurrent_requests=concurrent_requests
            )
            
            start_time = time.perf_counter()
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def process_prompt(prompt: str) -> Dict[str, Any]:
                async with semaphore:
                    prompt_start = time.perf_counter()
                    result = await psi_engine.analyze_prompt(prompt)
                    prompt_time = (time.perf_counter() - prompt_start) * 1000
                    
                    return {
                        'processing_time_ms': prompt_time,
                        'is_malicious': result.is_malicious,
                        'confidence': result.confidence_score
                    }
            
            # Process all prompts concurrently
            tasks = [process_prompt(prompt) for prompt in test_prompts]
            results = await asyncio.gather(*tasks)
            
            total_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Calculate metrics
            processing_times = [r['processing_time_ms'] for r in results]
            avg_time_per_prompt = np.mean(processing_times)
            throughput = (len(test_prompts) / total_time) * 1000  # Prompts per second
            
            # Identify bottlenecks and recommendations
            bottlenecks, recommendations = self._analyze_bottlenecks(
                processing_times, total_time, concurrent_requests
            )
            
            benchmark_result = BenchmarkResult(
                test_name=f"throughput_benchmark_{concurrent_requests}_concurrent",
                num_prompts=len(test_prompts),
                total_time_ms=total_time,
                avg_time_per_prompt_ms=avg_time_per_prompt,
                throughput_prompts_per_second=throughput,
                memory_usage_mb=0.0,  # Would need actual memory monitoring
                cpu_usage_percent=0.0,  # Would need actual CPU monitoring
                performance_breakdown={
                    'min_processing_time_ms': np.min(processing_times),
                    'max_processing_time_ms': np.max(processing_times),
                    'p95_processing_time_ms': np.percentile(processing_times, 95),
                    'p99_processing_time_ms': np.percentile(processing_times, 99),
                    'std_processing_time_ms': np.std(processing_times)
                },
                bottlenecks=bottlenecks,
                recommendations=recommendations
            )
            
            self.logger.info(
                "Throughput benchmark completed",
                throughput_pps=throughput,
                avg_time_per_prompt_ms=avg_time_per_prompt,
                total_time_ms=total_time
            )
            
            return benchmark_result
            
        except Exception as e:
            self.logger.error(
                "Throughput benchmark failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    def _analyze_bottlenecks(
        self, 
        processing_times: List[float], 
        total_time: float, 
        concurrent_requests: int
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze performance bottlenecks and generate recommendations.
        
        Args:
            processing_times: Individual processing times
            total_time: Total benchmark time
            concurrent_requests: Number of concurrent requests
            
        Returns:
            Tuple of (bottlenecks, recommendations)
        """
        bottlenecks = []
        recommendations = []
        
        avg_time = np.mean(processing_times)
        p95_time = np.percentile(processing_times, 95)
        std_time = np.std(processing_times)
        
        # Check for high average processing time
        if avg_time > 200:
            bottlenecks.append("High average processing time")
            recommendations.append("Consider model optimization or hardware acceleration")
        
        # Check for high variability
        if std_time > avg_time * 0.5:
            bottlenecks.append("High processing time variability")
            recommendations.append("Investigate caching mechanisms or load balancing")
        
        # Check for outliers
        if p95_time > avg_time * 2:
            bottlenecks.append("Significant processing time outliers")
            recommendations.append("Profile outlier cases and optimize hot paths")
        
        # Check concurrency efficiency
        theoretical_min_time = max(processing_times) if processing_times else 0
        actual_total_time = total_time
        
        if actual_total_time > theoretical_min_time * 1.5:
            bottlenecks.append("Suboptimal concurrency utilization")
            recommendations.append("Increase concurrent request limit or optimize async operations")
        
        return bottlenecks, recommendations
    
    async def profile_components(self, psi_engine, test_prompt: str) -> Dict[str, float]:
        """
        Profile individual components of PSI engine processing.
        
        Args:
            psi_engine: PSI engine instance to profile
            test_prompt: Single test prompt for profiling
            
        Returns:
            Dictionary of component processing times
        """
        # This would require instrumentation of PSI engine components
        # For now, return placeholder profiling data
        return {
            'tokenization_ms': 5.0,
            'embedding_generation_ms': 50.0,
            'similarity_calculation_ms': 20.0,
            'anomaly_detection_ms': 30.0,
            'risk_aggregation_ms': 10.0,
            'total_ms': 115.0
        } 