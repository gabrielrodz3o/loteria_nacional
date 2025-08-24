"""Probability calibration utilities for lottery prediction models."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrates prediction probabilities using various methods."""
    
    def __init__(self):
        self.calibrators = {}
        self.is_fitted = {}
        
    def fit_platt_scaling(self, model_name: str, probabilities: np.ndarray, 
                         true_labels: np.ndarray) -> None:
        """Fit Platt scaling calibrator for a model."""
        try:
            logger.info(f"Fitting Platt scaling for {model_name}")
            
            # Platt scaling uses logistic regression
            calibrator = LogisticRegression(max_iter=2000)
            
            # Reshape probabilities if needed
            if len(probabilities.shape) == 1:
                prob_features = probabilities.reshape(-1, 1)
            else:
                # Use max probability as feature
                prob_features = np.max(probabilities, axis=1).reshape(-1, 1)
            
            calibrator.fit(prob_features, true_labels)
            
            self.calibrators[f"{model_name}_platt"] = calibrator
            self.is_fitted[f"{model_name}_platt"] = True
            
            logger.info(f"Platt scaling fitted for {model_name}")
            
        except Exception as e:
            logger.error(f"Platt scaling fitting failed for {model_name}: {e}")
    
    def fit_isotonic_regression(self, model_name: str, probabilities: np.ndarray,
                               true_labels: np.ndarray) -> None:
        """Fit isotonic regression calibrator for a model."""
        try:
            logger.info(f"Fitting isotonic regression for {model_name}")
            
            # Use max probability for isotonic regression
            if len(probabilities.shape) > 1:
                prob_values = np.max(probabilities, axis=1)
            else:
                prob_values = probabilities
            
            # Create binary labels (correct/incorrect predictions)
            predicted_labels = np.argmax(probabilities, axis=1) if len(probabilities.shape) > 1 else (probabilities > 0.5).astype(int)
            binary_labels = (predicted_labels == true_labels).astype(int)
            
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(prob_values, binary_labels)
            
            self.calibrators[f"{model_name}_isotonic"] = calibrator
            self.is_fitted[f"{model_name}_isotonic"] = True
            
            logger.info(f"Isotonic regression fitted for {model_name}")
            
        except Exception as e:
            logger.error(f"Isotonic regression fitting failed for {model_name}: {e}")
    
    def calibrate_probabilities(self, model_name: str, probabilities: np.ndarray,
                               method: str = 'platt') -> np.ndarray:
        """Calibrate probabilities using specified method."""
        try:
            calibrator_key = f"{model_name}_{method}"
            
            if calibrator_key not in self.calibrators or not self.is_fitted[calibrator_key]:
                logger.warning(f"No fitted calibrator found for {calibrator_key}")
                return probabilities
            
            calibrator = self.calibrators[calibrator_key]
            
            if method == 'platt':
                # Platt scaling
                if len(probabilities.shape) == 1:
                    prob_features = probabilities.reshape(-1, 1)
                else:
                    prob_features = np.max(probabilities, axis=1).reshape(-1, 1)
                
                calibrated_probs = calibrator.predict_proba(prob_features)
                
                # Return calibrated probabilities in original format
                if len(probabilities.shape) == 1:
                    return calibrated_probs[:, 1]  # Probability of positive class
                else:
                    # Scale original probabilities
                    scaling_factor = calibrated_probs[:, 1] / np.max(probabilities, axis=1)
                    return probabilities * scaling_factor.reshape(-1, 1)
            
            elif method == 'isotonic':
                # Isotonic regression
                if len(probabilities.shape) > 1:
                    prob_values = np.max(probabilities, axis=1)
                    calibrated_max = calibrator.predict(prob_values)
                    
                    # Scale original probabilities
                    scaling_factor = calibrated_max / prob_values
                    return probabilities * scaling_factor.reshape(-1, 1)
                else:
                    return calibrator.predict(probabilities)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability calibration failed for {model_name}: {e}")
            return probabilities
    
    def evaluate_calibration(self, probabilities: np.ndarray, true_labels: np.ndarray,
                           n_bins: int = 10) -> Dict[str, float]:
        """Evaluate calibration quality using reliability diagram."""
        try:
            # Get predicted probabilities
            if len(probabilities.shape) > 1:
                pred_probs = np.max(probabilities, axis=1)
                pred_labels = np.argmax(probabilities, axis=1)
            else:
                pred_probs = probabilities
                pred_labels = (probabilities > 0.5).astype(int)
            
            # Create bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            # Calculate calibration metrics
            reliability = 0.0
            resolution = 0.0
            uncertainty = 0.0
            
            accuracies = []
            confidences = []
            bin_sizes = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Calculate accuracy in this bin
                    accuracy_in_bin = (pred_labels[in_bin] == true_labels[in_bin]).mean()
                    avg_confidence_in_bin = pred_probs[in_bin].mean()
                    
                    accuracies.append(accuracy_in_bin)
                    confidences.append(avg_confidence_in_bin)
                    bin_sizes.append(prop_in_bin)
                    
                    # Update reliability (calibration error)
                    reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
            
            # Expected Calibration Error (ECE)
            ece = sum(abs(acc - conf) * size for acc, conf, size in zip(accuracies, confidences, bin_sizes))
            
            # Maximum Calibration Error (MCE)
            mce = max(abs(acc - conf) for acc, conf in zip(accuracies, confidences)) if accuracies else 0
            
            # Brier Score
            brier_score = np.mean((pred_probs - (pred_labels == true_labels)) ** 2)
            
            return {
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'brier_score': brier_score,
                'reliability': reliability,
                'n_bins': n_bins
            }
            
        except Exception as e:
            logger.error(f"Calibration evaluation failed: {e}")
            return {'expected_calibration_error': 1.0, 'maximum_calibration_error': 1.0, 'brier_score': 1.0}


class TemperatureScaling:
    """Temperature scaling for calibrating neural network outputs."""
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, true_labels: np.ndarray, 
            max_iter: int = 50) -> None:
        """Fit temperature parameter using validation data."""
        try:
            logger.info("Fitting temperature scaling...")
            
            # Use scipy optimization to find optimal temperature
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temp):
                # Apply temperature scaling
                scaled_logits = logits / temp
                
                # Convert to probabilities
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # Calculate negative log likelihood
                nll = -np.mean(np.log(probabilities[np.arange(len(true_labels)), true_labels] + 1e-8))
                return nll
            
            # Find optimal temperature
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            
            if result.success:
                self.temperature = result.x
                self.is_fitted = True
                logger.info(f"Temperature scaling fitted. Optimal temperature: {self.temperature:.4f}")
            else:
                logger.warning("Temperature scaling optimization failed, using default temperature")
                self.temperature = 1.0
                self.is_fitted = True
                
        except Exception as e:
            logger.error(f"Temperature scaling fitting failed: {e}")
            self.temperature = 1.0
            self.is_fitted = True
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, returning original logits")
            return logits
        
        try:
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Convert to probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Temperature scaling transformation failed: {e}")
            # Fallback: simple softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class EnsembleCalibrator:
    """Ensemble-level probability calibration."""
    
    def __init__(self):
        self.model_weights = {}
        self.calibration_weights = {}
        self.is_fitted = False
    
    def fit(self, model_predictions: Dict[str, np.ndarray], 
            true_labels: np.ndarray, method: str = 'accuracy_weighted') -> None:
        """Fit ensemble calibration weights."""
        try:
            logger.info(f"Fitting ensemble calibration with method: {method}")
            
            if method == 'accuracy_weighted':
                # Weight models by their accuracy
                total_weight = 0
                for model_name, predictions in model_predictions.items():
                    if len(predictions.shape) > 1:
                        pred_labels = np.argmax(predictions, axis=1)
                    else:
                        pred_labels = (predictions > 0.5).astype(int)
                    
                    accuracy = np.mean(pred_labels == true_labels)
                    self.model_weights[model_name] = accuracy
                    total_weight += accuracy
                
                # Normalize weights
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
            elif method == 'brier_weighted':
                # Weight models by inverse Brier score
                total_weight = 0
                for model_name, predictions in model_predictions.items():
                    if len(predictions.shape) > 1:
                        pred_probs = np.max(predictions, axis=1)
                        pred_labels = np.argmax(predictions, axis=1)
                    else:
                        pred_probs = predictions
                        pred_labels = (predictions > 0.5).astype(int)
                    
                    # Calculate Brier score
                    brier_score = np.mean((pred_probs - (pred_labels == true_labels)) ** 2)
                    weight = 1.0 / (brier_score + 1e-8)  # Inverse Brier score
                    
                    self.model_weights[model_name] = weight
                    total_weight += weight
                
                # Normalize weights
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
            elif method == 'uniform':
                # Equal weights for all models
                n_models = len(model_predictions)
                for model_name in model_predictions:
                    self.model_weights[model_name] = 1.0 / n_models
            
            self.is_fitted = True
            
            logger.info(f"Ensemble weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Ensemble calibration fitting failed: {e}")
            # Fallback to uniform weights
            n_models = len(model_predictions)
            for model_name in model_predictions:
                self.model_weights[model_name] = 1.0 / n_models
            self.is_fitted = True
    
    def combine_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine model predictions using learned weights."""
        if not self.is_fitted:
            logger.warning("Ensemble calibrator not fitted, using uniform weights")
            n_models = len(model_predictions)
            weights = {name: 1.0/n_models for name in model_predictions.keys()}
        else:
            weights = self.model_weights
        
        try:
            # Combine predictions
            combined_prediction = None
            total_weight = 0
            
            for model_name, predictions in model_predictions.items():
                weight = weights.get(model_name, 0.0)
                
                if weight > 0:
                    if combined_prediction is None:
                        combined_prediction = weight * predictions
                    else:
                        combined_prediction += weight * predictions
                    total_weight += weight
            
            # Normalize if needed
            if total_weight > 0 and total_weight != 1.0:
                combined_prediction /= total_weight
            
            return combined_prediction
            
        except Exception as e:
            logger.error(f"Prediction combination failed: {e}")
            # Fallback: simple average
            predictions_list = list(model_predictions.values())
            return np.mean(predictions_list, axis=0)


class CalibrationMonitor:
    """Monitor calibration quality over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prediction_history = {}
        self.label_history = {}
        self.calibration_history = {}
    
    def update(self, model_name: str, predictions: np.ndarray, true_labels: np.ndarray) -> None:
        """Update calibration monitoring for a model."""
        try:
            # Initialize history if needed
            if model_name not in self.prediction_history:
                self.prediction_history[model_name] = []
                self.label_history[model_name] = []
                self.calibration_history[model_name] = []
            
            # Add new predictions
            self.prediction_history[model_name].extend(predictions.tolist())
            self.label_history[model_name].extend(true_labels.tolist())
            
            # Keep only recent history
            if len(self.prediction_history[model_name]) > self.window_size:
                self.prediction_history[model_name] = self.prediction_history[model_name][-self.window_size:]
                self.label_history[model_name] = self.label_history[model_name][-self.window_size:]
            
            # Calculate current calibration metrics
            if len(self.prediction_history[model_name]) >= 10:  # Minimum samples
                calibrator = ProbabilityCalibrator()
                recent_preds = np.array(self.prediction_history[model_name])
                recent_labels = np.array(self.label_history[model_name])
                
                metrics = calibrator.evaluate_calibration(recent_preds, recent_labels)
                self.calibration_history[model_name].append(metrics)
                
                # Keep calibration history manageable
                if len(self.calibration_history[model_name]) > 50:
                    self.calibration_history[model_name] = self.calibration_history[model_name][-50:]
            
        except Exception as e:
            logger.error(f"Calibration monitoring update failed for {model_name}: {e}")
    
    def get_calibration_trend(self, model_name: str) -> Dict[str, Any]:
        """Get calibration trend for a model."""
        if model_name not in self.calibration_history:
            return {'status': 'no_data'}
        
        try:
            history = self.calibration_history[model_name]
            if len(history) < 2:
                return {'status': 'insufficient_data'}
            
            # Get recent metrics
            recent_ece = [h['expected_calibration_error'] for h in history[-10:]]
            recent_brier = [h['brier_score'] for h in history[-10:]]
            
            # Calculate trends
            ece_trend = 'improving' if recent_ece[-1] < recent_ece[0] else 'degrading'
            brier_trend = 'improving' if recent_brier[-1] < recent_brier[0] else 'degrading'
            
            return {
                'status': 'ok',
                'current_ece': recent_ece[-1],
                'current_brier': recent_brier[-1],
                'ece_trend': ece_trend,
                'brier_trend': brier_trend,
                'history_length': len(history)
            }
            
        except Exception as e:
            logger.error(f"Calibration trend calculation failed for {model_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def should_recalibrate(self, model_name: str, ece_threshold: float = 0.1,
                          brier_threshold: float = 0.3) -> bool:
        """Determine if a model should be recalibrated."""
        try:
            trend = self.get_calibration_trend(model_name)
            
            if trend['status'] != 'ok':
                return False
            
            # Check if calibration quality is poor
            if (trend['current_ece'] > ece_threshold or 
                trend['current_brier'] > brier_threshold):
                return True
            
            # Check if calibration is degrading
            if (trend['ece_trend'] == 'degrading' and 
                trend['brier_trend'] == 'degrading'):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Recalibration check failed for {model_name}: {e}")
            return False


def apply_confidence_calibration(predictions: np.ndarray, confidence_scores: np.ndarray,
                                true_labels: np.ndarray, method: str = 'platt') -> Tuple[np.ndarray, np.ndarray]:
    """Apply calibration to both predictions and confidence scores."""
    try:
        calibrator = ProbabilityCalibrator()
        
        # Fit calibrator
        if method == 'platt':
            calibrator.fit_platt_scaling('confidence', confidence_scores, true_labels)
        elif method == 'isotonic':
            calibrator.fit_isotonic_regression('confidence', confidence_scores, true_labels)
        
        # Calibrate confidence scores
        calibrated_confidence = calibrator.calibrate_probabilities('confidence', confidence_scores, method)
        
        # Adjust predictions based on calibrated confidence
        calibrated_predictions = predictions.copy()
        
        # If confidence is low, move predictions toward uniform distribution
        uniform_dist = np.ones_like(predictions) / len(predictions)
        for i in range(len(predictions)):
            confidence = calibrated_confidence[i] if len(calibrated_confidence.shape) == 1 else np.max(calibrated_confidence[i])
            calibrated_predictions[i] = confidence * predictions[i] + (1 - confidence) * uniform_dist[i]
        
        return calibrated_predictions, calibrated_confidence
        
    except Exception as e:
        logger.error(f"Confidence calibration failed: {e}")
        return predictions, confidence_scores


def calculate_calibration_loss(probabilities: np.ndarray, true_labels: np.ndarray) -> float:
    """Calculate calibration loss for optimization."""
    try:
        calibrator = ProbabilityCalibrator()
        metrics = calibrator.evaluate_calibration(probabilities, true_labels)
        
        # Combine ECE and Brier score
        ece = metrics.get('expected_calibration_error', 1.0)
        brier = metrics.get('brier_score', 1.0)
        
        return 0.7 * ece + 0.3 * brier
        
    except Exception as e:
        logger.error(f"Calibration loss calculation failed: {e}")
        return 1.0


# Global calibrator instance
probability_calibrator = ProbabilityCalibrator()
calibration_monitor = CalibrationMonitor()

# Utility functions
def get_calibrated_predictions(model_name: str, predictions: np.ndarray, 
                              method: str = 'platt') -> np.ndarray:
    """Get calibrated predictions for a model."""
    return probability_calibrator.calibrate_probabilities(model_name, predictions, method)

def update_calibration_monitoring(model_name: str, predictions: np.ndarray, 
                                 true_labels: np.ndarray) -> None:
    """Update calibration monitoring for a model."""
    calibration_monitor.update(model_name, predictions, true_labels)

def check_recalibration_needed(model_name: str) -> bool:
    """Check if a model needs recalibration."""
    return calibration_monitor.should_recalibrate(model_name)