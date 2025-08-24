"""Advanced Bayesian models for lottery prediction."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from scipy.special import gammaln, digamma
from collections import defaultdict
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class DirichletMultinomialModel(BasePredictionModel):
    """Dirichlet-Multinomial Bayesian model."""
    
    def __init__(self, alpha_prior: float = 1.0, concentration: float = 10.0):
        super().__init__("dirichlet_multinomial", "1.0")
        self.alpha_prior = alpha_prior
        self.concentration = concentration
        self.alpha_posterior = None
        self.is_fitted = False
        
        self.metadata.description = "Dirichlet-Multinomial with conjugate priors"
        self.metadata.parameters = {
            'alpha_prior': alpha_prior,
            'concentration': concentration
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Dirichlet-Multinomial model."""
        try:
            logger.info("Training Dirichlet-Multinomial model...")
            
            # Combine data
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            # Count observations
            counts = np.zeros(100)
            for num in all_data:
                counts[int(num)] += 1
            
            # Update posterior
            self.alpha_posterior = self.alpha_prior + counts
            
            self.is_fitted = True
            self.metadata.training_samples = len(all_data)
            
            logger.info("Dirichlet-Multinomial training completed")
            
        except Exception as e:
            logger.error(f"Dirichlet-Multinomial training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction using posterior mode."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Mode of Dirichlet distribution
            alpha_sum = np.sum(self.alpha_posterior)
            probabilities = (self.alpha_posterior - 1) / (alpha_sum - 100)
            
            # Handle numerical issues
            probabilities = np.maximum(probabilities, 0)
            probabilities /= np.sum(probabilities)
            
            predicted_number = np.argmax(probabilities)
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"Dirichlet-Multinomial prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Expected probabilities from Dirichlet
            alpha_sum = np.sum(self.alpha_posterior)
            probabilities = self.alpha_posterior / alpha_sum
            
            return probabilities.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Dirichlet-Multinomial probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class BetaBinomialModel(BasePredictionModel):
    """Beta-Binomial model for binary lottery features."""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        super().__init__("beta_binomial", "1.0")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.posteriors = {}
        self.is_fitted = False
        
        self.metadata.description = "Beta-Binomial for binary features"
        self.metadata.parameters = {
            'alpha_prior': alpha_prior,
            'beta_prior': beta_prior
        }
    
    def _extract_binary_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract binary features from lottery numbers."""
        features = {}
        
        # Even/odd
        features['even'] = (data % 2 == 0).astype(int)
        
        # High/low (>=50)
        features['high'] = (data >= 50).astype(int)
        
        # Decade features
        for decade in range(10):
            features[f'decade_{decade}'] = ((data >= decade*10) & (data < (decade+1)*10)).astype(int)
        
        # Ending digit features
        for digit in range(10):
            features[f'ends_{digit}'] = (data % 10 == digit).astype(int)
        
        return features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Beta-Binomial model."""
        try:
            logger.info("Training Beta-Binomial model...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            # Extract features
            features = self._extract_binary_features(all_data)
            
            # Update posteriors for each feature
            self.posteriors = {}
            for feature_name, feature_values in features.items():
                successes = np.sum(feature_values)
                failures = len(feature_values) - successes
                
                # Beta posterior parameters
                alpha_post = self.alpha_prior + successes
                beta_post = self.beta_prior + failures
                
                self.posteriors[feature_name] = {
                    'alpha': alpha_post,
                    'beta': beta_post,
                    'probability': alpha_post / (alpha_post + beta_post)
                }
            
            self.is_fitted = True
            self.metadata.training_samples = len(all_data)
            
            logger.info("Beta-Binomial training completed")
            
        except Exception as e:
            logger.error(f"Beta-Binomial training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction using feature probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Score each number based on feature probabilities
            scores = np.zeros(100)
            
            for num in range(100):
                features = self._extract_binary_features(np.array([num]))
                
                score = 1.0
                for feature_name, feature_value in features.items():
                    if feature_name in self.posteriors:
                        prob = self.posteriors[feature_name]['probability']
                        if feature_value[0] == 1:
                            score *= prob
                        else:
                            score *= (1 - prob)
                
                scores[num] = score
            
            # Normalize scores
            scores /= np.sum(scores)
            
            predicted_number = np.argmax(scores)
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"Beta-Binomial prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            scores = np.zeros(100)
            
            for num in range(100):
                features = self._extract_binary_features(np.array([num]))
                
                score = 1.0
                for feature_name, feature_value in features.items():
                    if feature_name in self.posteriors:
                        prob = self.posteriors[feature_name]['probability']
                        if feature_value[0] == 1:
                            score *= prob
                        else:
                            score *= (1 - prob)
                
                scores[num] = score
            
            scores /= np.sum(scores)
            return scores.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Beta-Binomial probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class GaussianProcessModel(BasePredictionModel):
    """Simplified Gaussian Process for lottery prediction."""
    
    def __init__(self, length_scale: float = 1.0, noise_level: float = 0.1):
        super().__init__("gaussian_process", "1.0")
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.is_fitted = False
        
        self.metadata.description = "Gaussian Process with RBF kernel"
        self.metadata.parameters = {
            'length_scale': length_scale,
            'noise_level': noise_level
        }
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel function."""
        dist_matrix = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-dist_matrix / (2 * self.length_scale**2))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian Process model."""
        try:
            logger.info("Training Gaussian Process model...")
            
            # Use recent data for GP (computationally expensive)
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            # Limit data size for computational efficiency
            if len(all_data) > 100:
                all_data = all_data[-100:]
            
            # Create training data
            if len(all_data) < 10:
                raise ValueError("Insufficient data for GP training")
            
            self.X_train = all_data[:-1].reshape(-1, 1)
            self.y_train = all_data[1:]
            
            # Compute kernel matrix
            K = self._rbf_kernel(self.X_train, self.X_train)
            K += self.noise_level * np.eye(len(self.X_train))
            
            # Invert kernel matrix
            self.K_inv = np.linalg.inv(K)
            
            self.is_fitted = True
            self.metadata.training_samples = len(self.X_train)
            
            logger.info("Gaussian Process training completed")
            
        except Exception as e:
            logger.error(f"Gaussian Process training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate GP prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Use last value for prediction
            if len(X) > 0:
                X_test = np.array([[X[-1]]])
            else:
                X_test = np.array([[50]])  # Default
            
            # Compute kernel between test and training points
            K_star = self._rbf_kernel(X_test, self.X_train)
            
            # GP mean prediction
            mean_pred = np.dot(K_star, np.dot(self.K_inv, self.y_train))
            
            # Clamp to valid range
            predicted_number = int(np.clip(mean_pred[0], 0, 99))
            
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"Gaussian Process prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate GP prediction with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Use last value
            if len(X) > 0:
                X_test = np.array([[X[-1]]])
            else:
                X_test = np.array([[50]])
            
            K_star = self._rbf_kernel(X_test, self.X_train)
            K_star_star = self._rbf_kernel(X_test, X_test)
            
            # GP mean and variance
            mean_pred = np.dot(K_star, np.dot(self.K_inv, self.y_train))
            var_pred = K_star_star - np.dot(K_star, np.dot(self.K_inv, K_star.T))
            
            # Create probability distribution around prediction
            mean = mean_pred[0]
            std = np.sqrt(max(var_pred[0, 0], 0.1))
            
            # Generate probabilities for each number
            probabilities = np.zeros(100)
            for i in range(100):
                probabilities[i] = stats.norm.pdf(i, mean, std)
            
            # Normalize
            probabilities /= np.sum(probabilities)
            
            return probabilities.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Gaussian Process probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class BayesianLinearRegressionModel(BasePredictionModel):
    """Bayesian Linear Regression for lottery prediction."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__("bayesian_linear", "1.0")
        self.alpha = alpha  # Precision of prior
        self.beta = beta    # Precision of noise
        self.mean_posterior = None
        self.cov_posterior = None
        self.is_fitted = False
        
        self.metadata.description = "Bayesian Linear Regression with polynomial features"
        self.metadata.parameters = {
            'alpha': alpha,
            'beta': beta
        }
    
    def _create_polynomial_features(self, X: np.ndarray, degree: int = 3) -> np.ndarray:
        """Create polynomial features."""
        features = []
        n = len(X)
        
        for i in range(n):
            feature_row = [1]  # Bias term
            
            # Polynomial terms
            for d in range(1, degree + 1):
                feature_row.append(X[i] ** d)
            
            # Interaction terms with previous values
            if i > 0:
                feature_row.append(X[i] * X[i-1])
            else:
                feature_row.append(0)
            
            if i > 1:
                feature_row.append(X[i] * X[i-2])
            else:
                feature_row.append(0)
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Bayesian Linear Regression."""
        try:
            logger.info("Training Bayesian Linear Regression...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 10:
                raise ValueError("Insufficient data for Bayesian Linear Regression")
            
            # Create features and targets
            X_features = self._create_polynomial_features(all_data[:-1])
            y_targets = all_data[1:]
            
            # Bayesian linear regression
            # Prior: w ~ N(0, (1/alpha) * I)
            S0_inv = self.alpha * np.eye(X_features.shape[1])
            
            # Posterior covariance
            SN_inv = S0_inv + self.beta * np.dot(X_features.T, X_features)
            self.cov_posterior = np.linalg.inv(SN_inv)
            
            # Posterior mean
            self.mean_posterior = self.beta * np.dot(
                self.cov_posterior, np.dot(X_features.T, y_targets)
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_features)
            
            logger.info("Bayesian Linear Regression training completed")
            
        except Exception as e:
            logger.error(f"Bayesian Linear Regression training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction using posterior mean."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Use recent values for prediction
            if len(X) >= 2:
                recent_data = X[-2:]
            else:
                recent_data = np.concatenate([np.array([50]), X])
            
            # Create features
            X_test = self._create_polynomial_features(recent_data)
            
            # Prediction using posterior mean
            prediction = np.dot(X_test[-1], self.mean_posterior)
            
            # Clamp to valid range
            predicted_number = int(np.clip(prediction, 0, 99))
            
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"Bayesian Linear Regression prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Use recent values
            if len(X) >= 2:
                recent_data = X[-2:]
            else:
                recent_data = np.concatenate([np.array([50]), X])
            
            X_test = self._create_polynomial_features(recent_data)
            x_test = X_test[-1]
            
            # Predictive distribution
            mean_pred = np.dot(x_test, self.mean_posterior)
            var_pred = (1/self.beta) + np.dot(x_test, np.dot(self.cov_posterior, x_test))
            std_pred = np.sqrt(var_pred)
            
            # Generate probabilities
            probabilities = np.zeros(100)
            for i in range(100):
                probabilities[i] = stats.norm.pdf(i, mean_pred, std_pred)
            
            probabilities /= np.sum(probabilities)
            return probabilities.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Bayesian Linear Regression probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


# Register Bayesian models
def create_dirichlet_multinomial(**kwargs):
    return DirichletMultinomialModel(**kwargs)

def create_beta_binomial(**kwargs):
    return BetaBinomialModel(**kwargs)

def create_gaussian_process(**kwargs):
    return GaussianProcessModel(**kwargs)

def create_bayesian_linear(**kwargs):
    return BayesianLinearRegressionModel(**kwargs)

model_registry.register_model(DirichletMultinomialModel, create_dirichlet_multinomial)
model_registry.register_model(BetaBinomialModel, create_beta_binomial)
model_registry.register_model(GaussianProcessModel, create_gaussian_process)
model_registry.register_model(BayesianLinearRegressionModel, create_bayesian_linear)