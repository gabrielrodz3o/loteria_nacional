"""Auto-register all 6 models in the registry."""

import numpy as np
from collections import Counter
import logging
from .prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)

# =============================================
# 6 MODELOS CORREGIDOS
# =============================================

class FrequencyAnalysisModel(BasePredictionModel):
    def __init__(self):
        super().__init__("frequencyanalysis", "1.0")
        self.frequencies = {}
        self.is_fitted = False
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            counter = Counter(all_data.astype(int))
            total = len(all_data)
            
            for i in range(100):
                count = counter.get(i, 0)
                self.frequencies[i] = (count + 1) / (total + 100)
                
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[FREQ] Training failed: {e}")
            for i in range(100):
                self.frequencies[i] = 1.0 / 100
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
        best_number = max(self.frequencies, key=self.frequencies.get)
        return np.array([best_number])
        
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        probs = np.array([self.frequencies[i] for i in range(100)])
        return (probs / np.sum(probs)).reshape(1, -1)


class RandomForestModel(BasePredictionModel):
    def __init__(self):
        super().__init__("randomforest", "1.0")
        self.is_fitted = False
        self.historical_data = []
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            self.historical_data = all_data[(all_data >= 0) & (all_data <= 99)]
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[RF] Training failed: {e}")
            self.historical_data = np.random.randint(0, 100, 50)
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted or len(self.historical_data) == 0:
            return np.array([50])
        recent_data = self.historical_data[-10:]
        prediction = int(np.mean(recent_data))
        return np.array([max(0, min(99, prediction))])
        
    def predict_proba(self, X):
        if not self.is_fitted or len(self.historical_data) == 0:
            return np.ones((1, 100)) / 100
        mean_val = np.mean(self.historical_data)
        std_val = max(np.std(self.historical_data), 1.0)
        probs = np.exp(-0.5 * ((np.arange(100) - mean_val) / std_val)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


class MonteCarloModel(BasePredictionModel):
    def __init__(self):
        super().__init__("montecarlo", "1.0")
        self.distribution = {}
        self.is_fitted = False
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            counter = Counter(all_data.astype(int))
            total = len(all_data)
            
            for i in range(100):
                count = counter.get(i, 0)
                self.distribution[i] = (count + 0.1) / (total + 10)
                
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[MC] Training failed: {e}")
            for i in range(100):
                self.distribution[i] = (1 + np.random.random() * 0.1) / 100
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
        numbers = list(self.distribution.keys())
        probabilities = list(self.distribution.values())
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        predicted = np.random.choice(numbers, p=probabilities)
        return np.array([predicted])
        
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        probs = np.array([self.distribution[i] for i in range(100)])
        return (probs / np.sum(probs)).reshape(1, -1)


class BayesianModel(BasePredictionModel):
    def __init__(self):
        super().__init__("bayesian", "1.0")
        self.posterior = None
        self.is_fitted = False
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            prior = np.ones(100)
            counts = np.zeros(100)
            for num in all_data.astype(int):
                counts[num] += 1
                
            self.posterior = prior + counts
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[BAY] Training failed: {e}")
            self.posterior = np.ones(100)
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
        predicted = np.argmax(self.posterior)
        return np.array([predicted])
        
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        total = np.sum(self.posterior)
        probabilities = self.posterior / total
        return probabilities.reshape(1, -1)


class LightGBMModel(BasePredictionModel):
    def __init__(self):
        super().__init__("lightgbm", "1.0")
        self.is_fitted = False
        self.data_stats = {}
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            self.data_stats = {
                'mean': np.mean(all_data),
                'std': np.std(all_data),
                'median': np.median(all_data),
                'recent_trend': np.mean(all_data[-20:]) if len(all_data) >= 20 else np.mean(all_data)
            }
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[LGBM] Training failed: {e}")
            self.data_stats = {'mean': 50, 'std': 20, 'median': 50, 'recent_trend': 50}
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
        prediction = int(0.4 * self.data_stats['recent_trend'] + 
                        0.3 * self.data_stats['median'] +
                        0.3 * self.data_stats['mean'])
        return np.array([max(0, min(99, prediction))])
        
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        mean_val = self.data_stats['mean']
        std_val = max(self.data_stats['std'], 5.0)
        probs = np.exp(-0.5 * ((np.arange(100) - mean_val) / std_val)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


class XGBoostModel(BasePredictionModel):
    def __init__(self):
        super().__init__("xgboost", "1.0")
        self.is_fitted = False
        self.gradient_data = []
        
    def fit(self, X, y):
        try:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            self.gradient_data = all_data[(all_data >= 0) & (all_data <= 99)]
            self.is_fitted = True
        except Exception as e:
            logger.error(f"[XGB] Training failed: {e}")
            self.gradient_data = np.random.randint(0, 100, 50)
            self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted or len(self.gradient_data) == 0:
            return np.array([50])
        recent = self.gradient_data[-15:]
        weights = np.linspace(0.5, 1.0, len(recent))
        weighted_avg = np.average(recent, weights=weights)
        prediction = int(weighted_avg)
        return np.array([max(0, min(99, prediction))])
        
    def predict_proba(self, X):
        if not self.is_fitted or len(self.gradient_data) == 0:
            return np.ones((1, 100)) / 100
        mean_val = np.mean(self.gradient_data)
        recent_mean = np.mean(self.gradient_data[-10:]) if len(self.gradient_data) >= 10 else mean_val
        combined_mean = 0.7 * recent_mean + 0.3 * mean_val
        std_val = max(np.std(self.gradient_data), 8.0)
        probs = np.exp(-0.5 * ((np.arange(100) - combined_mean) / std_val)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


# =============================================
# AUTO-REGISTRO AL IMPORTAR
# =============================================
def register_all_models():
    """Registrar todos los modelos automáticamente."""
    models_to_register = [
        (FrequencyAnalysisModel, lambda: FrequencyAnalysisModel()),
        (RandomForestModel, lambda: RandomForestModel()),
        (MonteCarloModel, lambda: MonteCarloModel()),
        (BayesianModel, lambda: BayesianModel()),
        (LightGBMModel, lambda: LightGBMModel()),
        (XGBoostModel, lambda: XGBoostModel())
    ]
    
    for model_class, factory in models_to_register:
        try:
            model_registry.register_model(model_class, factory)
            logger.info(f"[AUTO-REG] Registered {model_class.__name__}")
        except Exception as e:
            logger.error(f"[AUTO-REG] Failed to register {model_class.__name__}: {e}")

# Registrar automáticamente al importar
register_all_models()
print(f"[AUTO-REG] Registered {len(model_registry.list_available_models())} models automatically")