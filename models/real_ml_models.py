"""Modelos ML reales para el sistema de predicciones."""

import numpy as np
from collections import Counter
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)

# =============================================
# VERIFICAR DEPENDENCIAS
# =============================================
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# =============================================
# MODELOS ML REALES
# =============================================

class RealRandomForestModel(BasePredictionModel):
    """Random Forest REAL usando scikit-learn."""
    
    def __init__(self):
        super().__init__("realrandomforest", "1.0")
        self.model = None
        self.is_fitted = False
        self.use_sklearn = SKLEARN_AVAILABLE
        
    def fit(self, X, y):
        try:
            if self.use_sklearn and len(X) >= 20:
                from sklearn.ensemble import RandomForestRegressor
                
                # Crear secuencias para ML
                sequence_length = min(10, len(X) // 3)
                X_sequences = []
                y_targets = []
                
                for i in range(sequence_length, len(X)):
                    X_sequences.append(X[i-sequence_length:i])
                    y_targets.append(y[i])
                
                if len(X_sequences) >= 10:
                    X_ml = np.array(X_sequences)
                    y_ml = np.array(y_targets)
                    
                    self.model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=8,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    self.model.fit(X_ml, y_ml)
                    self.sequence_length = sequence_length
                    self.is_fitted = True
                    return
            
            # Fallback estadístico
            self._train_fallback(X, y)
            
        except Exception as e:
            self._train_fallback(X, y)
    
    def _train_fallback(self, X, y):
        all_data = np.concatenate([X.ravel(), y.ravel()])
        self.fallback_data = all_data[(all_data >= 0) & (all_data <= 99)]
        self.is_fitted = True
        self.model = None
    
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
            
        if self.model is None:  # Fallback
            if hasattr(self, 'fallback_data') and len(self.fallback_data) > 0:
                return np.array([int(np.mean(self.fallback_data[-20:]))])
            return np.array([50])
        
        # Predicción sklearn
        try:
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:].reshape(1, -1)
                pred = self.model.predict(X_seq)[0]
                return np.array([max(0, min(99, int(pred)))])
            else:
                return np.array([50])
        except:
            return np.array([50])
    
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        
        pred = self.predict(X)[0]
        probs = np.exp(-0.5 * ((np.arange(100) - pred) / 15)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


class RealXGBoostModel(BasePredictionModel):
    """XGBoost REAL usando xgboost."""
    
    def __init__(self):
        super().__init__("realxgboost", "1.0")
        self.model = None
        self.is_fitted = False
        self.use_xgboost = XGBOOST_AVAILABLE
        
    def fit(self, X, y):
        try:
            if self.use_xgboost and len(X) >= 20:
                import xgboost as xgb
                
                sequence_length = min(8, len(X) // 4)
                X_sequences = []
                y_targets = []
                
                for i in range(sequence_length, len(X)):
                    X_sequences.append(X[i-sequence_length:i])
                    y_targets.append(y[i])
                
                if len(X_sequences) >= 10:
                    X_ml = np.array(X_sequences)
                    y_ml = np.array(y_targets)
                    
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                    
                    self.model.fit(X_ml, y_ml)
                    self.sequence_length = sequence_length
                    self.is_fitted = True
                    return
            
            # Fallback
            self._train_fallback(X, y)
            
        except Exception as e:
            self._train_fallback(X, y)
    
    def _train_fallback(self, X, y):
        all_data = np.concatenate([X.ravel(), y.ravel()])
        self.fallback_data = all_data[(all_data >= 0) & (all_data <= 99)]
        self.is_fitted = True
        self.model = None
    
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
            
        if self.model is None:  # Fallback
            if hasattr(self, 'fallback_data') and len(self.fallback_data) > 0:
                recent = self.fallback_data[-15:]
                weights = np.linspace(0.5, 1.0, len(recent))
                pred = np.average(recent, weights=weights)
                return np.array([max(0, min(99, int(pred)))])
            return np.array([50])
        
        # Predicción XGBoost
        try:
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:].reshape(1, -1)
                pred = self.model.predict(X_seq)[0]
                return np.array([max(0, min(99, int(pred)))])
            else:
                return np.array([50])
        except:
            return np.array([50])
    
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        
        pred = self.predict(X)[0]
        probs = np.exp(-0.5 * ((np.arange(100) - pred) / 12)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


class RealLightGBMModel(BasePredictionModel):
    """LightGBM REAL usando lightgbm."""
    
    def __init__(self):
        super().__init__("reallightgbm", "1.0")
        self.model = None
        self.is_fitted = False
        self.use_lightgbm = LIGHTGBM_AVAILABLE
        
    def fit(self, X, y):
        try:
            if self.use_lightgbm and len(X) >= 20:
                import lightgbm as lgb
                
                sequence_length = min(12, len(X) // 3)
                X_sequences = []
                y_targets = []
                
                for i in range(sequence_length, len(X)):
                    X_sequences.append(X[i-sequence_length:i])
                    y_targets.append(y[i])
                
                if len(X_sequences) >= 10:
                    X_ml = np.array(X_sequences)
                    y_ml = np.array(y_targets)
                    
                    self.model = lgb.LGBMRegressor(
                        n_estimators=80,
                        max_depth=7,
                        learning_rate=0.15,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1
                    )
                    
                    self.model.fit(X_ml, y_ml)
                    self.sequence_length = sequence_length
                    self.is_fitted = True
                    return
            
            # Fallback
            self._train_fallback(X, y)
            
        except Exception as e:
            self._train_fallback(X, y)
    
    def _train_fallback(self, X, y):
        all_data = np.concatenate([X.ravel(), y.ravel()])
        all_data = all_data[(all_data >= 0) & (all_data <= 99)]
        
        self.data_stats = {
            'mean': np.mean(all_data),
            'std': np.std(all_data),
            'median': np.median(all_data),
            'recent_trend': np.mean(all_data[-25:]) if len(all_data) >= 25 else np.mean(all_data)
        }
        self.is_fitted = True
        self.model = None
    
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
            
        if self.model is None:  # Fallback
            if hasattr(self, 'data_stats'):
                pred = int(0.4 * self.data_stats['recent_trend'] + 
                          0.3 * self.data_stats['median'] +
                          0.3 * self.data_stats['mean'])
                return np.array([max(0, min(99, pred))])
            return np.array([50])
        
        # Predicción LightGBM
        try:
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:].reshape(1, -1)
                pred = self.model.predict(X_seq)[0]
                return np.array([max(0, min(99, int(pred)))])
            else:
                return np.array([50])
        except:
            return np.array([50])
    
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        
        pred = self.predict(X)[0]
        std_dev = 10 if self.model else 15
        probs = np.exp(-0.5 * ((np.arange(100) - pred) / std_dev)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


class NeuralNetworkModel(BasePredictionModel):
    """Red Neuronal simple usando NumPy puro."""
    
    def __init__(self):
        super().__init__("neuralnetwork", "1.0")
        self.weights = None
        self.is_fitted = False
        
    def fit(self, X, y):
        try:
            if len(X) < 30:
                raise ValueError("Insufficient data for Neural Network")
            
            sequence_length = min(15, len(X) // 2)
            X_sequences = []
            y_targets = []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i-sequence_length:i])
                y_targets.append(y[i] / 99.0)  # Normalizar
            
            if len(X_sequences) < 15:
                raise ValueError("Insufficient sequences for NN")
                
            X_ml = np.array(X_sequences) / 99.0  # Normalizar entradas
            y_ml = np.array(y_targets)
            
            # Red neuronal simple
            input_size = sequence_length
            hidden_size = 32
            
            # Inicializar pesos
            np.random.seed(42)
            self.W1 = np.random.randn(input_size, hidden_size) * 0.5
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, 1) * 0.5
            self.b2 = np.zeros(1)
            
            # Entrenamiento
            learning_rate = 0.01
            epochs = 50
            
            for epoch in range(epochs):
                # Forward pass
                h1 = np.maximum(0, X_ml.dot(self.W1) + self.b1)  # ReLU
                output = h1.dot(self.W2) + self.b2
                
                # Loss (MSE)
                loss = np.mean((output.flatten() - y_ml) ** 2)
                
                # Backward pass
                d_output = 2 * (output.flatten() - y_ml) / len(y_ml)
                d_W2 = h1.T.dot(d_output.reshape(-1, 1))
                d_b2 = np.sum(d_output)
                d_h1 = d_output.reshape(-1, 1).dot(self.W2.T)
                d_h1[h1 <= 0] = 0  # ReLU derivative
                d_W1 = X_ml.T.dot(d_h1)
                d_b1 = np.sum(d_h1, axis=0)
                
                # Update weights
                self.W1 -= learning_rate * d_W1
                self.b1 -= learning_rate * d_b1
                self.W2 -= learning_rate * d_W2
                self.b2 -= learning_rate * d_b2
            
            self.sequence_length = sequence_length
            self.is_fitted = True
            
        except Exception as e:
            all_data = np.concatenate([X.ravel(), y.ravel()])
            self.fallback_data = all_data[(all_data >= 0) & (all_data <= 99)]
            self.is_fitted = True
            self.weights = None
    
    def predict(self, X):
        if not self.is_fitted:
            return np.array([50])
            
        if not hasattr(self, 'W1'):  # Fallback
            if hasattr(self, 'fallback_data') and len(self.fallback_data) > 0:
                return np.array([int(np.mean(self.fallback_data[-10:]))])
            return np.array([50])
        
        # Predicción con red neuronal
        try:
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:] / 99.0  # Normalizar
                h1 = np.maximum(0, X_seq.dot(self.W1) + self.b1)
                output = h1.dot(self.W2) + self.b2
                pred = output[0] * 99.0  # Desnormalizar
                return np.array([max(0, min(99, int(pred)))])
            else:
                return np.array([50])
        except:
            return np.array([50])
    
    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones((1, 100)) / 100
        
        pred = self.predict(X)[0]
        probs = np.exp(-0.5 * ((np.arange(100) - pred) / 8)**2)
        return (probs / np.sum(probs)).reshape(1, -1)


# =============================================
# REGISTRO AUTOMÁTICO DE MODELOS
# =============================================

def register_real_ml_models():
    """Registrar automáticamente todos los modelos ML reales."""
    
    models_to_register = [
        (RealRandomForestModel, lambda: RealRandomForestModel()),
        (RealXGBoostModel, lambda: RealXGBoostModel()),
        (RealLightGBMModel, lambda: RealLightGBMModel()),
        (NeuralNetworkModel, lambda: NeuralNetworkModel())
    ]
    
    registered_count = 0
    
    for model_class, factory in models_to_register:
        try:
            model_registry.register_model(model_class, factory)
            registered_count += 1
            logger.info(f"[REAL-ML] Successfully registered {model_class.__name__}")
        except Exception as e:
            logger.error(f"[REAL-ML] Failed to register {model_class.__name__}: {e}")
    
    logger.info(f"[REAL-ML] Registered {registered_count}/4 real ML models")
    return registered_count

# Auto-registrar al importar el módulo
register_real_ml_models()

# Verificar registro
available_models = model_registry.list_available_models()
logger.info(f"[REAL-ML] Total models available after registration: {len(available_models)}")
logger.info(f"[REAL-ML] Models: {available_models}")
