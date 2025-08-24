"""Gradient boosting models for lottery prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class XGBoostModel(BasePredictionModel):
    """XGBoost model for lottery prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("xgboost", "1.0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        self.metadata.description = "XGBoost gradient boosting classifier"
        self.metadata.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
    
    def _create_features(self, data: np.ndarray, lookback: int = 10) -> np.ndarray:
        """Create features from time series data."""
        features = []
        
        for i in range(lookback, len(data)):
            feature_row = []
            
            # Historical values
            for j in range(lookback):
                feature_row.append(data[i - j - 1])
            
            # Statistical features
            window = data[i-lookback:i]
            feature_row.extend([
                np.mean(window),           # Mean
                np.std(window),            # Standard deviation
                np.min(window),            # Min
                np.max(window),            # Max
                np.median(window),         # Median
                len(np.unique(window)),    # Unique count
                np.sum(window % 2),        # Even count
                np.sum(window >= 50),      # High numbers count
            ])
            
            # Trend features
            if len(window) > 1:
                feature_row.extend([
                    window[-1] - window[0],    # Overall change
                    np.mean(np.diff(window)),  # Average change
                    np.std(np.diff(window)),   # Change volatility
                ])
            else:
                feature_row.extend([0, 0, 0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        try:
            logger.info("Training XGBoost model...")
            
            # Combine data for feature creation
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 20:
                raise ValueError("Insufficient data for XGBoost training")
            
            # Create features
            features = self._create_features(all_data)
            targets = all_data[10:]  # Skip first 10 due to lookback
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Encode targets
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets_encoded, test_size=0.2, random_state=self.random_state
            )
            
            # Configure XGBoost
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                objective='multi:softprob',
                eval_metric='mlogloss',
                early_stopping_rounds=10,
                n_jobs=-1
            )
            
            # Train model
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            # Calculate accuracy
            train_acc = self.model.score(X_train, y_train)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"XGBoost training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using XGBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Create features from input
            features = self._create_features(X)
            
            if len(features) == 0:
                # Not enough data, return random prediction
                return np.array([np.random.randint(0, 100)])
            
            # Use last feature row for prediction
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            # Predict
            prediction_encoded = self.model.predict(features_scaled)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using XGBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Create features
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            # Use last feature row
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            # Get probabilities
            probabilities = self.model.predict_proba(features_scaled)
            
            # Map to full 100-number range
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            # Normalize
            full_probs /= np.sum(full_probs)
            
            return full_probs
            
        except Exception as e:
            logger.error(f"XGBoost probability prediction failed: {e}")
            return np.ones((1, 100)) / 100
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from XGBoost."""
        if not self.is_fitted or self.model is None:
            return None
        
        try:
            feature_names = [f'lag_{i}' for i in range(10)]
            feature_names.extend(['mean', 'std', 'min', 'max', 'median', 
                                'unique_count', 'even_count', 'high_count',
                                'overall_change', 'avg_change', 'change_volatility'])
            
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None


class LightGBMModel(BasePredictionModel):
    """LightGBM model for lottery prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        super().__init__("lightgbm", "1.0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.metadata.description = "LightGBM gradient boosting classifier"
        self.metadata.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
    
    def _create_features(self, data: np.ndarray, lookback: int = 8) -> np.ndarray:
        """Create features optimized for LightGBM."""
        features = []
        
        for i in range(lookback, len(data)):
            feature_row = []
            
            # Recent values
            window = data[i-lookback:i]
            feature_row.extend(window)
            
            # Aggregated features
            feature_row.extend([
                np.mean(window),
                np.std(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                np.sum(window < 25),      # Low numbers
                np.sum((window >= 25) & (window < 75)),  # Mid numbers
                np.sum(window >= 75),     # High numbers
            ])
            
            # Pattern features
            if len(window) > 1:
                diffs = np.diff(window)
                feature_row.extend([
                    np.mean(diffs),        # Average change
                    np.std(diffs),         # Change volatility
                    np.sum(diffs > 0),     # Increasing count
                    np.sum(diffs < 0),     # Decreasing count
                ])
            else:
                feature_row.extend([0, 0, 0, 0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM model."""
        try:
            logger.info("Training LightGBM model...")
            
            # Prepare data
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 16:
                raise ValueError("Insufficient data for LightGBM training")
            
            # Create features
            features = self._create_features(all_data)
            targets = all_data[8:]  # Skip first 8 due to lookback
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Encode targets
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets_encoded, test_size=0.2, random_state=self.random_state
            )
            
            # Configure LightGBM
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                objective='multiclass',
                metric='multi_logloss',
                verbosity=-1,
                n_jobs=-1
            )
            
            # Train model
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            # Calculate accuracy
            train_acc = self.model.score(X_train, y_train)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"LightGBM training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using LightGBM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.array([np.random.randint(0, 100)])
            
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            prediction_encoded = self.model.predict(features_scaled)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using LightGBM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            probabilities = self.model.predict_proba(features_scaled)
            
            # Map to full range
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"LightGBM probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class CatBoostModel(BasePredictionModel):
    """CatBoost model for lottery prediction."""
    
    def __init__(self, iterations: int = 100, depth: int = 6, 
                 learning_rate: float = 0.1, random_seed: int = 42):
        super().__init__("catboost", "1.0")
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self.metadata.description = "CatBoost gradient boosting classifier"
        self.metadata.parameters = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'random_seed': random_seed
        }
    
    def _create_features(self, data: np.ndarray, lookback: int = 12) -> np.ndarray:
        """Create features for CatBoost."""
        features = []
        
        for i in range(lookback, len(data)):
            feature_row = []
            
            # Historical values
            window = data[i-lookback:i]
            feature_row.extend(window)
            
            # Statistical features
            feature_row.extend([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.sum(window % 10 == 0),     # Numbers ending in 0
                np.sum(window % 5 == 0),      # Numbers ending in 0 or 5
                len(set(window)),             # Unique count
            ])
            
            # Categorical features (decade)
            last_decade = int(window[-1] // 10)
            feature_row.append(last_decade)
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the CatBoost model."""
        if cb is None:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        
        try:
            logger.info("Training CatBoost model...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 24:
                raise ValueError("Insufficient data for CatBoost training")
            
            features = self._create_features(all_data)
            targets = all_data[12:]
            
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets_encoded, test_size=0.2, random_state=self.random_seed
            )
            
            # Specify categorical features
            cat_features = [len(features[0]) - 1]  # Last feature (decade)
            
            self.model = cb.CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_seed=self.random_seed,
                cat_features=cat_features,
                verbose=False,
                early_stopping_rounds=20
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            train_acc = self.model.score(X_train, y_train)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"CatBoost training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using CatBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.array([np.random.randint(0, 100)])
            
            last_features = features[-1:].reshape(1, -1)
            prediction_encoded = self.model.predict(last_features)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"CatBoost prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using CatBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            probabilities = self.model.predict_proba(last_features)
            
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"CatBoost probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


# Register gradient boosting models
def create_xgboost(**kwargs):
    return XGBoostModel(**kwargs)

def create_lightgbm(**kwargs):
    return LightGBMModel(**kwargs)

def create_catboost(**kwargs):
    return CatBoostModel(**kwargs)

model_registry.register_model(XGBoostModel, create_xgboost)
model_registry.register_model(LightGBMModel, create_lightgbm)
if cb is not None:
    model_registry.register_model(CatBoostModel, create_catboost)