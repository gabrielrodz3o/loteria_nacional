"""Random Forest models for lottery prediction."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class RandomForestModel(BasePredictionModel):
    """Random Forest classifier for lottery prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 5, random_state: int = 42):
        super().__init__("random_forest", "1.0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.metadata.description = "Random Forest with feature engineering"
        self.metadata.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state
        }
    
    def _create_features(self, data: np.ndarray, lookback: int = 10) -> np.ndarray:
        """Create features from time series data."""
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
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                len(np.unique(window)),
                np.sum(window % 2),  # Even count
                np.sum(window >= 50),  # High numbers
            ])
            
            # Trend features
            if len(window) > 1:
                diffs = np.diff(window)
                feature_row.extend([
                    np.mean(diffs),
                    np.std(diffs),
                    np.sum(diffs > 0),
                    np.sum(diffs < 0),
                ])
            else:
                feature_row.extend([0, 0, 0, 0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        try:
            logger.info("Training Random Forest model...")
            
            # Prepare data
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 20:
                raise ValueError("Insufficient data for Random Forest training")
            
            # Create features
            features = self._create_features(all_data)
            targets = all_data[10:]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Encode targets
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets_encoded, test_size=0.2, random_state=self.random_state
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            # Calculate accuracy
            train_acc = self.model.score(X_train, y_train)
            val_acc = self.model.score(X_val, y_val)
            self.metadata.accuracy_score = val_acc
            
            logger.info(f"Random Forest training completed. Train: {train_acc:.4f}, Val: {val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using Random Forest."""
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
            logger.error(f"Random Forest prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using Random Forest."""
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
            logger.error(f"Random Forest probability prediction failed: {e}")
            return np.ones((1, 100)) / 100
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Random Forest."""
        if not self.is_fitted or self.model is None:
            return None
        
        try:
            feature_names = [f'lag_{i}' for i in range(10)]
            feature_names.extend(['mean', 'std', 'min', 'max', 'median', 
                                'q25', 'q75', 'unique_count', 'even_count', 'high_count',
                                'trend_mean', 'trend_std', 'increasing_count', 'decreasing_count'])
            
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None


class ExtraTreesModel(BasePredictionModel):
    """Extra Trees classifier for lottery prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        super().__init__("extra_trees", "1.0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.metadata.description = "Extra Trees with randomized splits"
        self.metadata.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
    
    def _create_features(self, data: np.ndarray, lookback: int = 8) -> np.ndarray:
        """Create features optimized for Extra Trees."""
        features = []
        
        for i in range(lookback, len(data)):
            feature_row = []
            
            # Recent values
            window = data[i-lookback:i]
            feature_row.extend(window)
            
            # Simple aggregations
            feature_row.extend([
                np.mean(window),
                np.std(window),
                window[-1] - window[0],  # Recent change
                np.sum(window % 10 == 0),  # Multiples of 10
                len(set(window)),  # Unique values
            ])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Extra Trees model."""
        try:
            logger.info("Training Extra Trees model...")
            
            # Import here to avoid dependency issues
            from sklearn.ensemble import ExtraTreesClassifier
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 16:
                raise ValueError("Insufficient data for Extra Trees training")
            
            features = self._create_features(all_data)
            targets = all_data[8:]
            
            features_scaled = self.scaler.fit_transform(features)
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets_encoded, test_size=0.2, random_state=self.random_state
            )
            
            self.model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            val_acc = self.model.score(X_val, y_val)
            self.metadata.accuracy_score = val_acc
            
            logger.info(f"Extra Trees training completed. Accuracy: {val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Extra Trees training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using Extra Trees."""
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
            logger.error(f"Extra Trees prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using Extra Trees."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            probabilities = self.model.predict_proba(features_scaled)
            
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"Extra Trees probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


# Register models
def create_random_forest(**kwargs):
    return RandomForestModel(**kwargs)

def create_extra_trees(**kwargs):
    return ExtraTreesModel(**kwargs)

model_registry.register_model(RandomForestModel, create_random_forest)
model_registry.register_model(ExtraTreesModel, create_extra_trees)