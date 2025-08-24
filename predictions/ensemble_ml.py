"""Ensemble machine learning models for lottery prediction."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class VotingEnsembleModel(BasePredictionModel):
    """Voting ensemble combining multiple base models."""
    
    def __init__(self, voting: str = 'soft', weights: Optional[List[float]] = None):
        super().__init__("voting_ensemble", "1.0")
        self.voting = voting
        self.weights = weights
        self.ensemble = None
        self.label_encoder = LabelEncoder()
        self.base_models = []
        self.is_fitted = False
        
        self.metadata.description = "Voting ensemble of multiple algorithms"
        self.metadata.parameters = {
            'voting': voting,
            'weights': weights
        }
    
    def _create_base_models(self):
        """Create base models for ensemble."""
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42)),
            ('nb', GaussianNB()),
        ]
        
        # Add SVM only for soft voting (needs probability estimates)
        if self.voting == 'soft':
            base_models.append(('svm', SVC(probability=True, random_state=42)))
        
        return base_models
    
    def _create_features(self, data: np.ndarray, lookback: int = 12) -> np.ndarray:
        """Create features for ensemble models."""
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
                len(np.unique(window)),
                np.sum(window % 2),
                np.sum(window >= 50),
            ])
            
            # Pattern features
            if len(window) > 1:
                diffs = np.diff(window)
                feature_row.extend([
                    np.mean(diffs),
                    np.std(diffs),
                    np.sum(diffs > 0),
                ])
            else:
                feature_row.extend([0, 0, 0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the voting ensemble."""
        try:
            logger.info("Training Voting Ensemble...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 24:
                raise ValueError("Insufficient data for ensemble training")
            
            # Create features and targets
            features = self._create_features(all_data)
            targets = all_data[12:]
            
            # Encode targets
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Create base models
            self.base_models = self._create_base_models()
            
            # Create voting classifier
            self.ensemble = VotingClassifier(
                estimators=self.base_models,
                voting=self.voting,
                weights=self.weights
            )
            
            # Train ensemble
            self.ensemble.fit(features, targets_encoded)
            
            self.is_fitted = True
            self.metadata.training_samples = len(features)
            
            # Calculate accuracy
            train_acc = self.ensemble.score(features, targets_encoded)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"Voting Ensemble training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Voting Ensemble training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction."""
        if not self.is_fitted or self.ensemble is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.array([np.random.randint(0, 100)])
            
            last_features = features[-1:].reshape(1, -1)
            prediction_encoded = self.ensemble.predict(last_features)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Voting Ensemble prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction probabilities."""
        if not self.is_fitted or self.ensemble is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            
            if self.voting == 'soft':
                probabilities = self.ensemble.predict_proba(last_features)
            else:
                # For hard voting, simulate probabilities
                prediction = self.ensemble.predict(last_features)
                probabilities = np.zeros((1, len(self.label_encoder.classes_)))
                probabilities[0, prediction[0]] = 1.0
            
            # Map to full range
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99 and i < probabilities.shape[1]:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"Voting Ensemble probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class StackingEnsembleModel(BasePredictionModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, cv_folds: int = 3):
        super().__init__("stacking_ensemble", "1.0")
        self.cv_folds = cv_folds
        self.ensemble = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self.metadata.description = "Stacking ensemble with meta-learner"
        self.metadata.parameters = {
            'cv_folds': cv_folds
        }
    
    def _create_base_models(self):
        """Create base models for stacking."""
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        
        return [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=50, max_depth=6, random_state=42)),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(max_depth=8, random_state=42)),
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the stacking ensemble."""
        try:
            logger.info("Training Stacking Ensemble...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 30:
                raise ValueError("Insufficient data for stacking ensemble training")
            
            # Create features
            features = self._create_features(all_data, lookback=10)
            targets = all_data[10:]
            
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Create base models
            base_models = self._create_base_models()
            
            # Create stacking classifier with logistic regression meta-learner
            self.ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(random_state=42, max_iter=2000),
                cv=self.cv_folds,
                stack_method='predict_proba'
            )
            
            # Train ensemble
            self.ensemble.fit(features, targets_encoded)
            
            self.is_fitted = True
            self.metadata.training_samples = len(features)
            
            train_acc = self.ensemble.score(features, targets_encoded)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"Stacking Ensemble training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Stacking Ensemble training failed: {e}")
            raise
    
    def _create_features(self, data: np.ndarray, lookback: int = 10) -> np.ndarray:
        """Create features for stacking models."""
        features = []
        
        for i in range(lookback, len(data)):
            feature_row = []
            
            window = data[i-lookback:i]
            feature_row.extend(window)
            
            # Basic statistics
            feature_row.extend([
                np.mean(window),
                np.std(window),
                np.median(window),
                np.sum(window % 2),
                np.sum(window >= 50),
            ])
            
            features.append(feature_row)
        
        return np.array(features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacking ensemble prediction."""
        if not self.is_fitted or self.ensemble is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X, lookback=10)
            
            if len(features) == 0:
                return np.array([np.random.randint(0, 100)])
            
            last_features = features[-1:].reshape(1, -1)
            prediction_encoded = self.ensemble.predict(last_features)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Stacking Ensemble prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate stacking ensemble probabilities."""
        if not self.is_fitted or self.ensemble is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X, lookback=10)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            probabilities = self.ensemble.predict_proba(last_features)
            
            # Map to full range
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99 and i < probabilities.shape[1]:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"Stacking Ensemble probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class BlendingEnsembleModel(BasePredictionModel):
    """Blending ensemble with holdout validation."""
    
    def __init__(self, holdout_ratio: float = 0.2):
        super().__init__("blending_ensemble", "1.0")
        self.holdout_ratio = holdout_ratio
        self.base_models = []
        self.meta_model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self.metadata.description = "Blending ensemble with holdout validation"
        self.metadata.parameters = {
            'holdout_ratio': holdout_ratio
        }
    
    def _create_base_models(self):
        """Create base models for blending."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        
        return [
            RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42),
            GaussianNB(),
            LogisticRegression(random_state=42, max_iter=2000)
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the blending ensemble."""
        try:
            logger.info("Training Blending Ensemble...")
            
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 25:
                raise ValueError("Insufficient data for blending ensemble training")
            
            features = self._create_features(all_data, lookback=8)
            targets = all_data[8:]
            targets_encoded = self.label_encoder.fit_transform(targets)
            
            # Split into train and holdout
            split_idx = int(len(features) * (1 - self.holdout_ratio))
            
            X_train = features[:split_idx]
            y_train = targets_encoded[:split_idx]
            X_holdout = features[split_idx:]
            y_holdout = targets_encoded[split_idx:]
            
            # Train base models
            self.base_models = self._create_base_models()
            
            for model in self.base_models:
                model.fit(X_train, y_train)
            
            # Generate holdout predictions
            holdout_predictions = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_holdout)
                    holdout_predictions.append(pred_proba)
                else:
                    pred = model.predict(X_holdout)
                    # Convert to one-hot
                    pred_proba = np.zeros((len(pred), len(np.unique(y_train))))
                    for i, p in enumerate(pred):
                        if p < len(pred_proba[0]):
                            pred_proba[i, p] = 1.0
                    holdout_predictions.append(pred_proba)
            
            # Create meta-features
            meta_features = np.hstack(holdout_predictions)
            
            # Train meta-model
            self.meta_model = LogisticRegression(random_state=42, max_iter=2000)
            self.meta_model.fit(meta_features, y_holdout)
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            logger.info("Blending Ensemble training completed")
            
        except Exception as e:
            logger.error(f"Blending Ensemble training failed: {e}")
            raise
    
    def _create_features(self, data: np.ndarray, lookback: int = 8) -> np.ndarray:
        """Create features for blending models."""
        features = []
        
        for i in range(lookback, len(data)):
            window = data[i-lookback:i]
            feature_row = list(window)
            feature_row.extend([
                np.mean(window),
                np.std(window),
                np.sum(window % 2),
            ])
            features.append(feature_row)
        
        return np.array(features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate blending ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X, lookback=8)
            
            if len(features) == 0:
                return np.array([np.random.randint(0, 100)])
            
            last_features = features[-1:].reshape(1, -1)
            
            # Get base model predictions
            base_predictions = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(last_features)
                    base_predictions.append(pred_proba)
                else:
                    pred = model.predict(last_features)
                    pred_proba = np.zeros((1, len(np.unique(self.label_encoder.classes_))))
                    if pred[0] < len(pred_proba[0]):
                        pred_proba[0, pred[0]] = 1.0
                    base_predictions.append(pred_proba)
            
            # Create meta-features
            meta_features = np.hstack(base_predictions)
            
            # Meta-model prediction
            prediction_encoded = self.meta_model.predict(meta_features)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Blending Ensemble prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate blending ensemble probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            features = self._create_features(X, lookback=8)
            
            if len(features) == 0:
                return np.ones((1, 100)) / 100
            
            last_features = features[-1:].reshape(1, -1)
            
            # Get base predictions
            base_predictions = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(last_features)
                    base_predictions.append(pred_proba)
                else:
                    pred = model.predict(last_features)
                    pred_proba = np.zeros((1, len(np.unique(self.label_encoder.classes_))))
                    if pred[0] < len(pred_proba[0]):
                        pred_proba[0, pred[0]] = 1.0
                    base_predictions.append(pred_proba)
            
            meta_features = np.hstack(base_predictions)
            probabilities = self.meta_model.predict_proba(meta_features)
            
            # Map to full range
            full_probs = np.zeros((1, 100))
            classes = self.label_encoder.classes_
            
            for i, class_label in enumerate(classes):
                if 0 <= class_label <= 99:
                    full_probs[0, int(class_label)] = probabilities[0, i]
            
            full_probs /= np.sum(full_probs)
            return full_probs
            
        except Exception as e:
            logger.error(f"Blending Ensemble probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


# Register ensemble models
def create_voting_ensemble(**kwargs):
    return VotingEnsembleModel(**kwargs)

def create_stacking_ensemble(**kwargs):
    return StackingEnsembleModel(**kwargs)

def create_blending_ensemble(**kwargs):
    return BlendingEnsembleModel(**kwargs)

model_registry.register_model(VotingEnsembleModel, create_voting_ensemble)
model_registry.register_model(StackingEnsembleModel, create_stacking_ensemble)
model_registry.register_model(BlendingEnsembleModel, create_blending_ensemble)