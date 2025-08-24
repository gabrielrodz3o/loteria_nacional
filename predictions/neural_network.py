"""Neural network models for lottery prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class NeuralNetworkModel(BasePredictionModel):
    """Deep neural network for lottery number prediction."""
    
    def __init__(self, sequence_length: int = 30, hidden_units: List[int] = None, 
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        super().__init__("neural_network", "1.0")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self.metadata.description = "Deep neural network with LSTM and Dense layers"
        self.metadata.parameters = {
            'sequence_length': sequence_length,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
    
    def _build_model(self, input_shape: Tuple[int, ...], output_classes: int) -> keras.Model:
        """Build the neural network architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # LSTM layers for sequence processing
            layers.LSTM(self.hidden_units[0], return_sequences=True, dropout=self.dropout_rate),
            layers.LSTM(self.hidden_units[1], return_sequences=False, dropout=self.dropout_rate),
            
            # Dense layers
            layers.Dense(self.hidden_units[2], activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.hidden_units[2] // 2, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(output_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _prepare_sequences_xy(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for training with separate X and y."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(X_data)):
            X.append(X_data[i-self.sequence_length:i])
            y.append(y_data[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the neural network model."""
        try:
            logger.info("Training neural network model...")
            
            # Validate input data
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty input data provided")
            
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
            
            # Clean input data
            valid_mask = ~pd.isna(X) & ~pd.isna(y) & (X >= 0) & (y >= 0)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) == 0:
                raise ValueError("No valid data after cleaning")
            
            logger.info(f"Using {len(X_clean)} valid samples out of {len(X)} total")
            
            # Prepare data
            if len(X_clean.shape) == 1:
                X_clean = X_clean.reshape(-1, 1)
            
            # Scale features only (not labels)
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # Validate and encode labels directly from original y values
            y_clean_int = y_clean.astype(int)
            unique_labels = np.unique(y_clean_int)
            
            logger.info(f"Unique labels in data: {unique_labels}")
            
            # Ensure labels are in valid range
            if np.any(unique_labels < 0):
                raise ValueError(f"Invalid negative labels found: {unique_labels[unique_labels < 0]}")
            
            # Encode labels - ensure they start from 0
            y_encoded = self.label_encoder.fit_transform(y_clean_int.ravel())
            
            # Validate encoded labels
            if np.any(y_encoded < 0):
                raise ValueError(f"LabelEncoder produced invalid values: {y_encoded[y_encoded < 0]}")
            
            logger.info(f"Encoded labels range: {y_encoded.min()} to {y_encoded.max()}")
            
            # Create sequences using scaled X and encoded y separately
            X_seq, y_seq = self._prepare_sequences_xy(X_scaled.ravel(), y_encoded)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Validate sequence labels
            if np.any(y_seq < 0):
                raise ValueError(f"Invalid sequence labels found: {y_seq[y_seq < 0]}")
            
            # Reshape for LSTM input
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )
            
            # Final validation of training labels
            if np.any(y_train < 0) or np.any(y_val < 0):
                raise ValueError("Invalid labels in training/validation split")
            
            # Build model
            output_classes = len(np.unique(y_encoded))
            logger.info(f"Building model with {output_classes} output classes")
            self.model = self._build_model(X_train.shape[1:], output_classes)
            
            # Train model
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_train)
            
            # Calculate training accuracy
            train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
            self.metadata.accuracy_score = train_acc
            
            logger.info(f"Neural network training completed. Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Scale input
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            X_scaled = self.scaler.transform(X)
            
            # Create sequences for prediction
            if len(X_scaled) >= self.sequence_length:
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((self.sequence_length - len(X_scaled), 1))
                X_padded = np.vstack([padding, X_scaled])
                X_seq = X_padded.reshape(1, self.sequence_length, 1)
            
            # Get prediction probabilities
            probabilities = self.model.predict(X_seq, verbose=0)
            
            # Get class with highest probability
            predicted_class = np.argmax(probabilities, axis=1)
            
            # Transform back to original space
            predictions = self.label_encoder.inverse_transform(predicted_class)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Scale input
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            X_scaled = self.scaler.transform(X)
            
            # Create sequences for prediction
            if len(X_scaled) >= self.sequence_length:
                X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            else:
                # Pad with zeros if not enough data
                padding = np.zeros((self.sequence_length - len(X_scaled), 1))
                X_padded = np.vstack([padding, X_scaled])
                X_seq = X_padded.reshape(1, self.sequence_length, 1)
            
            # Get prediction probabilities
            probabilities = self.model.predict(X_seq, verbose=0)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Neural network probability prediction failed: {e}")
            return np.array([[]])
    
    def get_confidence_score(self, prediction: Any) -> float:
        """Calculate confidence score for prediction."""
        try:
            if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                # For probability arrays, use max probability as confidence
                if len(prediction.shape) == 2:
                    return float(np.max(prediction))
                else:
                    return 0.7  # Default high confidence for NN
            return 0.7
        except:
            return 0.7


class LSTMModel(BasePredictionModel):
    """LSTM-focused model for time series prediction."""
    
    def __init__(self, sequence_length: int = 20, lstm_units: int = 100, 
                 num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__("lstm_model", "1.0")
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.metadata.description = "Multi-layer LSTM for sequential pattern recognition"
        self.metadata.parameters = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate
        }
    
    def _build_model(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Build LSTM model architecture."""
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        # Add LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(layers.LSTM(
                self.lstm_units, 
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Output layers
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(100, activation='softmax'))  # 100 possible numbers (0-99)
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model."""
        try:
            logger.info("Training LSTM model...")
            
            # Prepare sequential data
            sequences = []
            targets = []
            
            for i in range(self.sequence_length, len(X)):
                sequences.append(X[i-self.sequence_length:i])
                targets.append(y[i])
            
            X_seq = np.array(sequences)
            y_seq = np.array(targets)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data for LSTM training")
            
            # Reshape for LSTM
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
            
            # Build and train model
            self.model = self._build_model(X_seq.shape[1:])
            
            history = self.model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=1
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_seq)
            
            logger.info("LSTM model training completed")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate LSTM predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Prepare last sequence
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            else:
                # Pad with mean if not enough data
                mean_val = np.mean(X) if len(X) > 0 else 50
                padding = np.full((self.sequence_length - len(X),), mean_val)
                X_padded = np.concatenate([padding, X])
                X_seq = X_padded.reshape(1, self.sequence_length, 1)
            
            # Predict
            probabilities = self.model.predict(X_seq, verbose=0)
            predicted_class = np.argmax(probabilities, axis=1)
            
            return predicted_class
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate LSTM prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Prepare last sequence
            if len(X) >= self.sequence_length:
                X_seq = X[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            else:
                mean_val = np.mean(X) if len(X) > 0 else 50
                padding = np.full((self.sequence_length - len(X),), mean_val)
                X_padded = np.concatenate([padding, X])
                X_seq = X_padded.reshape(1, self.sequence_length, 1)
            
            # Get probabilities
            probabilities = self.model.predict(X_seq, verbose=0)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"LSTM probability prediction failed: {e}")
            return np.array([[]])


# Register models
def create_neural_network(**kwargs):
    return NeuralNetworkModel(**kwargs)

def create_lstm_model(**kwargs):
    return LSTMModel(**kwargs)

model_registry.register_model(NeuralNetworkModel, create_neural_network)
model_registry.register_model(LSTMModel, create_lstm_model)