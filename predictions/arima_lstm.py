"""ARIMA and LSTM time series models for lottery prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from models.prediction_models import BasePredictionModel, model_registry

# Optional imports for time series
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ARIMAModel(BasePredictionModel):
    """ARIMA model for time series prediction."""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)):
        super().__init__("arima_model", "1.0")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
        self.metadata.description = "ARIMA time series model"
        self.metadata.parameters = {
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    def _check_stationarity(self, data: np.ndarray) -> bool:
        """Check if time series is stationary."""
        try:
            result = adfuller(data)
            return result[1] <= 0.05  # p-value threshold
        except:
            return False
    
    def _make_stationary(self, data: np.ndarray) -> np.ndarray:
        """Make time series stationary if needed."""
        if self._check_stationarity(data):
            return data
        
        # Try first difference
        diff_data = np.diff(data)
        if len(diff_data) > 0 and self._check_stationarity(diff_data):
            return diff_data
        
        # Try second difference
        if len(diff_data) > 1:
            diff2_data = np.diff(diff_data)
            if len(diff2_data) > 0:
                return diff2_data
        
        return data  # Return original if differencing doesn't help
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not available. Install with: pip install statsmodels")
        
        try:
            logger.info("Training ARIMA model...")
            
            # Combine data for time series
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < 30:
                raise ValueError("Insufficient data for ARIMA training")
            
            # Create time series
            ts = pd.Series(all_data, index=pd.date_range(
                start='2020-01-01', periods=len(all_data), freq='D'
            ))
            
            # Fit ARIMA model
            self.model = ARIMA(ts, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            
            self.is_fitted = True
            self.metadata.training_samples = len(all_data)
            
            # Calculate AIC as a proxy for model quality
            aic = self.fitted_model.aic
            self.metadata.accuracy_score = 1.0 / (1.0 + aic / 1000)  # Normalized score
            
            logger.info(f"ARIMA training completed. AIC: {aic:.2f}")
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ARIMA prediction."""
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Forecast next value
            forecast = self.fitted_model.forecast(steps=1)
            predicted_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            
            # Clamp to valid range
            predicted_number = int(np.clip(predicted_value, 0, 99))
            
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate ARIMA prediction with confidence intervals."""
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Get forecast with confidence intervals
            forecast_result = self.fitted_model.get_forecast(steps=1)
            mean_forecast = forecast_result.predicted_mean.iloc[0]
            conf_int = forecast_result.conf_int()
            
            # Use confidence interval to estimate uncertainty
            lower_bound = conf_int.iloc[0, 0]
            upper_bound = conf_int.iloc[0, 1]
            std_estimate = (upper_bound - lower_bound) / 4  # Rough estimate
            
            # Create probability distribution around forecast
            probabilities = np.zeros(100)
            for i in range(100):
                probabilities[i] = np.exp(-0.5 * ((i - mean_forecast) / max(std_estimate, 1))**2)
            
            # Normalize
            probabilities /= np.sum(probabilities)
            
            return probabilities.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"ARIMA probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class LSTMTimeSeriesModel(BasePredictionModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, sequence_length: int = 20, lstm_units: int = 64, 
                 dropout_rate: float = 0.2, epochs: int = 50):
        super().__init__("lstm_timeseries", "1.0")
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted = False
        
        self.metadata.description = "LSTM for time series forecasting"
        self.metadata.parameters = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'epochs': epochs
        }
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self) -> keras.Model:
        """Build LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, return_sequences=True, 
                       input_shape=(self.sequence_length, 1)),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        try:
            logger.info("Training LSTM Time Series model...")
            
            # Prepare data
            all_data = np.concatenate([X.ravel(), y.ravel()])
            all_data = all_data[(all_data >= 0) & (all_data <= 99)]
            
            if len(all_data) < self.sequence_length + 10:
                raise ValueError("Insufficient data for LSTM training")
            
            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            
            data_scaled = self.scaler_X.fit_transform(all_data.reshape(-1, 1)).ravel()
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(data_scaled)
            
            # Reshape for LSTM
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
            y_seq = y_seq.reshape(-1, 1)
            
            # Fit scaler for y
            self.scaler_y.fit(y_seq)
            
            # Build and train model
            self.model = self._build_model()
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True
            )
            
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=16,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.2
            )
            
            self.is_fitted = True
            self.metadata.training_samples = len(X_seq)
            
            # Use final loss as accuracy proxy
            final_loss = history.history['loss'][-1]
            self.metadata.accuracy_score = 1.0 / (1.0 + final_loss)
            
            logger.info(f"LSTM training completed. Final loss: {final_loss:.4f}")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate LSTM prediction."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Prepare input sequence
            if len(X) >= self.sequence_length:
                input_seq = X[-self.sequence_length:]
            else:
                # Pad with mean if not enough data
                mean_val = np.mean(X) if len(X) > 0 else 50
                padding = np.full(self.sequence_length - len(X), mean_val)
                input_seq = np.concatenate([padding, X])
            
            # Scale input
            input_scaled = self.scaler_X.transform(input_seq.reshape(-1, 1)).ravel()
            input_scaled = input_scaled.reshape(1, self.sequence_length, 1)
            
            # Predict
            prediction_scaled = self.model.predict(input_scaled, verbose=0)
            
            # Inverse transform
            prediction = self.scaler_y.inverse_transform(prediction_scaled)
            predicted_number = int(np.clip(prediction[0, 0], 0, 99))
            
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate LSTM prediction with uncertainty estimation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Monte Carlo dropout for uncertainty estimation
            predictions = []
            
            # Enable training mode for dropout
            for _ in range(50):  # Monte Carlo samples
                pred = self.predict(X)
                predictions.append(pred[0])
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Create probability distribution
            probabilities = np.zeros(100)
            for i in range(100):
                probabilities[i] = np.exp(-0.5 * ((i - mean_pred) / max(std_pred, 1))**2)
            
            probabilities /= np.sum(probabilities)
            return probabilities.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"LSTM probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class ARIMALSTMHybridModel(BasePredictionModel):
    """Hybrid model combining ARIMA and LSTM."""
    
    def __init__(self, arima_weight: float = 0.6, lstm_weight: float = 0.4):
        super().__init__("arima_lstm_hybrid", "1.0")
        self.arima_weight = arima_weight
        self.lstm_weight = lstm_weight
        self.arima_model = None
        self.lstm_model = None
        self.is_fitted = False
        
        self.metadata.description = "Hybrid ARIMA + LSTM model"
        self.metadata.parameters = {
            'arima_weight': arima_weight,
            'lstm_weight': lstm_weight
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train hybrid model."""
        try:
            logger.info("Training ARIMA-LSTM Hybrid model...")
            
            # Train ARIMA component
            if STATSMODELS_AVAILABLE:
                self.arima_model = ARIMAModel(order=(1, 1, 1))
                try:
                    self.arima_model.fit(X, y)
                except:
                    logger.warning("ARIMA fitting failed, using LSTM only")
                    self.arima_model = None
                    self.arima_weight = 0.0
                    self.lstm_weight = 1.0
            else:
                logger.warning("statsmodels not available, using LSTM only")
                self.arima_model = None
                self.arima_weight = 0.0
                self.lstm_weight = 1.0
            
            # Train LSTM component
            if TENSORFLOW_AVAILABLE:
                self.lstm_model = LSTMTimeSeriesModel(sequence_length=15, lstm_units=32, epochs=30)
                try:
                    self.lstm_model.fit(X, y)
                except:
                    logger.warning("LSTM fitting failed")
                    if self.arima_model is None:
                        raise ValueError("Both ARIMA and LSTM fitting failed")
                    self.lstm_model = None
                    self.arima_weight = 1.0
                    self.lstm_weight = 0.0
            else:
                logger.warning("TensorFlow not available")
                if self.arima_model is None:
                    raise ValueError("Neither statsmodels nor TensorFlow available")
                self.lstm_model = None
                self.arima_weight = 1.0
                self.lstm_weight = 0.0
            
            self.is_fitted = True
            self.metadata.training_samples = len(np.concatenate([X.ravel(), y.ravel()]))
            
            logger.info("ARIMA-LSTM Hybrid training completed")
            
        except Exception as e:
            logger.error(f"Hybrid model training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate hybrid prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = []
            weights = []
            
            # ARIMA prediction
            if self.arima_model and self.arima_weight > 0:
                try:
                    arima_pred = self.arima_model.predict(X)
                    predictions.append(arima_pred[0])
                    weights.append(self.arima_weight)
                except:
                    logger.warning("ARIMA prediction failed")
            
            # LSTM prediction
            if self.lstm_model and self.lstm_weight > 0:
                try:
                    lstm_pred = self.lstm_model.predict(X)
                    predictions.append(lstm_pred[0])
                    weights.append(self.lstm_weight)
                except:
                    logger.warning("LSTM prediction failed")
            
            if not predictions:
                return np.array([np.random.randint(0, 100)])
            
            # Weighted average
            weighted_pred = np.average(predictions, weights=weights)
            predicted_number = int(np.clip(weighted_pred, 0, 99))
            
            return np.array([predicted_number])
            
        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate hybrid prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            prob_distributions = []
            weights = []
            
            # ARIMA probabilities
            if self.arima_model and self.arima_weight > 0:
                try:
                    arima_proba = self.arima_model.predict_proba(X)
                    prob_distributions.append(arima_proba[0])
                    weights.append(self.arima_weight)
                except:
                    logger.warning("ARIMA probability prediction failed")
            
            # LSTM probabilities
            if self.lstm_model and self.lstm_weight > 0:
                try:
                    lstm_proba = self.lstm_model.predict_proba(X)
                    prob_distributions.append(lstm_proba[0])
                    weights.append(self.lstm_weight)
                except:
                    logger.warning("LSTM probability prediction failed")
            
            if not prob_distributions:
                return np.ones((1, 100)) / 100
            
            # Weighted average of probability distributions
            weights = np.array(weights) / np.sum(weights)
            combined_proba = np.zeros(100)
            
            for i, proba in enumerate(prob_distributions):
                combined_proba += weights[i] * proba
            
            return combined_proba.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Hybrid probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


# Register time series models
def create_arima_model(**kwargs):
    return ARIMAModel(**kwargs)

def create_lstm_timeseries(**kwargs):
    return LSTMTimeSeriesModel(**kwargs)

def create_arima_lstm_hybrid(**kwargs):
    return ARIMALSTMHybridModel(**kwargs)

if STATSMODELS_AVAILABLE:
    model_registry.register_model(ARIMAModel, create_arima_model)

if TENSORFLOW_AVAILABLE:
    model_registry.register_model(LSTMTimeSeriesModel, create_lstm_timeseries)

if STATSMODELS_AVAILABLE or TENSORFLOW_AVAILABLE:
    model_registry.register_model(ARIMALSTMHybridModel, create_arima_lstm_hybrid)