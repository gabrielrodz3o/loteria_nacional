"""Monte Carlo simulation models for lottery prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
from scipy.stats import entropy
import logging
from models.prediction_models import BasePredictionModel, model_registry

logger = logging.getLogger(__name__)


class MonteCarloModel(BasePredictionModel):
    """Basic Monte Carlo simulation for lottery prediction."""
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = None):
        super().__init__("monte_carlo", "1.0")
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.historical_frequencies = None
        self.transition_matrix = None
        self.is_fitted = False
        
        if random_seed:
            np.random.seed(random_seed)
        
        self.metadata.description = "Monte Carlo simulation with frequency analysis"
        self.metadata.parameters = {
            'n_simulations': n_simulations,
            'random_seed': random_seed
        }
    
    def _calculate_frequencies(self, data: np.ndarray) -> Dict[int, float]:
        """Calculate historical frequencies of numbers."""
        unique, counts = np.unique(data, return_counts=True)
        total = len(data)
        frequencies = {}
        
        # Initialize all numbers 0-99 with small probability
        for i in range(100):
            frequencies[i] = 1.0 / total  # Laplace smoothing
        
        # Update with actual frequencies
        for num, count in zip(unique, counts):
            if 0 <= num <= 99:
                frequencies[int(num)] = (count + 1) / (total + 100)  # Laplace smoothing
        
        # Normalize
        total_prob = sum(frequencies.values())
        for num in frequencies:
            frequencies[num] /= total_prob
        
        return frequencies
    
    def _build_transition_matrix(self, data: np.ndarray) -> np.ndarray:
        """Build Markov transition matrix."""
        matrix = np.ones((100, 100)) * 0.01  # Laplace smoothing
        
        for i in range(len(data) - 1):
            current = int(data[i]) if 0 <= data[i] <= 99 else 50
            next_num = int(data[i + 1]) if 0 <= data[i + 1] <= 99 else 50
            matrix[current][next_num] += 1
        
        # Normalize rows
        for i in range(100):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                matrix[i] /= row_sum
        
        return matrix
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Monte Carlo model with historical data."""
        try:
            logger.info("Fitting Monte Carlo model...")
            
            # Combine X and y for frequency analysis
            all_data = np.concatenate([X.ravel(), y.ravel()])
            
            # Calculate frequencies
            self.historical_frequencies = self._calculate_frequencies(all_data)
            
            # Build transition matrix
            self.transition_matrix = self._build_transition_matrix(all_data)
            
            self.is_fitted = True
            self.metadata.training_samples = len(all_data)
            
            logger.info(f"Monte Carlo model fitted with {len(all_data)} data points")
            
        except Exception as e:
            logger.error(f"Monte Carlo fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using Monte Carlo simulation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Run multiple simulations
            simulation_results = []
            
            for _ in range(self.n_simulations):
                # Use frequency-based sampling
                numbers = list(self.historical_frequencies.keys())
                probabilities = list(self.historical_frequencies.values())
                
                # Sample based on frequencies
                predicted_number = np.random.choice(numbers, p=probabilities)
                simulation_results.append(predicted_number)
            
            # Return most frequent result
            unique, counts = np.unique(simulation_results, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            
            return np.array([most_frequent])
            
        except Exception as e:
            logger.error(f"Monte Carlo prediction failed: {e}")
            return np.array([np.random.randint(0, 100)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Run simulations and count occurrences
            simulation_results = []
            
            for _ in range(self.n_simulations):
                numbers = list(self.historical_frequencies.keys())
                probabilities = list(self.historical_frequencies.values())
                predicted_number = np.random.choice(numbers, p=probabilities)
                simulation_results.append(predicted_number)
            
            # Calculate probabilities from simulation results
            unique, counts = np.unique(simulation_results, return_counts=True)
            total_simulations = len(simulation_results)
            
            # Create probability array for all numbers 0-99
            prob_array = np.zeros(100)
            for num, count in zip(unique, counts):
                if 0 <= num <= 99:
                    prob_array[int(num)] = count / total_simulations
            
            return prob_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Monte Carlo probability prediction failed: {e}")
            # Return uniform distribution as fallback
            return np.ones((1, 100)) / 100
    
    def get_top_predictions(self, n_predictions: int = 3) -> List[Tuple[int, float]]:
        """Get top N predictions with probabilities."""
        if not self.is_fitted:
            return [(np.random.randint(0, 100), 0.01) for _ in range(n_predictions)]
        
        try:
            # Sort frequencies and return top N
            sorted_freqs = sorted(self.historical_frequencies.items(), 
                                key=lambda x: x[1], reverse=True)
            return sorted_freqs[:n_predictions]
            
        except Exception as e:
            logger.error(f"Failed to get top predictions: {e}")
            return [(np.random.randint(0, 100), 0.01) for _ in range(n_predictions)]


class QuasiMonteCarloModel(BasePredictionModel):
    """Quasi-Monte Carlo using low-discrepancy sequences."""
    
    def __init__(self, n_simulations: int = 5000, sequence_type: str = "sobol"):
        super().__init__("quasi_monte_carlo", "1.0")
        self.n_simulations = n_simulations
        self.sequence_type = sequence_type
        self.historical_data = None
        self.is_fitted = False
        
        self.metadata.description = "Quasi-Monte Carlo with low-discrepancy sequences"
        self.metadata.parameters = {
            'n_simulations': n_simulations,
            'sequence_type': sequence_type
        }
    
    def _generate_sobol_sequence(self, n_points: int, dimension: int = 1) -> np.ndarray:
        """Generate Sobol sequence (simplified implementation)."""
        # Simplified Sobol sequence - in production, use scipy.stats.qmc.Sobol
        sequence = []
        for i in range(n_points):
            # Van der Corput sequence in base 2
            value = 0
            base = 0.5
            n = i + 1
            while n > 0:
                if n % 2 == 1:
                    value += base
                base /= 2
                n //= 2
            sequence.append(value)
        
        return np.array(sequence)
    
    def _generate_halton_sequence(self, n_points: int, base: int = 2) -> np.ndarray:
        """Generate Halton sequence."""
        sequence = []
        for i in range(n_points):
            value = 0
            f = 1.0 / base
            n = i + 1
            while n > 0:
                value += f * (n % base)
                n //= base
                f /= base
            sequence.append(value)
        
        return np.array(sequence)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Quasi-Monte Carlo model."""
        try:
            logger.info("Fitting Quasi-Monte Carlo model...")
            
            # Store historical data for analysis
            self.historical_data = np.concatenate([X.ravel(), y.ravel()])
            
            # Calculate basic statistics
            self.mean = np.mean(self.historical_data)
            self.std = np.std(self.historical_data)
            
            self.is_fitted = True
            self.metadata.training_samples = len(self.historical_data)
            
            logger.info(f"Quasi-Monte Carlo model fitted")
            
        except Exception as e:
            logger.error(f"Quasi-Monte Carlo fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using quasi-random sequences."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate low-discrepancy sequence
            if self.sequence_type == "sobol":
                sequence = self._generate_sobol_sequence(self.n_simulations)
            else:  # halton
                sequence = self._generate_halton_sequence(self.n_simulations)
            
            # Transform sequence to number range [0, 99]
            # Use historical distribution characteristics
            predictions = []
            
            for value in sequence:
                # Transform uniform [0,1] to lottery number using inverse CDF
                # Simplified: use normal distribution around historical mean
                z_score = stats.norm.ppf(value)
                number = int(self.mean + z_score * self.std)
                
                # Clamp to valid range
                number = max(0, min(99, number))
                predictions.append(number)
            
            # Return most frequent prediction
            unique, counts = np.unique(predictions, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            
            return np.array([most_frequent])
            
        except Exception as e:
            logger.error(f"Quasi-Monte Carlo prediction failed: {e}")
            return np.array([int(self.mean) if hasattr(self, 'mean') else 50])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using quasi-random sampling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate sequence and transform to predictions
            if self.sequence_type == "sobol":
                sequence = self._generate_sobol_sequence(self.n_simulations)
            else:
                sequence = self._generate_halton_sequence(self.n_simulations)
            
            predictions = []
            for value in sequence:
                z_score = stats.norm.ppf(max(0.001, min(0.999, value)))
                number = int(self.mean + z_score * self.std)
                number = max(0, min(99, number))
                predictions.append(number)
            
            # Calculate probabilities
            unique, counts = np.unique(predictions, return_counts=True)
            prob_array = np.zeros(100)
            
            for num, count in zip(unique, counts):
                prob_array[num] = count / self.n_simulations
            
            return prob_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Quasi-Monte Carlo probability prediction failed: {e}")
            return np.ones((1, 100)) / 100


class MarkovChainMonteCarloModel(BasePredictionModel):
    """MCMC model using Markov chains for prediction."""
    
    def __init__(self, n_simulations: int = 8000, chain_length: int = 10):
        super().__init__("mcmc_model", "1.0")
        self.n_simulations = n_simulations
        self.chain_length = chain_length
        self.transition_matrix = None
        self.stationary_distribution = None
        self.is_fitted = False
        
        self.metadata.description = "Markov Chain Monte Carlo for sequential prediction"
        self.metadata.parameters = {
            'n_simulations': n_simulations,
            'chain_length': chain_length
        }
    
    def _build_transition_matrix(self, data: np.ndarray) -> np.ndarray:
        """Build transition matrix from historical data."""
        matrix = np.ones((100, 100)) * 0.001  # Small smoothing
        
        for i in range(len(data) - 1):
            current = int(data[i]) if 0 <= data[i] <= 99 else 50
            next_num = int(data[i + 1]) if 0 <= data[i + 1] <= 99 else 50
            matrix[current][next_num] += 1
        
        # Normalize rows
        for i in range(100):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                matrix[i] /= row_sum
        
        return matrix
    
    def _calculate_stationary_distribution(self) -> np.ndarray:
        """Calculate stationary distribution of the Markov chain."""
        try:
            # Find stationary distribution by solving π = πP
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            
            # Find eigenvector corresponding to eigenvalue 1
            stationary_index = np.argmin(np.abs(eigenvalues - 1))
            stationary = np.real(eigenvectors[:, stationary_index])
            
            # Normalize to probability distribution
            stationary = np.abs(stationary)
            stationary /= np.sum(stationary)
            
            return stationary
            
        except Exception as e:
            logger.warning(f"Failed to calculate stationary distribution: {e}")
            return np.ones(100) / 100  # Uniform fallback
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the MCMC model."""
        try:
            logger.info("Fitting MCMC model...")
            
            # Combine data for transition matrix
            all_data = np.concatenate([X.ravel(), y.ravel()])
            
            # Build transition matrix
            self.transition_matrix = self._build_transition_matrix(all_data)
            
            # Calculate stationary distribution
            self.stationary_distribution = self._calculate_stationary_distribution()
            
            self.is_fitted = True
            self.metadata.training_samples = len(all_data)
            
            logger.info("MCMC model fitted successfully")
            
        except Exception as e:
            logger.error(f"MCMC fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction using MCMC simulation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Start from last observed value or random state
            if len(X) > 0:
                current_state = int(X[-1]) if 0 <= X[-1] <= 99 else 50
            else:
                current_state = np.random.choice(100, p=self.stationary_distribution)
            
            # Run Markov chain simulation
            chain_results = []
            
            for _ in range(self.n_simulations):
                state = current_state
                
                # Run chain for specified length
                for _ in range(self.chain_length):
                    # Sample next state based on transition probabilities
                    next_state = np.random.choice(100, p=self.transition_matrix[state])
                    state = next_state
                
                chain_results.append(state)
            
            # Return most frequent final state
            unique, counts = np.unique(chain_results, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            
            return np.array([most_frequent])
            
        except Exception as e:
            logger.error(f"MCMC prediction failed: {e}")
            return np.array([50])  # Fallback to middle value
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities using MCMC."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Start from last observed value or random state
            if len(X) > 0:
                current_state = int(X[-1]) if 0 <= X[-1] <= 99 else 50
            else:
                current_state = np.random.choice(100, p=self.stationary_distribution)
            
            # Run simulations
            chain_results = []
            
            for _ in range(self.n_simulations):
                state = current_state
                
                for _ in range(self.chain_length):
                    next_state = np.random.choice(100, p=self.transition_matrix[state])
                    state = next_state
                
                chain_results.append(state)
            
            # Calculate probabilities from simulation results
            unique, counts = np.unique(chain_results, return_counts=True)
            prob_array = np.zeros(100)
            
            for num, count in zip(unique, counts):
                prob_array[num] = count / self.n_simulations
            
            return prob_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"MCMC probability prediction failed: {e}")
            return self.stationary_distribution.reshape(1, -1)


# Register Monte Carlo models
def create_monte_carlo(**kwargs):
    return MonteCarloModel(**kwargs)

def create_quasi_monte_carlo(**kwargs):
    return QuasiMonteCarloModel(**kwargs)

def create_mcmc_model(**kwargs):
    return MarkovChainMonteCarloModel(**kwargs)

model_registry.register_model(MonteCarloModel, create_monte_carlo)
model_registry.register_model(QuasiMonteCarloModel, create_quasi_monte_carlo)
model_registry.register_model(MarkovChainMonteCarloModel, create_mcmc_model)