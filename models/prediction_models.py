"""Prediction models registry and metadata management - FIXED VERSION."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for prediction models."""
    name: str
    version: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    accuracy_score: Optional[float] = None
    last_trained: Optional[datetime] = None
    training_samples: int = 0
    is_active: bool = True


class BasePredictionModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str, version: str = "1.0"):
        # CORRECCIÓN: Asegurar que todos los modelos tengan el nombre correcto
        self.name = name
        self.model_name = name  # Alias para compatibilidad con predictor_engine
        self.version = version
        self.metadata = ModelMetadata(name=name, version=version, description="")
        self.is_trained = False
        
        logger.info(f"[MODEL-INIT] Initialized base model: {name} v{version}")
        
    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Train the model with provided data."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Generate prediction probabilities."""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported."""
        return None
    
    def get_confidence_score(self, prediction: Any) -> float:
        """Calculate confidence score for prediction."""
        try:
            if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                # For probability arrays, use max probability as confidence
                if len(prediction.shape) == 2:
                    return float(np.max(prediction))
                elif len(prediction.shape) == 1:
                    return float(np.max(prediction))
                else:
                    return 0.7
            return 0.7  # Default high confidence
        except:
            return 0.7


class ModelRegistry:
    """Registry for managing prediction models."""
    
    def __init__(self):
        self._models: Dict[str, BasePredictionModel] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._factory_functions: Dict[str, Callable] = {}
        logger.info("[REGISTRY] Model registry initialized")
        
    def register_model(self, model_class: type, factory_function: Callable = None):
        """Register a model class with optional factory function."""
        # CORRECCIÓN: Mejor extracción del nombre del modelo
        model_name = model_class.__name__.lower().replace('model', '')
        
        # Use factory function if provided, otherwise use class constructor
        factory = factory_function or model_class
        self._factory_functions[model_name] = factory
        
        # CORRECCIÓN: Crear instancia temporal para obtener metadata
        try:
            temp_instance = factory()
            if hasattr(temp_instance, 'metadata'):
                self._metadata[model_name] = temp_instance.metadata
            else:
                # Create default metadata
                self._metadata[model_name] = ModelMetadata(
                    name=model_name,
                    version="1.0",
                    description=f"Model class: {model_class.__name__}"
                )
        except Exception as e:
            logger.warning(f"[REGISTRY] Could not create temp instance for {model_name}: {e}")
            # Create minimal metadata
            self._metadata[model_name] = ModelMetadata(
                name=model_name,
                version="1.0",
                description=f"Model class: {model_class.__name__}"
            )
        
        logger.info(f"[REGISTRY] Registered model: {model_name}")
    
    def create_model(self, model_name: str, **kwargs) -> Optional[BasePredictionModel]:
        """Create model instance by name."""
        if model_name not in self._factory_functions:
            logger.error(f"[REGISTRY] Model '{model_name}' not registered. Available: {list(self._factory_functions.keys())}")
            return None
        
        try:
            model = self._factory_functions[model_name](**kwargs)
            
            # CORRECCIÓN: Asegurar que el modelo tenga los atributos necesarios
            if not hasattr(model, 'name'):
                model.name = model_name
            if not hasattr(model, 'model_name'):
                model.model_name = model_name
            
            # Update metadata with actual name
            if hasattr(model, 'metadata'):
                model.metadata.name = model_name
            
            self._models[model_name] = model
            logger.info(f"[REGISTRY] Created model instance: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to create model '{model_name}': {e}")
            return None
    
    def get_model(self, model_name: str) -> Optional[BasePredictionModel]:
        """Get existing model instance."""
        return self._models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._factory_functions.keys())
    
    def update_metadata(self, model_name: str, metadata: ModelMetadata):
        """Update model metadata."""
        self._metadata[model_name] = metadata
        if model_name in self._models:
            self._models[model_name].metadata = metadata
    
    def get_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._metadata.get(model_name)
    
    def get_active_models(self) -> List[str]:
        """Get list of active model names."""
        return [name for name, meta in self._metadata.items() if meta.is_active]


@dataclass
class PredictionResult:
    """Structure for prediction results."""
    posicion: int
    numeros: List[int]
    probabilidad: float
    metodo_generacion: str  # CORRECCIÓN: Campo clave para tracking del método
    score_confianza: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'posicion': self.posicion,
            'probabilidad': self.probabilidad,
            'metodo_generacion': self.metodo_generacion,  # CORRECCIÓN: Asegurar que se preserve
            'score_confianza': self.score_confianza
        }
        
        # Handle different number formats
        if len(self.numeros) == 1:
            result['numero'] = self.numeros[0]
        else:
            result['numeros'] = self.numeros
            
        return result


@dataclass
class GamePredictions:
    """Container for game-specific predictions."""
    quiniela: List[PredictionResult] = field(default_factory=list)
    pale: List[PredictionResult] = field(default_factory=list)
    tripleta: List[PredictionResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to dictionary format."""
        return {
            'quiniela': [pred.to_dict() for pred in self.quiniela],
            'pale': [pred.to_dict() for pred in self.pale],
            'tripleta': [pred.to_dict() for pred in self.tripleta]
        }


class ModelEvaluator:
    """Evaluates model performance and selects best models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.evaluation_history: Dict[str, List[float]] = {}
        logger.info("[EVALUATOR] Model evaluator initialized")
    
    def evaluate_model(self, model_name: str, predictions: List[Any], 
                      actual_results: List[Any]) -> Dict[str, float]:
        """Evaluate a model's performance."""
        try:
            accuracy = self._calculate_accuracy(predictions, actual_results)
            precision = self._calculate_precision(predictions, actual_results)
            recall = self._calculate_recall(predictions, actual_results)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
            # Update evaluation history
            if model_name not in self.evaluation_history:
                self.evaluation_history[model_name] = []
            self.evaluation_history[model_name].append(accuracy)
            
            # Keep only last 100 evaluations per model
            if len(self.evaluation_history[model_name]) > 100:
                self.evaluation_history[model_name] = self.evaluation_history[model_name][-100:]
            
            logger.info(f"[EVALUATOR] Evaluated {model_name}: accuracy={accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"[EVALUATOR] Model evaluation failed for {model_name}: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def _calculate_accuracy(self, predictions: List[Any], actual: List[Any]) -> float:
        """Calculate prediction accuracy."""
        if not predictions or not actual or len(predictions) != len(actual):
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actual) if p == a)
        return correct / len(predictions)
    
    def _calculate_precision(self, predictions: List[Any], actual: List[Any]) -> float:
        """Calculate precision score."""
        # Simplified precision calculation
        return self._calculate_accuracy(predictions, actual)
    
    def _calculate_recall(self, predictions: List[Any], actual: List[Any]) -> float:
        """Calculate recall score."""
        # Simplified recall calculation
        return self._calculate_accuracy(predictions, actual)
    
    def get_best_models(self, top_k: int = 3) -> List[str]:
        """Get top performing models based on recent performance."""
        if not self.evaluation_history:
            return self.registry.get_active_models()[:top_k]
        
        # Calculate average performance over recent evaluations
        model_scores = {}
        for model_name, scores in self.evaluation_history.items():
            # Use last 10 evaluations or all if less than 10
            recent_scores = scores[-10:]
            model_scores[model_name] = sum(recent_scores) / len(recent_scores)
        
        # Sort by performance and return top k
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model_name for model_name, _ in sorted_models[:top_k]]
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance statistics for a model."""
        if model_name not in self.evaluation_history:
            return {'avg_accuracy': 0.0, 'recent_accuracy': 0.0, 'evaluations': 0}
        
        scores = self.evaluation_history[model_name]
        return {
            'avg_accuracy': sum(scores) / len(scores),
            'recent_accuracy': sum(scores[-5:]) / min(5, len(scores)),
            'evaluations': len(scores)
        }
    
    def record_performance(self, model_name: str, accuracy: float, 
                         predictions_made: int, correct_predictions: int,
                         metadata: Dict[str, Any] = None) -> None:
        """Record performance metrics for a model."""
        try:
            if model_name not in self.evaluation_history:
                self.evaluation_history[model_name] = []
            
            self.evaluation_history[model_name].append(accuracy)
            
            # Keep only last 100 records per model
            if len(self.evaluation_history[model_name]) > 100:
                self.evaluation_history[model_name] = self.evaluation_history[model_name][-100:]
            
            logger.info(f"[EVALUATOR] Recorded performance for {model_name}: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"[EVALUATOR] Failed to record performance for {model_name}: {e}")


# CORRECCIÓN: Importar numpy para el método get_confidence_score
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available, confidence scores will use defaults")
    np = None

# Global registry instance
model_registry = ModelRegistry()
model_evaluator = ModelEvaluator(model_registry)