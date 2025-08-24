"""Models package for lottery prediction system."""

from .database_models import (
    Base,
    TipoLoteria,
    TipoJuego,
    Sorteo,
    PrediccionQuiniela,
    PrediccionPale,
    PrediccionTripleta,
    MetodoPrediccion,
    Vector,
    ResultadoPrediccion
)

from .prediction_models import (
    BasePredictionModel,
    ModelRegistry,
    ModelMetadata,
    PredictionResult,
    GamePredictions,
    ModelEvaluator,
    model_registry,
    model_evaluator
)

__all__ = [
    # Database models
    'Base',
    'TipoLoteria',
    'TipoJuego', 
    'Sorteo',
    'PrediccionQuiniela',
    'PrediccionPale',
    'PrediccionTripleta',
    'MetodoPrediccion',
    'Vector',
    'ResultadoPrediccion',
    
    # Prediction models
    'BasePredictionModel',
    'ModelRegistry',
    'ModelMetadata',
    'PredictionResult',
    'GamePredictions',
    'ModelEvaluator',
    'model_registry',
    'model_evaluator'
]
from . import register_models  # Auto-registra los 6 modelos
from . import real_ml_models  # Auto-load real ML models
