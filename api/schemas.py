"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
from enum import Enum


class GameType(str, Enum):
    """Game type enumeration."""
    QUINIELA = "quiniela"
    PALE = "pale"
    TRIPLETA = "tripleta"


class MethodType(str, Enum):
    """Prediction method enumeration."""
    NEURAL_NETWORK = "neural_network"
    MONTE_CARLO = "monte_carlo"
    STATISTICAL = "statistical"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ENSEMBLE = "ensemble_ml"
    BAYESIAN = "bayesian_model"
    RANDOM_FOREST = "random_forest"
    ARIMA_LSTM = "arima_lstm"


# Prediction schemas
class QuinielaPrediction(BaseModel):
    """Schema for quiniela prediction."""
    posicion: int = Field(..., ge=1, le=3, description="Posición de la predicción (1-3)")
    numero: int = Field(..., ge=0, le=99, description="Número predicho (0-99)")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de acierto")
    metodo_generacion: str = Field(..., description="Método usado para generar la predicción")
    score_confianza: float = Field(..., ge=0.0, le=1.0, description="Score de confianza")


class PalePrediction(BaseModel):
    """Schema for pale prediction."""
    posicion: int = Field(..., ge=1, le=3, description="Posición de la predicción (1-3)")
    numeros: List[int] = Field(..., min_items=2, max_items=2, description="Par de números")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de acierto")
    metodo_generacion: str = Field(..., description="Método usado para generar la predicción")
    score_confianza: float = Field(..., ge=0.0, le=1.0, description="Score de confianza")
    
    @validator('numeros')
    def validate_numeros(cls, v):
        """Validate that numbers are in range and different."""
        for num in v:
            if not (0 <= num <= 99):
                raise ValueError('Números deben estar entre 0 y 99')
        
        if len(set(v)) != len(v):
            raise ValueError('Los números no pueden repetirse')
        
        return sorted(v)


class TripletaPrediction(BaseModel):
    """Schema for tripleta prediction."""
    posicion: int = Field(..., ge=1, le=3, description="Posición de la predicción (1-3)")
    numeros: List[int] = Field(..., min_items=3, max_items=3, description="Trío de números")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de acierto")
    metodo_generacion: str = Field(..., description="Método usado para generar la predicción")
    score_confianza: float = Field(..., ge=0.0, le=1.0, description="Score de confianza")
    
    @validator('numeros')
    def validate_numeros(cls, v):
        """Validate that numbers are in range and different."""
        for num in v:
            if not (0 <= num <= 99):
                raise ValueError('Números deben estar entre 0 y 99')
        
        if len(set(v)) != len(v):
            raise ValueError('Los números no pueden repetirse')
        
        return sorted(v)


class PredictionResponse(BaseModel):
    """Complete prediction response schema."""
    fecha: str = Field(..., description="Fecha de las predicciones")
    tipo_loteria_id: int = Field(..., description="ID del tipo de lotería")
    quiniela: List[QuinielaPrediction] = Field(default=[], description="Predicciones de quiniela")
    pale: List[PalePrediction] = Field(default=[], description="Predicciones de palé")
    tripleta: List[TripletaPrediction] = Field(default=[], description="Predicciones de tripleta")
    
    class Config:
        schema_extra = {
            "example": {
                "fecha": "2024-01-15",
                "tipo_loteria_id": 1,
                "quiniela": [
                    {
                        "posicion": 1,
                        "numero": 23,
                        "probabilidad": 0.85,
                        "metodo_generacion": "neural_network",
                        "score_confianza": 0.92
                    }
                ],
                "pale": [
                    {
                        "posicion": 1,
                        "numeros": [23, 45],
                        "probabilidad": 0.68,
                        "metodo_generacion": "monte_carlo",
                        "score_confianza": 0.75
                    }
                ],
                "tripleta": [
                    {
                        "posicion": 1,
                        "numeros": [23, 45, 67],
                        "probabilidad": 0.45,
                        "metodo_generacion": "statistical",
                        "score_confianza": 0.68
                    }
                ]
            }
        }


# Sorteo schemas
class SorteoInput(BaseModel):
    """Schema for sorteo input."""
    fecha: date = Field(..., description="Fecha del sorteo")
    tipo_loteria_id: int = Field(..., ge=1, description="ID del tipo de lotería")
    primer_lugar: int = Field(..., ge=0, le=99, description="Número en primer lugar")
    segundo_lugar: int = Field(..., ge=0, le=99, description="Número en segundo lugar")
    tercer_lugar: int = Field(..., ge=0, le=99, description="Número en tercer lugar")
    fuente_scraping: Optional[str] = Field(None, description="Fuente del scraping")
    
    @validator('fecha')
    def validate_fecha(cls, v):
        """Validate that date is not in the future."""
        if v > date.today():
            raise ValueError('La fecha no puede ser futura')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "fecha": "2024-01-15",
                "tipo_loteria_id": 1,
                "primer_lugar": 23,
                "segundo_lugar": 45,
                "tercer_lugar": 67,
                "fuente_scraping": "loteria-nacional.com"
            }
        }


class SorteoResponse(BaseModel):
    """Schema for sorteo response."""
    id: int = Field(..., description="ID único del sorteo")
    fecha: date = Field(..., description="Fecha del sorteo")
    tipo_loteria_id: int = Field(..., description="ID del tipo de lotería")
    primer_lugar: int = Field(..., description="Número en primer lugar")
    segundo_lugar: int = Field(..., description="Número en segundo lugar")
    tercer_lugar: int = Field(..., description="Número en tercer lugar")
    fuente_scraping: Optional[str] = Field(None, description="Fuente del scraping")
    creado_en: datetime = Field(..., description="Timestamp de creación")


# Statistics schemas
class NumberFrequency(BaseModel):
    """Schema for number frequency statistics."""
    numero: int = Field(..., ge=0, le=99, description="Número")
    frecuencia: int = Field(..., ge=0, description="Frecuencia de aparición")


class StatisticsResponse(BaseModel):
    """Schema for historical statistics response."""
    timestamp: datetime = Field(..., description="Timestamp de la consulta")
    periodo_dias: int = Field(..., description="Período en días analizado")
    tipo_loteria_id: Optional[int] = Field(None, description="ID del tipo de lotería")
    total_sorteos: int = Field(..., description="Total de sorteos en el período")
    numeros_mas_frecuentes: List[NumberFrequency] = Field(..., description="Números más frecuentes")
    numeros_menos_frecuentes: List[NumberFrequency] = Field(..., description="Números menos frecuentes")
    promedios: Dict[str, float] = Field(..., description="Promedios por posición")


# Model performance schemas
class ModelPerformance(BaseModel):
    """Schema for individual model performance."""
    avg_accuracy: float = Field(..., description="Precisión promedio")
    recent_accuracy: float = Field(..., description="Precisión reciente")
    evaluations: int = Field(..., description="Número de evaluaciones")


class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    description: str = Field(..., description="Descripción del modelo")
    version: str = Field(..., description="Versión del modelo")
    is_active: bool = Field(..., description="Si el modelo está activo")


class ModelInfo(BaseModel):
    """Schema for complete model information."""
    performance: ModelPerformance
    metadata: ModelMetadata


class ModelPerformanceResponse(BaseModel):
    """Schema for model performance response."""
    timestamp: datetime = Field(..., description="Timestamp de la consulta")
    models: Dict[str, ModelInfo] = Field(..., description="Información de modelos")


# Configuration schemas
class TipoLoteriaResponse(BaseModel):
    """Schema for lottery type response."""
    id: int = Field(..., description="ID único")
    nombre: str = Field(..., description="Nombre del tipo de lotería")
    descripcion: Optional[str] = Field(None, description="Descripción")
    hora_sorteo: Optional[str] = Field(None, description="Hora del sorteo")


class TipoJuegoResponse(BaseModel):
    """Schema for game type response."""
    id: int = Field(..., description="ID único")
    nombre: str = Field(..., description="Nombre del tipo de juego")
    descripcion: Optional[str] = Field(None, description="Descripción")
    formato_numeros: str = Field(..., description="Formato de números")


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str = Field(..., description="Detalle del error")
    status_code: int = Field(..., description="Código de estado HTTP")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp del error")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Tipo de lotería no encontrado",
                "status_code": 404,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


# Training schemas
class TrainingRequest(BaseModel):
    """Schema for model training request."""
    tipo_loteria_id: Optional[int] = Field(None, description="ID específico de lotería")
    forzar: bool = Field(False, description="Forzar reentrenamiento")
    modelos: Optional[List[str]] = Field(None, description="Modelos específicos a entrenar")


class TrainingResponse(BaseModel):
    """Schema for training response."""
    message: str = Field(..., description="Mensaje de resultado")
    resultados: Dict[str, Any] = Field(..., description="Resultados del entrenamiento")
    timestamp: datetime = Field(..., description="Timestamp del entrenamiento")


# Prediction generation schemas
class PredictionGenerationRequest(BaseModel):
    """Schema for prediction generation request."""
    fecha: Optional[date] = Field(None, description="Fecha objetivo")
    tipo_loteria_id: int = Field(..., description="ID del tipo de lotería")
    metodos: Optional[List[str]] = Field(None, description="Métodos específicos a usar")


class PredictionGenerationResponse(BaseModel):
    """Schema for prediction generation response."""
    message: str = Field(..., description="Mensaje de resultado")
    fecha: str = Field(..., description="Fecha de las predicciones")
    tipo_loteria_id: int = Field(..., description="ID del tipo de lotería")
    predicciones: Dict[str, Any] = Field(..., description="Predicciones generadas")


# System status schemas
class DatabaseStatus(BaseModel):
    """Schema for database status."""
    total_sorteos: int = Field(..., description="Total de sorteos")
    total_predicciones: int = Field(..., description="Total de predicciones")
    ultimo_sorteo: Optional[str] = Field(None, description="Fecha del último sorteo")


class ModelsStatus(BaseModel):
    """Schema for models status."""
    modelos_disponibles: int = Field(..., description="Modelos disponibles")
    archivos_cache: int = Field(..., description="Archivos en caché")


class CacheStatus(BaseModel):
    """Schema for cache status."""
    estado: str = Field(..., description="Estado del caché")


class SystemStatusResponse(BaseModel):
    """Schema for system status response."""
    estado: str = Field(..., description="Estado general del sistema")
    timestamp: datetime = Field(..., description="Timestamp de la consulta")
    base_datos: DatabaseStatus = Field(..., description="Estado de la base de datos")
    modelos: ModelsStatus = Field(..., description="Estado de los modelos")
    cache: CacheStatus = Field(..., description="Estado del caché")


# Cache management schemas
class CacheResponse(BaseModel):
    """Schema for cache operation response."""
    message: str = Field(..., description="Mensaje de resultado")
    keys_eliminadas: int = Field(..., description="Número de keys eliminadas")
    timestamp: datetime = Field(..., description="Timestamp de la operación")


# Validation helpers
def validate_lottery_numbers(numbers: List[int], min_count: int, max_count: int) -> List[int]:
    """Validate lottery numbers."""
    if len(numbers) < min_count or len(numbers) > max_count:
        raise ValueError(f'Se requieren entre {min_count} y {max_count} números')
    
    for num in numbers:
        if not (0 <= num <= 99):
            raise ValueError('Números deben estar entre 0 y 99')
    
    if len(set(numbers)) != len(numbers):
        raise ValueError('Los números no pueden repetirse')
    
    return sorted(numbers)


def validate_prediction_position(position: int) -> int:
    """Validate prediction position."""
    if not (1 <= position <= 3):
        raise ValueError('Posición debe ser 1, 2 o 3')
    return position


def validate_probability(probability: float) -> float:
    """Validate probability value."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError('Probabilidad debe estar entre 0.0 y 1.0')
    return probability


def validate_confidence_score(score: float) -> float:
    """Validate confidence score."""
    if not (0.0 <= score <= 1.0):
        raise ValueError('Score de confianza debe estar entre 0.0 y 1.0')
    return score