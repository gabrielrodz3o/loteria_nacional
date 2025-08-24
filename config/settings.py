"""Optimized application settings using Pydantic BaseSettings with performance improvements."""

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os
import multiprocessing as mp


class OptimizedSettings(BaseSettings):
    """Optimized application settings with performance tuning for high-volume processing."""
    
    # ========================================
    # DATABASE CONFIGURATION (OPTIMIZED)
    # ========================================
    database_url: str = "postgresql://loteria_user:LoteriaPass2024!@localhost:5432/loteria_db"
    test_database_url: str = "postgresql://loteria_user:LoteriaPass2024!@localhost:5432/loteria_test_db"
    
    # Connection pool settings (SIGNIFICANTLY INCREASED)
    db_pool_size: int = 25  # Increased from default 10
    db_max_overflow: int = 50  # Increased from default 20  
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600  # 1 hour
    db_pool_pre_ping: bool = True  # Verify connections
    db_echo: bool = False  # Disable for performance
    
    # Query optimization
    db_query_timeout: int = 30  # Query timeout in seconds
    db_batch_size: int = 1000  # Batch insert size
    
    # ========================================
    # REDIS CONFIGURATION (OPTIMIZED)
    # ========================================
    redis_url: str = "redis://:LoteriaRedis2024!@localhost:6379/0"
    redis_max_connections: int = 20  # Connection pool
    redis_socket_timeout: int = 10
    redis_socket_connect_timeout: int = 10
    redis_retry_on_timeout: bool = True
    redis_health_check_interval: int = 30
    
    # ========================================
    # API CONFIGURATION 
    # ========================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False  # Disabled for production performance
    api_workers: int = min(mp.cpu_count(), 4)  # Multiple workers
    secret_key: str = "your-secret-key-here-change-in-production"
    
    # Request handling optimization
    request_timeout: int = 60  # Increased for heavy ML operations
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    keepalive_timeout: int = 5
    
    # ========================================
    # SCRAPING CONFIGURATION
    # ========================================
    selenium_enabled: bool = True
    selenium_headless: bool = True
    scraping_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Scraping optimization
    scraping_concurrent_sessions: int = 3  # Multiple concurrent scrapers
    scraping_request_delay: float = 1.0  # Delay between requests
    scraping_retry_attempts: int = 3
    scraping_timeout: int = 30
    
    # ========================================
    # MACHINE LEARNING CONFIGURATION (ACTUALIZADO AGOSTO 2025)
    # ========================================
    ml_model_cache_days: int = 7
    
    # CAMBIO CRÍTICO: Limitar ventana histórica
    historical_window_days: int = 365  # MANTENER: máximo 1 año
    prediction_window_days: int = 90   # NUEVO: 3 meses para predicciones diarias
    max_training_window_days: int = 730  # NUEVO: 2 años absoluto máximo
    
    min_training_samples: int = 20  # CAMBIO: reducido de 10 a 20 para mejor calidad
    prediction_batch_size: int = 300  # CAMBIO: reducido de 500 a 300
    
    # Training optimization (OPTIMIZADO PARA AGOSTO)
    max_training_workers: int = min(mp.cpu_count(), 6)  # CAMBIO: máximo 6 workers
    training_timeout_seconds: int = 300  # CAMBIO: 5 minutos (aumentado de 3min)
    training_memory_limit_mb: int = 1024  # MANTENER: 1GB por modelo
    
    # Model selection optimization (MODELOS ACTUALIZADOS - CRÍTICO)
    priority_models: List[str] = [
        'frequencyanalysis', 'randomforest', 'montecarlo', 'bayesian', 'lightgbm', 'xgboost'
    ]
    
    # NUEVO: Modelos a saltar si hay pocos datos
    skip_heavy_models_threshold: int = 100  # Skip neural networks si < 100 muestras
    
    # Model-specific optimized configurations (AJUSTADO PARA AGOSTO)
    ml_model_configs: Dict[str, Dict[str, Any]] = {
        'lightgbm': {
            'n_estimators': 30,  # CAMBIO: reducido de 50 a 30
            'max_depth': 5,      # CAMBIO: reducido de 6 a 5
            'learning_rate': 0.2,  # CAMBIO: aumentado para convergencia rápida
            'n_jobs': -1,
            'verbosity': -1
        },
        'xgboost': {
            'n_estimators': 30,  # CAMBIO: reducido de 50 a 30
            'max_depth': 5,      # CAMBIO: reducido de 6 a 5
            'learning_rate': 0.2,
            'n_jobs': -1,
            'verbosity': 0
        },
        'randomforest': {
            'n_estimators': 20,  # CAMBIO: reducido de 30 a 20
            'max_depth': 6,      # CAMBIO: reducido de 8 a 6
            'n_jobs': -1,
            'verbose': 0
        },
        'neural_network': {
            'sequence_length': 10,  # CAMBIO: reducido de 15 a 10
            'hidden_units': [32, 16],  # CAMBIO: arquitectura más pequeña
            'epochs': 15,  # CAMBIO: reducido de 20 a 15
            'batch_size': 16  # CAMBIO: reducido de 32 a 16
        },
        'frequencyanalysis': {
            'window_size': 50,  # NUEVO: ventana pequeña
            'smoothing_factor': 0.5
        },
        'montecarlo': {
            'n_simulations': 5000,  # NUEVO: reducido para velocidad
            'random_seed': 42
        },
        'bayesian': {
            'prior_alpha': 0.5,  # NUEVO: prior más informativo
            'update_rate': 0.05
        }
    }
    
    # ========================================
    # SCHEDULER CONFIGURATION (OPTIMIZED)
    # ========================================
    enable_scheduler: bool = True
    scraping_schedule: str = "0 22 * * *"  # Daily at 10 PM
    prediction_schedule: str = "0 6 * * *"   # Changed to 6 AM for better performance
    cleanup_schedule: str = "0 2 * * 0"     # Weekly at 2 AM Sunday
    
    # Batch processing schedules
    batch_training_schedule: str = "0 1 * * *"  # 1 AM daily training
    batch_prediction_schedule: str = "30 6 * * *"  # 6:30 AM predictions
    
    # ========================================
    # LOGGING CONFIGURATION
    # ========================================
    log_level: str = "INFO"
    log_format: str = "json"
    log_file_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    log_performance_metrics: bool = True
    
    # Performance logging
    enable_performance_logging: bool = True
    log_slow_queries_threshold: float = 1.0  # Log queries > 1 second
    log_memory_usage: bool = True
    
    # ========================================
    # CACHE CONFIGURATION (OPTIMIZADO AGOSTO)
    # ========================================
    cache_ttl_predictions: int = 1200  # CAMBIO: 20 minutos (reducido de 30min)
    cache_ttl_statistics: int = 2400   # CAMBIO: 40 minutos (reducido de 1 hora)
    cache_ttl_models: int = 43200      # CAMBIO: 12 horas (reducido de 24 horas)
    
    # In-memory cache settings (OPTIMIZADO)
    memory_cache_size_mb: int = 512  # CAMBIO: aumentado de 256MB a 512MB
    cache_cleanup_interval: int = 180  # CAMBIO: 3 minutos (reducido de 5min)
    
    # Data cache optimization (MEJORADO)
    data_cache_enabled: bool = True
    data_cache_ttl: int = 1800  # CAMBIO: 30 minutos (reducido de 1 hora)
    data_cache_max_entries: int = 500  # CAMBIO: reducido de 1000 a 500
    
    # ========================================
    # VECTOR/EMBEDDING CONFIGURATION (OPTIMIZED)
    # ========================================
    embedding_dimensions: int = 64  # MANTENER: 64 dimensiones
    embedding_batch_size: int = 50   # CAMBIO: reducido de 100 a 50
    vector_similarity_threshold: float = 0.75  # CAMBIO: reducido de 0.8 a 0.75
    
    # ========================================
    # LOTTERY SPECIFIC (OPTIMIZED)
    # ========================================
    max_predictions_per_game: int = 3
    number_range_min: int = 0
    number_range_max: int = 99
    
    # Processing optimization
    parallel_game_processing: bool = True
    prediction_confidence_threshold: float = 0.05  # CAMBIO: reducido de 0.1 a 0.05
    
    # ========================================
    # SECURITY CONFIGURATION
    # ========================================
    cors_origins: Optional[str] = "*"  # Configure appropriately for production
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60  # Increased for batch operations
    rate_limit_burst: int = 10
    
    # ========================================
    # PERFORMANCE MONITORING (AGOSTO 2025)
    # ========================================
    enable_profiling: bool = False
    memory_profiling: bool = False
    query_profiling: bool = False
    
    # Performance thresholds (AJUSTADOS)
    max_memory_usage_mb: int = 3072  # CAMBIO: 3GB limit (aumentado de 2GB)
    max_cpu_usage_percent: float = 85.0  # CAMBIO: aumentado de 80% a 85%
    max_processing_time_minutes: int = 45  # CAMBIO: aumentado de 30 a 45 min
    
    # Garbage collection optimization (MEJORADO)
    gc_threshold: int = 50  # CAMBIO: reducido de 100 a 50
    gc_enabled: bool = True
    
    # ========================================
    # BATCH PROCESSING (OPTIMIZADO AGOSTO 2025)
    # ========================================
    batch_processing_enabled: bool = True
    batch_size: int = 20  # CAMBIO: reducido de 50 a 20 para agosto
    max_concurrent_batches: int = 2  # CAMBIO: reducido de 3 a 2
    batch_retry_attempts: int = 3  # CAMBIO: aumentado de 2 a 3
    batch_timeout_minutes: int = 60  # CAMBIO: aumentado de 45 a 60 min
    
    # Date range processing (CRÍTICO PARA AGOSTO)
    max_dates_per_batch: int = 15  # CAMBIO: reducido de 30 a 15
    skip_existing_predictions: bool = True
    
    # NUEVO: Configuraciones específicas para procesamiento masivo
    enable_progressive_loading: bool = True  # Carga progresiva de datos
    memory_cleanup_interval: int = 10  # Limpiar memoria cada 10 operaciones
    force_gc_after_batch: bool = True  # Forzar garbage collection después de cada lote
    
    # NUEVO: Límites de recursos para agosto
    max_models_per_batch: int = 6  # Máximo 6 modelos por lote
    model_training_max_memory_mb: int = 800  # 800MB por modelo
    enable_memory_monitoring: bool = True  # Monitorear memoria activamente
    
    # ========================================
    # DEVELOPMENT/DEBUG SETTINGS
    # ========================================
    debug: bool = False
    testing: bool = False
    enable_sql_echo: bool = False  # Disable SQL logging for performance
    
    # ========================================
    # PYDANTIC CONFIGURATION (FIX FOR WARNINGS)
    # ========================================
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
        "protected_namespaces": ()  # This fixes the namespace warnings
    }


# Global settings instance - Use optimized version
settings = OptimizedSettings()

# Backward compatibility alias
Settings = OptimizedSettings