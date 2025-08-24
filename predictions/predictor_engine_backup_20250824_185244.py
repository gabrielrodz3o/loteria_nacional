"""Main prediction engine that orchestrates all models and generates daily predictions - FIXED VERSION."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import json
import pickle
import os
import hashlib
import time

from config.database import get_db_connection, db_manager
from models.database_models import (
    Sorteo, PrediccionQuiniela, PrediccionPale, PrediccionTripleta,
    TipoLoteria, TipoJuego, Vector
)
from models.prediction_models import (
    model_registry, model_evaluator, PredictionResult, GamePredictions
)
from predictions.calibration import ProbabilityCalibrator
from utils.embeddings import create_embedding, store_embedding
from utils.helpers import validate_number_range, generate_combinations
from config.settings import settings

logger = logging.getLogger(__name__)


class OptimizedPredictorEngine:
    """Optimized engine for generating lottery predictions using multiple models."""
    
    def __init__(self):
        self.models = {}
        self.calibrator = ProbabilityCalibrator()
        self.model_cache_dir = "models_cache"
        self.performance_history = defaultdict(list)
        
        # Optimized configuration
        self.data_cache = {}
        self.max_workers = min(mp.cpu_count(), 8)  # Use more CPUs
        self.chunk_size = 100  # Process in batches
        self.cache_ttl = 3600  # Cache TTL in seconds
        
        # Ensure cache directory exists
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Initialize available models (lazy loading)
        self._models_initialized = False
    
    def _initialize_models(self):
        """Initialize models on demand - FIXED VERSION WITH PERMANENT BYPASS."""
        if self._models_initialized:
            return
            
        try:
            logger.info("[INIT] Initializing prediction models with PERMANENT BYPASS...")
            
            # BYPASS PERMANENTE: Usar TODOS los modelos del registry, ignorar settings
            available_models = model_registry.list_available_models()
            
            logger.info(f"[INIT] Available models from registry: {available_models}")
            logger.info(f"[INIT] BYPASS: Ignoring settings.priority_models, using ALL registry models")
            
            # Configurar TODOS los modelos disponibles del registry
            for model_name in available_models:
                try:
                    # Store model configuration instead of instance (for multiprocessing)
                    self.models[model_name] = {'type': model_name, 'config': {}}
                    logger.info(f"[INIT] Configured model: {model_name}")
                except Exception as e:
                    logger.warning(f"[INIT] Failed to configure model {model_name}: {e}")
            
            self._models_initialized = True
            logger.info(f"[INIT] Successfully configured {len(self.models)} models (PERMANENT BYPASS VERSION)")
            logger.info(f"[INIT] Configured models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"[INIT] Model initialization failed: {e}")
            raise
    def _get_data_cache_key(self, tipo_loteria_id: int, days_back: int) -> str:
        """Generate unique cache key for data."""
        return f"data_{tipo_loteria_id}_{days_back}_{date.today().isoformat()}"
    
    def _load_historical_data(self, tipo_loteria_id: int, days_back: int = None, session=None) -> pd.DataFrame:
        """Load historical lottery data with optimized caching."""
        try:
            days_back = days_back or settings.historical_window_days
            cache_key = self._get_data_cache_key(tipo_loteria_id, days_back)
            
            # Check cache first
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    logger.info(f"[CACHE-HIT] Using cached data for lottery {tipo_loteria_id}")
                    return cached_data
            
            logger.info(f"[LOAD-DATA] Loading historical data for lottery_type_id={tipo_loteria_id}, days={days_back}")
            
            if session is None:
                with get_db_connection() as session:
                    return self._fetch_data_from_db(session, tipo_loteria_id, days_back, cache_key)
            else:
                return self._fetch_data_from_db(session, tipo_loteria_id, days_back, cache_key)
                
        except Exception as e:
            logger.error(f"[LOAD-DATA] Error loading historical data for lottery_type_id={tipo_loteria_id}: {e}")
            return pd.DataFrame()
    
    def _fetch_data_from_db(self, session, tipo_loteria_id: int, days_back: int, cache_key: str) -> pd.DataFrame:
        """Fetch data from database and cache it."""
        query = session.query(Sorteo).filter(
            Sorteo.tipo_loteria_id == tipo_loteria_id
        ).order_by(Sorteo.fecha.desc()).limit(days_back)
        
        results = query.all()
        
        if not results:
            logger.warning(f"[LOAD-DATA] No data found for lottery type {tipo_loteria_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame efficiently
        data = [{
            'fecha': sorteo.fecha,
            'primer_lugar': sorteo.primer_lugar,
            'segundo_lugar': sorteo.segundo_lugar,
            'tercer_lugar': sorteo.tercer_lugar
        } for sorteo in results]
        
        df = pd.DataFrame(data).sort_values('fecha')
        
        # Cache the result
        self.data_cache[cache_key] = (time.time(), df)
        
        logger.info(f"[LOAD-DATA] Loaded and cached {len(df)} records for lottery type {tipo_loteria_id}")
        return df
    
    def _prepare_all_game_data_batch(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare training data for all game types in a single pass."""
        logger.info(f"[BATCH-PREP] Preparing data for all games. Input rows: {len(df)}")
        
        if df.empty:
            return {game: (np.array([]), np.array([])) for game in ['quiniela', 'pale', 'tripleta']}
        
        try:
            # Clean data once
            df_clean = df.dropna(subset=['primer_lugar', 'segundo_lugar', 'tercer_lugar'])
            
            for col in ['primer_lugar', 'segundo_lugar', 'tercer_lugar']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean = df_clean[(df_clean[col] >= 0) & (df_clean[col] <= 99) & df_clean[col].notna()]
            
            if df_clean.empty:
                logger.warning("[BATCH-PREP] No valid data after cleaning")
                return {game: (np.array([]), np.array([])) for game in ['quiniela', 'pale', 'tripleta']}
            
            logger.info(f"[BATCH-PREP] Clean data: {len(df_clean)} rows")
            
            results = {}
            
            # QUINIELA - All positions as sequence
            quiniela_data = pd.concat([
                df_clean['primer_lugar'],
                df_clean['segundo_lugar'],
                df_clean['tercer_lugar']
            ]).values
            
            quiniela_data = quiniela_data[~pd.isna(quiniela_data) & (quiniela_data >= 0)]
            if len(quiniela_data) >= 2:
                results['quiniela'] = (quiniela_data[:-1], quiniela_data[1:])
                logger.info(f"[BATCH-PREP] Quiniela sequences: {len(results['quiniela'][0])}")
            else:
                results['quiniela'] = (np.array([]), np.array([]))
            
            # PALE - Individual numbers from combinations
            pale_numbers = []
            for _, row in df_clean.iterrows():
                nums = [row['primer_lugar'], row['segundo_lugar'], row['tercer_lugar']]
                pale_numbers.extend([n for n in nums if pd.notna(n) and n >= 0])
            
            pale_numbers = np.array(pale_numbers)
            if len(pale_numbers) >= 2:
                results['pale'] = (pale_numbers[:-1], pale_numbers[1:])
                logger.info(f"[BATCH-PREP] Pale sequences: {len(results['pale'][0])}")
            else:
                results['pale'] = (np.array([]), np.array([]))
            
            # TRIPLETA - Flattened sequence
            tripleta_data = df_clean[['primer_lugar', 'segundo_lugar', 'tercer_lugar']].values.ravel()
            tripleta_data = tripleta_data[~pd.isna(tripleta_data) & (tripleta_data >= 0)]
            
            if len(tripleta_data) >= 2:
                results['tripleta'] = (tripleta_data[:-1], tripleta_data[1:])
                logger.info(f"[BATCH-PREP] Tripleta sequences: {len(results['tripleta'][0])}")
            else:
                results['tripleta'] = (np.array([]), np.array([]))
            
            return results
            
        except Exception as e:
            logger.error(f"[BATCH-PREP] Error preparing training data: {e}")
            return {game: (np.array([]), np.array([])) for game in ['quiniela', 'pale', 'tripleta']}
    
    def _get_model_cache_path(self, model_name: str, lottery_type_id: int, game_type: str, fecha: date) -> str:
        """Generate cache path for model."""
        cache_key = f"{model_name}_{lottery_type_id}_{game_type}_{fecha.strftime('%Y%m%d')}"
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return os.path.join(self.model_cache_dir, f"model_{hash_key}.pkl")
    
    def _train_single_model_task(self, args):
        """Train a single model (optimized for multiprocessing)."""
        model_name, X, y, lottery_type_id, game_type, fecha = args
        
        try:
            logger.info(f"[TRAIN-MODEL] Starting training for {model_name} on {game_type}")
            
            # Check cache first
            cache_path = self._get_model_cache_path(model_name, lottery_type_id, game_type, fecha)
            if os.path.exists(cache_path):
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < self.cache_ttl:
                    logger.info(f"[TRAIN-MODEL] Using cached model for {model_name}")
                    return model_name, {'status': 'cached', 'cache_path': cache_path}
            
            # Skip if insufficient data
            if len(X) < 10:  # Reduced minimum requirement
                logger.warning(f"[TRAIN-MODEL] Insufficient data for {model_name}: {len(X)} samples")
                return model_name, {'status': 'insufficient_data', 'samples': len(X)}
            
            # Create and train model
            model = model_registry.create_model(model_name)
            if model is None:
                logger.error(f"[TRAIN-MODEL] Model creation failed for {model_name}")
                return model_name, {'status': 'failed', 'error': 'Model creation failed'}
            
            # CORRECCIÓN: Asegurar que el modelo tenga el nombre correcto
            if not hasattr(model, 'name'):
                model.name = model_name
            if not hasattr(model, 'model_name'):
                model.model_name = model_name
            
            logger.info(f"[TRAIN-MODEL] Training {model_name} with {len(X)} samples")
            
            # Train with timeout protection
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            logger.info(f"[TRAIN-MODEL] Successfully trained {model_name} in {training_time:.2f}s")
            
            # CORRECCIÓN: Asegurar que el modelo mantenga su identidad después del entrenamiento
            model.trained_model_name = model_name
            model.training_info = {
                'lottery_type_id': lottery_type_id,
                'game_type': game_type,
                'fecha': fecha.isoformat(),
                'training_samples': len(X)
            }
            
            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"[TRAIN-MODEL] Cached trained model {model_name} at {cache_path}")
            
            return model_name, {
                'status': 'success',
                'training_samples': len(X),
                'training_time': training_time,
                'accuracy': getattr(model.metadata, 'accuracy_score', None) if hasattr(model, 'metadata') else None,
                'cache_path': cache_path,
                'model_name': model_name  # CORRECCIÓN: Incluir nombre explícitamente
            }
            
        except Exception as e:
            logger.error(f"[TRAIN-MODEL] Training failed for {model_name}: {e}", exc_info=True)
            return model_name, {'status': 'failed', 'error': str(e)}
    
    def entrenar_modelos(self, fecha: date = None, tipo_loteria_id: int = None) -> Dict[str, Any]:
        """Train all models with optimized batch processing."""
        try:
            fecha = fecha or date.today()
            logger.info(f"[TRAIN] Starting optimized model training for {fecha}")
            
            # Initialize models if needed
            self._initialize_models()
            
            # Get lottery types to train
            if tipo_loteria_id:
                lottery_types = [tipo_loteria_id]
            else:
                with get_db_connection() as session:
                    types = session.query(TipoLoteria).filter(TipoLoteria.activo == True).all()
                    lottery_types = [t.id for t in types]
            
            logger.info(f"[TRAIN] Processing lottery types: {lottery_types}")
            
            training_results = {}
            
            for lot_type_id in lottery_types:
                logger.info(f"[TRAIN] Processing lottery type {lot_type_id}")
                
                # Load historical data once
                df = self._load_historical_data(lot_type_id)
                
                if df.empty:
                    logger.warning(f"[TRAIN] No data for lottery type {lot_type_id}")
                    continue
                
                # Prepare all game data in batch
                all_game_data = self._prepare_all_game_data_batch(df)
                
                type_results = {}
                
                # Process each game type
                for game_type in ['quiniela', 'pale', 'tripleta']:
                    X, y = all_game_data[game_type]
                    
                    if len(X) < 10:
                        logger.warning(f"[TRAIN] Insufficient data for {game_type}: {len(X)} samples")
                        continue
                    
                    logger.info(f"[TRAIN] Training {game_type} with {len(X)} samples")
                    
                    # Prepare training tasks
                    training_tasks = []
                    for model_name in self.models.keys():
                        training_tasks.append((model_name, X.copy(), y.copy(), lot_type_id, game_type, fecha))
                    
                    # Execute training in parallel
                    game_results = {}
                    
                    # Use ProcessPoolExecutor for CPU-intensive training
                    max_workers = min(len(training_tasks), self.max_workers)
                    
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all training tasks
                        future_to_model = {
                            executor.submit(self._train_single_model_task, task): task[0] 
                            for task in training_tasks
                        }
                        
                        # Collect results as they complete
                        completed = 0
                        for future in as_completed(future_to_model, timeout=300):  # 5 min timeout per model
                            model_name = future_to_model[future]
                            try:
                                result_name, result_data = future.result()
                                game_results[result_name] = result_data
                                completed += 1
                                
                                if completed % 5 == 0:  # Log progress every 5 models
                                    logger.info(f"[TRAIN] Progress: {completed}/{len(training_tasks)} models completed for {game_type}")
                                    
                            except Exception as e:
                                logger.error(f"[TRAIN] Training failed for {model_name}: {e}")
                                game_results[model_name] = {'status': 'failed', 'error': str(e)}
                    
                    type_results[game_type] = game_results
                    logger.info(f"[TRAIN] Completed {game_type}: {len(game_results)} models processed")
                
                training_results[f'lottery_type_{lot_type_id}'] = type_results
            
            # Clear data cache to free memory
            self.data_cache.clear()
            
            logger.info(f"[TRAIN] Training completed. Results: {len(training_results)} lottery types processed")
            return training_results
            
        except Exception as e:
            logger.error(f"[TRAIN] Training failed: {e}")
            raise
    
    def _load_cached_model(self, model_name: str, lottery_type_id: int, game_type: str, fecha: date):
        """Load trained model from cache."""
        try:
            cache_path = self._get_model_cache_path(model_name, lottery_type_id, game_type, fecha)
            
            if os.path.exists(cache_path):
                # Check if cache is still valid
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < self.cache_ttl:
                    logger.info(f"[CACHE-LOAD] Loading cached model {model_name} from {cache_path}")
                    
                    with open(cache_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # CORRECCIÓN: Verificar y restaurar identidad del modelo
                    if not hasattr(model, 'name'):
                        model.name = model_name
                    if not hasattr(model, 'model_name'):
                        model.model_name = model_name
                    if not hasattr(model, 'trained_model_name'):
                        model.trained_model_name = model_name
                    
                    logger.info(f"[CACHE-LOAD] Successfully loaded {model_name} with identity preserved")
                    return model
                else:
                    logger.info(f"[CACHE-LOAD] Cache expired for {model_name}, age: {cache_age}s")
            else:
                logger.info(f"[CACHE-LOAD] No cache found for {model_name} at {cache_path}")
            
            return None
            
        except Exception as e:
            logger.error(f"[CACHE-LOAD] Error loading cached model {model_name}: {e}")
            return None
    
    def generar_predicciones_diarias(self, fecha: str, tipo_loteria_id: int) -> Dict[str, List[Dict]]:
        """Generate daily predictions with optimized processing."""
        try:
            fecha_obj = datetime.strptime(fecha, '%Y-%m-%d').date()
            logger.info(f"[PREDICT] Generating predictions for {fecha} - lottery type {tipo_loteria_id}")
            
            # Initialize models if needed
            self._initialize_models()
            
            # Load historical data once
            df = self._load_historical_data(tipo_loteria_id)
            
            if df.empty:
                logger.warning("[PREDICT] No historical data available")
                return self._generate_fallback_predictions()
            
            # Prepare all game data once
            all_game_data = self._prepare_all_game_data_batch(df)
            
            predictions = {}
            
            # Generate predictions for each game type
            for game_type in ['quiniela', 'pale', 'tripleta']:
                X, _ = all_game_data[game_type]
                
                if len(X) == 0:
                    logger.warning(f"[PREDICT] No data available for {game_type}")
                    predictions[game_type] = self._generate_fallback_game_predictions(game_type)
                    continue
                
                logger.info(f"[PREDICT] Generating {game_type} predictions with {len(self.models)} models")
                
                # Collect predictions from all available models
                model_predictions = []
                successful_models = 0
                
                for model_name in self.models.keys():
                    try:
                        logger.info(f"[PREDICT] Trying model {model_name} for {game_type}")
                        
                        # Try to load cached trained model
                        cached_model = self._load_cached_model(model_name, tipo_loteria_id, game_type, fecha_obj)
                        
                        if cached_model:
                            logger.info(f"[PREDICT] Loaded cached model {model_name}")
                            pred = self._generate_single_model_predictions(cached_model, X, game_type)
                            if pred:
                                model_predictions.extend(pred)
                                successful_models += 1
                                logger.info(f"[PREDICT] Got {len(pred)} predictions from {model_name}")
                            else:
                                logger.warning(f"[PREDICT] No predictions from {model_name}")
                        else:
                            logger.warning(f"[PREDICT] No cached model found for {model_name}")
                            
                    except Exception as e:
                        logger.warning(f"[PREDICT] Prediction failed for {model_name} on {game_type}: {e}")
                
                logger.info(f"[PREDICT] Got predictions from {successful_models}/{len(self.models)} models for {game_type}")
                
                # Select best predictions
                if model_predictions:
                    best_predictions = self._select_best_predictions(model_predictions, game_type, n_final=3)
                    predictions[game_type] = [pred.to_dict() for pred in best_predictions]
                    
                    # CORRECCIÓN: Verificar que el método se preserve
                    for i, pred_dict in enumerate(predictions[game_type]):
                        if 'metodo_generacion' not in pred_dict or pred_dict['metodo_generacion'] == 'unknown':
                            # Intentar recuperar del objeto original
                            if i < len(best_predictions):
                                pred_dict['metodo_generacion'] = best_predictions[i].metodo_generacion
                            logger.warning(f"[PREDICT] Fixed missing method for prediction {i+1}: {pred_dict['metodo_generacion']}")
                    
                    logger.info(f"[PREDICT] Generated {len(predictions[game_type])} final predictions for {game_type}")
                else:
                    logger.warning(f"[PREDICT] No model predictions available for {game_type}, using fallback")
                    predictions[game_type] = self._generate_fallback_game_predictions(game_type)
            
            logger.info("[PREDICT] Daily predictions generated successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"[PREDICT] Prediction generation failed: {e}", exc_info=True)
            return self._generate_fallback_predictions()
    
    def _generate_single_model_predictions(self, model, X: np.ndarray, game_type: str, n_predictions: int = 5) -> List[Dict]:
        """Generate predictions from a single model."""
        try:
            predictions = []
            
            # CORRECCIÓN: Obtener nombre del modelo correctamente
            model_name = getattr(model, 'trained_model_name', 
                         getattr(model, 'model_name', 
                         getattr(model, 'name', 'unknown')))
            
            # Verificar si el modelo tiene metadata con el nombre
            if hasattr(model, 'metadata') and hasattr(model.metadata, 'name'):
                model_name = model.metadata.name
            
            logger.info(f"[PREDICT-MODEL] Generating predictions with model: {model_name}")
            
            # Get probability predictions
            probabilities = model.predict_proba(X)
            
            if probabilities.size == 0:
                logger.warning(f"[PREDICT-MODEL] Empty probabilities from {model_name}")
                return []
            
            # Handle different probability shapes
            if len(probabilities.shape) == 2:
                probs = probabilities[0]
            else:
                probs = probabilities
            
            # Get top N numbers by probability
            if len(probs) > 0:
                top_indices = np.argsort(probs)[-n_predictions:][::-1]
                
                for i, idx in enumerate(top_indices):
                    if 0 <= idx < len(probs):
                        if game_type == 'quiniela':
                            numbers = [int(idx)]
                        elif game_type == 'pale':
                            numbers = self._generate_pale_combination(int(idx), probs)
                        elif game_type == 'tripleta':
                            numbers = self._generate_tripleta_combination(int(idx), probs)
                        
                        prediction_dict = {
                            'numbers': numbers,
                            'probability': float(probs[idx]),
                            'confidence': self._get_model_confidence(model, probabilities),
                            'model': model_name  # CORRECCIÓN: Usar model_name obtenido correctamente
                        }
                        
                        predictions.append(prediction_dict)
                        logger.info(f"[PREDICT-MODEL] Generated prediction from {model_name}: {numbers} (prob: {probs[idx]:.4f})")
            
            return predictions
            
        except Exception as e:
            logger.error(f"[PREDICT-MODEL] Model prediction generation failed: {e}")
            return []
    
    def _get_model_confidence(self, model, probabilities) -> float:
        """Calculate model confidence score."""
        try:
            if hasattr(model, 'get_confidence_score'):
                return float(model.get_confidence_score(probabilities))
            else:
                # Default confidence based on probability distribution
                if len(probabilities.shape) == 2:
                    max_prob = np.max(probabilities[0])
                else:
                    max_prob = np.max(probabilities)
                return float(max_prob)
        except:
            return 0.5  # Default confidence
    
    def _generate_pale_combination(self, base_number: int, probabilities: np.ndarray) -> List[int]:
        """Generate a valid pale combination based on base number."""
        try:
            sorted_indices = np.argsort(probabilities)[::-1]
            
            for idx in sorted_indices:
                if 0 <= idx < 100 and idx != base_number:
                    return sorted([base_number, int(idx)])
            
            # Fallback
            second = (base_number + np.random.randint(1, 99)) % 100
            return sorted([base_number, second])
        except:
            return sorted([base_number, (base_number + 1) % 100])
    
    def _generate_tripleta_combination(self, base_number: int, probabilities: np.ndarray) -> List[int]:
        """Generate a valid tripleta combination based on base number."""
        try:
            sorted_indices = np.argsort(probabilities)[::-1]
            numbers = [base_number]
            
            for idx in sorted_indices:
                if 0 <= idx < 100 and idx not in numbers and len(numbers) < 3:
                    numbers.append(int(idx))
            
            # Fill remaining slots if needed
            while len(numbers) < 3:
                candidate = np.random.randint(0, 100)
                if candidate not in numbers:
                    numbers.append(candidate)
            
            return sorted(numbers)
        except:
            # Fallback
            return sorted([base_number, (base_number + 1) % 100, (base_number + 2) % 100])
    
    def _select_best_predictions(self, all_predictions: List[Dict], game_type: str, n_final: int = 3) -> List[PredictionResult]:
        """Select the best N predictions from all model outputs."""
        try:
            logger.info(f"[SELECT-BEST] Selecting from {len(all_predictions)} predictions for {game_type}")
            
            # Sort by combined score (probability * confidence)
            for pred in all_predictions:
                pred['score'] = pred.get('probability', 0.01) * pred.get('confidence', 0.1)
                logger.info(f"[SELECT-BEST] Prediction from {pred.get('model', 'unknown')}: score={pred['score']:.6f}")
            
            all_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Select top unique predictions
            selected = []
            seen_combinations = set()
            
            for pred in all_predictions:
                combo_key = tuple(sorted(pred['numbers']))
                
                if combo_key not in seen_combinations and len(selected) < n_final:
                    seen_combinations.add(combo_key)
                    
                    # CORRECCIÓN: Asegurar que el método se preserve en PredictionResult
                    method_name = pred.get('model', 'unknown')
                    
                    result = PredictionResult(
                        posicion=len(selected) + 1,
                        numeros=pred['numbers'],
                        probabilidad=pred.get('probability', 0.01),
                        metodo_generacion=method_name,  # CORRECCIÓN: Usar método correcto
                        score_confianza=pred.get('confidence', 0.1)
                    )
                    selected.append(result)
                    
                    logger.info(f"[SELECT-BEST] Selected P{result.posicion}: {result.numeros} from {method_name}")
            
            # Fill remaining slots with random valid combinations
            while len(selected) < n_final:
                if game_type == 'quiniela':
                    numbers = [np.random.randint(0, 100)]
                elif game_type == 'pale':
                    numbers = list(np.random.choice(100, 2, replace=False))
                elif game_type == 'tripleta':
                    numbers = list(np.random.choice(100, 3, replace=False))
                
                combo_key = tuple(sorted(numbers))
                if combo_key not in seen_combinations:
                    seen_combinations.add(combo_key)
                    
                    result = PredictionResult(
                        posicion=len(selected) + 1,
                        numeros=numbers,
                        probabilidad=0.01,
                        metodo_generacion='random_fallback',  # CORRECCIÓN: Método identificado
                        score_confianza=0.1
                    )
                    selected.append(result)
                    logger.info(f"[SELECT-BEST] Added fallback P{result.posicion}: {result.numeros}")
            
            return selected
            
        except Exception as e:
            logger.error(f"[SELECT-BEST] Best prediction selection failed: {e}")
            return []
    
    def _generate_fallback_predictions(self) -> Dict[str, List[Dict]]:
        """Generate fallback predictions when primary generation fails."""
        logger.info("[FALLBACK] Generating fallback predictions")
        
        fallback = {}
        
        for game_type in ['quiniela', 'pale', 'tripleta']:
            fallback[game_type] = self._generate_fallback_game_predictions(game_type)
        
        return fallback
    
    def _generate_fallback_game_predictions(self, game_type: str) -> List[Dict]:
        """Generate fallback predictions for a specific game type."""
        predictions = []
        
        logger.info(f"[FALLBACK] Generating fallback predictions for {game_type}")
        
        for i in range(3):
            if game_type == 'quiniela':
                pred = {
                    'posicion': i + 1,
                    'numero': np.random.randint(0, 100),
                    'probabilidad': 0.01 + np.random.random() * 0.05,
                    'metodo_generacion': 'fallback_statistical',  # CORRECCIÓN: Método específico
                    'score_confianza': 0.1
                }
            elif game_type == 'pale':
                nums = list(np.random.choice(100, 2, replace=False))
                pred = {
                    'posicion': i + 1,
                    'numeros': nums,
                    'probabilidad': 0.005 + np.random.random() * 0.02,
                    'metodo_generacion': 'fallback_statistical',  # CORRECCIÓN: Método específico
                    'score_confianza': 0.1
                }
            elif game_type == 'tripleta':
                nums = list(np.random.choice(100, 3, replace=False))
                pred = {
                    'posicion': i + 1,
                    'numeros': nums,
                    'probabilidad': 0.001 + np.random.random() * 0.005,
                    'metodo_generacion': 'fallback_statistical',  # CORRECCIÓN: Método específico
                    'score_confianza': 0.1
                }
            
            predictions.append(pred)
            logger.info(f"[FALLBACK] Created {game_type} P{i+1} with method: fallback_statistical")
        
        return predictions
    
    def insertar_predicciones_en_bd(self, predicciones: Dict, fecha: date, tipo_loteria_id: int) -> None:
        """Insert predictions into database."""
        try:
            logger.info(f"[DB] Inserting predictions for {fecha}")
            
            with get_db_connection() as session:
                # Insert quiniela predictions
                for pred in predicciones.get('quiniela', []):
                    # CORRECCIÓN: Verificar y limpiar método
                    metodo = pred.get('metodo_generacion', 'unknown')
                    if not metodo or metodo == 'unknown':
                        metodo = 'fallback_statistical'
                    
                    quiniela = PrediccionQuiniela(
                        fecha_prediccion=fecha,
                        tipo_loteria_id=tipo_loteria_id,
                        posicion=pred.get('posicion', 1),
                        numero_predicho=pred.get('numero', pred.get('numeros', [0])[0] if pred.get('numeros') else 0),
                        probabilidad=pred.get('probabilidad', 0.01),
                        metodo_generacion=metodo,  # CORRECCIÓN: Usar método verificado
                        score_confianza=pred.get('score_confianza', 0.1)
                    )
                    session.add(quiniela)
                    logger.info(f"[DB] Inserted quiniela P{pred.get('posicion', 1)} with method: {metodo}")
                
                # Insert pale predictions
                for pred in predicciones.get('pale', []):
                    nums = pred.get('numeros', [0, 1])
                    metodo = pred.get('metodo_generacion', 'unknown')
                    if not metodo or metodo == 'unknown':
                        metodo = 'fallback_statistical'
                        
                    pale = PrediccionPale(
                        fecha_prediccion=fecha,
                        tipo_loteria_id=tipo_loteria_id,
                        posicion=pred.get('posicion', 1),
                        numero_1=nums[0] if len(nums) > 0 else 0,
                        numero_2=nums[1] if len(nums) > 1 else 1,
                        probabilidad=pred.get('probabilidad', 0.01),
                        metodo_generacion=metodo,  # CORRECCIÓN: Usar método verificado
                        score_confianza=pred.get('score_confianza', 0.1)
                    )
                    session.add(pale)
                    logger.info(f"[DB] Inserted pale P{pred.get('posicion', 1)} with method: {metodo}")
                
                # Insert tripleta predictions
                for pred in predicciones.get('tripleta', []):
                    nums = pred.get('numeros', [0, 1, 2])
                    metodo = pred.get('metodo_generacion', 'unknown')
                    if not metodo or metodo == 'unknown':
                        metodo = 'fallback_statistical'
                        
                    tripleta = PrediccionTripleta(
                        fecha_prediccion=fecha,
                        tipo_loteria_id=tipo_loteria_id,
                        posicion=pred.get('posicion', 1),
                        numero_1=nums[0] if len(nums) > 0 else 0,
                        numero_2=nums[1] if len(nums) > 1 else 1,
                        numero_3=nums[2] if len(nums) > 2 else 2,
                        probabilidad=pred.get('probabilidad', 0.01),
                        metodo_generacion=metodo,  # CORRECCIÓN: Usar método verificado
                        score_confianza=pred.get('score_confianza', 0.1)
                    )
                    session.add(tripleta)
                    logger.info(f"[DB] Inserted tripleta P{pred.get('posicion', 1)} with method: {metodo}")
                
                session.commit()
                logger.info("[DB] Predictions inserted successfully")
                
        except Exception as e:
            logger.error(f"[DB] Failed to insert predictions: {e}")
            raise
    
    def obtener_predicciones_hoy(self, tipo_loteria_id: int) -> Dict[str, Any]:
        """Get today's predictions from database with caching."""
        try:
            today = date.today()
            cache_key = f"predictions_today_{tipo_loteria_id}_{today.isoformat()}"
            
            # Check cache
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                if time.time() - cache_time < 300:  # 5 minute cache
                    return cached_data
            
            with get_db_connection() as session:
                # Get all predictions in a single query each
                quinielas = session.query(PrediccionQuiniela).filter(
                    PrediccionQuiniela.fecha_prediccion == today,
                    PrediccionQuiniela.tipo_loteria_id == tipo_loteria_id
                ).order_by(PrediccionQuiniela.posicion).all()
                
                pales = session.query(PrediccionPale).filter(
                    PrediccionPale.fecha_prediccion == today,
                    PrediccionPale.tipo_loteria_id == tipo_loteria_id
                ).order_by(PrediccionPale.posicion).all()
                
                tripletas = session.query(PrediccionTripleta).filter(
                    PrediccionTripleta.fecha_prediccion == today,
                    PrediccionTripleta.tipo_loteria_id == tipo_loteria_id
                ).order_by(PrediccionTripleta.posicion).all()
                
                # Format results
                result = {
                    'fecha': today.strftime('%Y-%m-%d'),
                    'tipo_loteria_id': tipo_loteria_id,
                    'quiniela': [
                        {
                            'posicion': q.posicion,
                            'numero': q.numero_predicho,
                            'probabilidad': q.probabilidad,
                            'metodo_generacion': q.metodo_generacion,
                            'score_confianza': q.score_confianza
                        } for q in quinielas
                    ],
                    'pale': [
                        {
                            'posicion': p.posicion,
                            'numeros': [p.numero_1, p.numero_2],
                            'probabilidad': p.probabilidad,
                            'metodo_generacion': p.metodo_generacion,
                            'score_confianza': p.score_confianza
                        } for p in pales
                    ],
                    'tripleta': [
                        {
                            'posicion': t.posicion,
                            'numeros': [t.numero_1, t.numero_2, t.numero_3],
                            'probabilidad': t.probabilidad,
                            'metodo_generacion': t.metodo_generacion,
                            'score_confianza': t.score_confianza
                        } for t in tripletas
                    ]
                }
                
                # Cache result
                self.data_cache[cache_key] = (time.time(), result)
                
                return result
                
        except Exception as e:
            logger.error(f"[DB] Failed to get today's predictions: {e}")
            return {}
    
    def limpiar_cache_modelos(self, dias_antiguedad: int = None) -> int:
        """Clean old model cache files with parallel processing."""
        try:
            dias_antiguedad = dias_antiguedad or getattr(settings, 'model_cache_days', 7)
            cutoff_timestamp = time.time() - (dias_antiguedad * 24 * 3600)
            
            removed_count = 0
            files_to_remove = []
            
            # Collect files to remove
            for filename in os.listdir(self.model_cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.model_cache_dir, filename)
                    try:
                        if os.path.getmtime(file_path) < cutoff_timestamp:
                            files_to_remove.append(file_path)
                    except OSError:
                        continue
            
            # Remove files in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(os.remove, file_path) for file_path in files_to_remove]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"[CACHE] Failed to remove file: {e}")
            
            # Clear in-memory cache
            current_time = time.time()
            expired_keys = [
                key for key, (cache_time, _) in self.data_cache.items()
                if current_time - cache_time > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.data_cache[key]
            
            logger.info(f"[CACHE] Cleaned {removed_count} old model files and {len(expired_keys)} cache entries")
            return removed_count
            
        except Exception as e:
            logger.error(f"[CACHE] Cache cleanup failed: {e}")
            return 0
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        try:
            self._initialize_models()
            
            summary = {}
            
            for model_name in self.models.keys():
                try:
                    performance = model_evaluator.get_model_performance(model_name)
                    model_meta = model_registry.get_metadata(model_name)
                    
                    summary[model_name] = {
                        'performance': performance,
                        'metadata': {
                            'description': model_meta.description if model_meta else '',
                            'version': model_meta.version if model_meta else '',
                            'is_active': model_meta.is_active if model_meta else True
                        },
                        'cache_info': self._get_model_cache_info(model_name)
                    }
                except Exception as e:
                    logger.warning(f"[SUMMARY] Failed to get info for {model_name}: {e}")
                    summary[model_name] = {'error': str(e)}
            
            return summary
            
        except Exception as e:
            logger.error(f"[SUMMARY] Failed to get model performance summary: {e}")
            return {}
    
    def _get_model_cache_info(self, model_name: str) -> Dict[str, Any]:
        """Get cache information for a specific model."""
        try:
            cache_files = [
                f for f in os.listdir(self.model_cache_dir)
                if f.startswith(f"model_") and f.endswith('.pkl')
            ]
            
            model_files = 0
            total_size = 0
            
            for filename in cache_files:
                file_path = os.path.join(self.model_cache_dir, filename)
                try:
                    # Load model to check if it matches
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    if hasattr(model, 'model_name') and model.model_name == model_name:
                        model_files += 1
                        total_size += os.path.getsize(file_path)
                except:
                    continue
            
            return {
                'cached_files': model_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def procesar_fechas_pendientes(self, fecha_inicio: date = None, fecha_fin: date = None) -> Dict[str, Any]:
        """Process pending dates in optimized batches."""
        try:
            if not fecha_inicio:
                fecha_inicio = date.today() - timedelta(days=30)
            if not fecha_fin:
                fecha_fin = date.today()
            
            logger.info(f"[BATCH] Processing dates from {fecha_inicio} to {fecha_fin}")
            
            # Get all lottery types
            with get_db_connection() as session:
                lottery_types = session.query(TipoLoteria).filter(TipoLoteria.activo == True).all()
                lottery_type_ids = [lt.id for lt in lottery_types]
            
            # Generate date range
            current_date = fecha_inicio
            dates_to_process = []
            
            while current_date <= fecha_fin:
                dates_to_process.append(current_date)
                current_date += timedelta(days=1)
            
            logger.info(f"[BATCH] Processing {len(dates_to_process)} dates for {len(lottery_type_ids)} lottery types")
            
            # Process in chunks to avoid memory issues
            results = {'processed': 0, 'failed': 0, 'dates': {}}
            
            for i in range(0, len(dates_to_process), self.chunk_size):
                chunk = dates_to_process[i:i + self.chunk_size]
                
                logger.info(f"[BATCH] Processing chunk {i//self.chunk_size + 1}: {len(chunk)} dates")
                
                for fecha in chunk:
                    fecha_str = fecha.strftime('%Y-%m-%d')
                    
                    try:
                        # Train models for this date
                        training_result = self.entrenar_modelos(fecha)
                        
                        # Generate predictions for each lottery type
                        for lottery_type_id in lottery_type_ids:
                            predictions = self.generar_predicciones_diarias(fecha_str, lottery_type_id)
                            self.insertar_predicciones_en_bd(predictions, fecha, lottery_type_id)
                        
                        results['processed'] += 1
                        results['dates'][fecha_str] = 'success'
                        
                        # Log progress
                        if results['processed'] % 10 == 0:
                            logger.info(f"[BATCH] Progress: {results['processed']}/{len(dates_to_process)} dates processed")
                        
                    except Exception as e:
                        logger.error(f"[BATCH] Failed to process {fecha_str}: {e}")
                        results['failed'] += 1
                        results['dates'][fecha_str] = f'failed: {str(e)}'
                
                # Clean cache between chunks
                if i % (self.chunk_size * 2) == 0:
                    self.data_cache.clear()
                    logger.info("[BATCH] Cache cleared between chunks")
            
            logger.info(f"[BATCH] Completed: {results['processed']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"[BATCH] Batch processing failed: {e}")
            return {'processed': 0, 'failed': 0, 'error': str(e)}


# Global predictor engine instance (singleton)
predictor_engine = OptimizedPredictorEngine()