#!/usr/bin/env python3
"""
Single Command Predictor - Sistema optimizado para datos históricos masivos (1960-2025)
Entrena modelos y genera predicciones en un solo comando.

Usage:
    python scripts/train_and_predict.py
    python scripts/train_and_predict.py --date 2025-08-23 --lottery-types 1,2
    python scripts/train_and_predict.py --retrain --use-optimal-window
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import date, datetime, timedelta
import logging
import time
import gc
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import get_db_connection, check_database_connection
from config.settings import settings
from predictions.predictor_engine import predictor_engine
from models.database_models import Sorteo, TipoLoteria, PrediccionQuiniela
from sqlalchemy import func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/train_predict.log')
    ]
)
logger = logging.getLogger(__name__)


class OptimalPredictor:
    """Predictor optimizado para datos históricos masivos."""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'data_analysis_time': 0,
            'training_time': 0,
            'prediction_time': 0,
            'total_records_analyzed': 0,
            'models_trained': 0,
            'predictions_generated': 0
        }
    
    def analyze_historical_data(self, lottery_type_id: int) -> Dict[str, Any]:
        """Analiza datos históricos para determinar ventana óptima de entrenamiento."""
        logger.info(f"[ANALYZE] Analyzing historical data for lottery type {lottery_type_id}")
        start_time = time.time()
        
        try:
            with get_db_connection() as session:
                # Análisis básico de datos
                total_records = session.query(Sorteo).filter(
                    Sorteo.tipo_loteria_id == lottery_type_id
                ).count()
                
                # Fechas extremas
                fecha_min = session.query(func.min(Sorteo.fecha)).filter(
                    Sorteo.tipo_loteria_id == lottery_type_id
                ).scalar()
                
                fecha_max = session.query(func.max(Sorteo.fecha)).filter(
                    Sorteo.tipo_loteria_id == lottery_type_id
                ).scalar()
                
                # Análisis de calidad por períodos
                periods = self._analyze_data_quality_by_periods(session, lottery_type_id)
                
                # Determinar ventana óptima
                optimal_window = self._calculate_optimal_window(
                    total_records, fecha_min, fecha_max, periods
                )
                
                analysis = {
                    'total_records': total_records,
                    'date_range': (fecha_min, fecha_max),
                    'total_years': (fecha_max - fecha_min).days / 365.25 if fecha_max and fecha_min else 0,
                    'periods_analysis': periods,
                    'optimal_window_days': optimal_window['days'],
                    'optimal_start_date': optimal_window['start_date'],
                    'recommended_models': optimal_window['recommended_models'],
                    'data_quality_score': optimal_window['quality_score']
                }
                
                self.stats['data_analysis_time'] = time.time() - start_time
                self.stats['total_records_analyzed'] = total_records
                
                logger.info(f"[ANALYZE] Data analysis completed:")
                logger.info(f"  📊 Total records: {total_records:,}")
                logger.info(f"  📅 Date range: {fecha_min} - {fecha_max}")
                logger.info(f"  🎯 Optimal window: {optimal_window['days']} days")
                logger.info(f"  📈 Quality score: {optimal_window['quality_score']:.2f}/1.0")
                
                return analysis
                
        except Exception as e:
            logger.error(f"[ANALYZE] Data analysis failed: {e}")
            return self._get_default_analysis()
    
    def _analyze_data_quality_by_periods(self, session, lottery_type_id: int) -> List[Dict]:
        """Analiza calidad de datos por períodos históricos."""
        periods = [
            ('1960-1980', '1960-01-01', '1980-12-31'),
            ('1981-2000', '1981-01-01', '2000-12-31'), 
            ('2001-2010', '2001-01-01', '2010-12-31'),
            ('2011-2020', '2011-01-01', '2020-12-31'),
            ('2021-2025', '2021-01-01', '2025-12-31')
        ]
        
        period_analysis = []
        
        for period_name, start_str, end_str in periods:
            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
            
            count = session.query(Sorteo).filter(
                Sorteo.tipo_loteria_id == lottery_type_id,
                Sorteo.fecha >= start_date,
                Sorteo.fecha <= end_date
            ).count()
            
            # Análisis de consistencia (gaps en fechas)
            if count > 0:
                expected_days = (min(end_date, date.today()) - start_date).days + 1
                consistency = count / expected_days if expected_days > 0 else 0
            else:
                consistency = 0
            
            period_analysis.append({
                'period': period_name,
                'records': count,
                'consistency': consistency,
                'quality_score': min(consistency * 2, 1.0)  # Normalize to 0-1
            })
            
            logger.debug(f"  Period {period_name}: {count} records, consistency: {consistency:.2f}")
        
        return period_analysis
    
    def _calculate_optimal_window(self, total_records: int, fecha_min: date, 
                                fecha_max: date, periods: List[Dict]) -> Dict[str, Any]:
        """Calcula la ventana óptima de entrenamiento basada en análisis de datos."""
        try:
            # Reglas para ventana óptima con datos desde 1960
            if total_records > 10000:  # Datos abundantes
                # Usar últimos 5-7 años para máxima relevancia
                optimal_days = 2190  # 6 años
                recommended_models = [
                    'lightgbm', 'xgboost', 'neural_network', 'frequency_analysis',
                    'random_forest', 'ensemble_ml', 'arima_lstm'
                ]
                quality_score = 0.95
                
            elif total_records > 5000:  # Datos moderados
                # Usar últimos 3-4 años
                optimal_days = 1460  # 4 años
                recommended_models = [
                    'lightgbm', 'frequency_analysis', 'random_forest', 
                    'neural_network', 'statistical'
                ]
                quality_score = 0.85
                
            elif total_records > 1000:  # Datos limitados
                # Usar últimos 2 años o todos los datos disponibles
                optimal_days = 730  # 2 años
                recommended_models = [
                    'frequency_analysis', 'statistical', 'lightgbm', 'monte_carlo'
                ]
                quality_score = 0.75
                
            else:  # Datos muy limitados
                # Usar todos los datos disponibles
                total_days = (fecha_max - fecha_min).days if fecha_max and fecha_min else 365
                optimal_days = min(total_days, 365)  # Máximo 1 año
                recommended_models = [
                    'frequency_analysis', 'statistical', 'monte_carlo'
                ]
                quality_score = 0.60
            
            # Ajustar según calidad de períodos recientes
            recent_periods = [p for p in periods if '2020' in p['period'] or '2021' in p['period']]
            if recent_periods:
                avg_recent_quality = sum(p['quality_score'] for p in recent_periods) / len(recent_periods)
                quality_score *= avg_recent_quality
            
            # Calcular fecha de inicio
            start_date = fecha_max - timedelta(days=optimal_days) if fecha_max else date.today() - timedelta(days=optimal_days)
            
            return {
                'days': optimal_days,
                'start_date': start_date,
                'recommended_models': recommended_models,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal window: {e}")
            return self._get_default_window()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Análisis por defecto cuando falla el análisis principal."""
        return {
            'total_records': 0,
            'date_range': (None, None),
            'total_years': 0,
            'optimal_window_days': 1095,  # 3 años por defecto
            'optimal_start_date': date.today() - timedelta(days=1095),
            'recommended_models': ['frequency_analysis', 'statistical', 'lightgbm'],
            'data_quality_score': 0.5
        }
    
    def _get_default_window(self) -> Dict[str, Any]:
        """Ventana por defecto."""
        return {
            'days': 1095,  # 3 años
            'start_date': date.today() - timedelta(days=1095),
            'recommended_models': ['frequency_analysis', 'statistical', 'lightgbm'],
            'quality_score': 0.5
        }
    
    def train_optimized_models(self, lottery_type_id: int, analysis: Dict[str, Any], 
                             force_retrain: bool = False) -> Dict[str, Any]:
        """Entrena modelos usando configuración optimizada."""
        logger.info(f"[TRAIN] Starting optimized training for lottery type {lottery_type_id}")
        start_time = time.time()
        
        try:
            # Verificar si ya hay modelos entrenados recientemente
            if not force_retrain and self._has_recent_models(lottery_type_id):
                logger.info("[TRAIN] Recent models found, skipping training (use --retrain to force)")
                return {'status': 'skipped', 'reason': 'recent_models_exist'}
            
            # Configurar parámetros optimizados
            self._configure_optimal_settings(analysis)
            
            # Entrenar solo modelos recomendados
            recommended_models = analysis.get('recommended_models', ['frequency_analysis'])
            logger.info(f"[TRAIN] Training {len(recommended_models)} recommended models: {recommended_models}")
            
            # Entrenar modelos
            training_results = predictor_engine.entrenar_modelos(
                fecha=date.today(),
                tipo_loteria_id=lottery_type_id
            )
            
            self.stats['training_time'] = time.time() - start_time
            self.stats['models_trained'] = len(recommended_models)
            
            logger.info(f"[TRAIN] Training completed in {self.stats['training_time']:.1f}s")
            logger.info(f"[TRAIN] {self.stats['models_trained']} models trained")
            
            return {
                'status': 'success',
                'models_trained': self.stats['models_trained'],
                'training_time': self.stats['training_time'],
                'results': training_results
            }
            
        except Exception as e:
            logger.error(f"[TRAIN] Training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _has_recent_models(self, lottery_type_id: int) -> bool:
        """Verifica si hay modelos entrenados recientemente."""
        try:
            # Verificar si hay predicciones recientes (último indicador de modelos entrenados)
            with get_db_connection() as session:
                recent_prediction = session.query(PrediccionQuiniela).filter(
                    PrediccionQuiniela.tipo_loteria_id == lottery_type_id,
                    PrediccionQuiniela.fecha_prediccion >= date.today() - timedelta(days=1)
                ).first()
                
                return recent_prediction is not None
        except Exception:
            return False
    
    def _configure_optimal_settings(self, analysis: Dict[str, Any]) -> None:
        """Configura settings optimizados basados en análisis."""
        try:
            # Configurar ventana histórica
            optimal_window = analysis.get('optimal_window_days', 1095)
            
            # Ajustar configuraciones en predictor_engine
            if hasattr(predictor_engine, 'configure_training_window'):
                predictor_engine.configure_training_window(optimal_window)
            
            # Configurar modelos prioritarios
            recommended_models = analysis.get('recommended_models', [])
            if hasattr(predictor_engine, 'set_priority_models'):
                predictor_engine.set_priority_models(recommended_models)
            
            logger.info(f"[CONFIG] Configured optimal window: {optimal_window} days")
            logger.info(f"[CONFIG] Priority models: {recommended_models}")
            
        except Exception as e:
            logger.warning(f"[CONFIG] Failed to configure optimal settings: {e}")
    
    def generate_predictions(self, prediction_date: date, lottery_types: List[int]) -> Dict[str, Any]:
        """Genera predicciones para la fecha especificada."""
        logger.info(f"[PREDICT] Generating predictions for {prediction_date}")
        start_time = time.time()
        
        results = {}
        
        for lottery_type_id in lottery_types:
            try:
                logger.info(f"[PREDICT] Processing lottery type {lottery_type_id}")
                
                # Generar predicciones
                predictions = predictor_engine.generar_predicciones_diarias(
                    prediction_date.strftime('%Y-%m-%d'), 
                    lottery_type_id
                )
                
                if predictions:
                    # Guardar en base de datos
                    predictor_engine.insertar_predicciones_en_bd(
                        predictions, prediction_date, lottery_type_id
                    )
                    
                    results[f'lottery_{lottery_type_id}'] = {
                        'status': 'success',
                        'predictions': predictions
                    }
                    
                    self.stats['predictions_generated'] += 1
                    
                    logger.info(f"[PREDICT] Successfully generated predictions for lottery {lottery_type_id}")
                    
                    # Mostrar predicciones generadas
                    self._display_predictions(lottery_type_id, predictions)
                    
                else:
                    logger.warning(f"[PREDICT] No predictions generated for lottery {lottery_type_id}")
                    results[f'lottery_{lottery_type_id}'] = {
                        'status': 'failed',
                        'error': 'no_predictions_generated'
                    }
                
            except Exception as e:
                logger.error(f"[PREDICT] Failed to generate predictions for lottery {lottery_type_id}: {e}")
                results[f'lottery_{lottery_type_id}'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.stats['prediction_time'] = time.time() - start_time
        logger.info(f"[PREDICT] Prediction generation completed in {self.stats['prediction_time']:.1f}s")
        
        return results
    
    def _display_predictions(self, lottery_type_id: int, predictions: Dict[str, Any]) -> None:
        """Muestra las predicciones generadas de forma legible."""
        lottery_name = "Lotería Nacional" if lottery_type_id == 1 else "Gana Más"
        
        logger.info(f"🎲 PREDICCIONES GENERADAS - {lottery_name}")
        logger.info("=" * 50)
        
        for game_type, preds in predictions.items():
            game_name = game_type.upper()
            logger.info(f"🎯 {game_name}:")
            
            for pred in preds:
                position = pred.get('posicion', '?')
                method = pred.get('metodo_generacion', 'unknown')
                prob = pred.get('probabilidad', 0) * 100
                conf = pred.get('score_confianza', 0) * 100
                
                if game_type == 'quiniela':
                    number = pred.get('numero', '??')
                    logger.info(f"  P{position}: {number:02d} | {method} | Prob: {prob:.1f}% | Conf: {conf:.1f}%")
                else:
                    numbers = pred.get('numeros', [])
                    numbers_str = '-'.join([f"{n:02d}" for n in numbers])
                    logger.info(f"  P{position}: {numbers_str} | {method} | Prob: {prob:.1f}% | Conf: {conf:.1f}%")
        
        logger.info("=" * 50)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento."""
        total_time = time.time() - self.start_time
        
        return {
            **self.stats,
            'total_execution_time': total_time,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calcula score de eficiencia basado en tiempo y resultados."""
        try:
            total_time = time.time() - self.start_time
            
            # Factores de eficiencia
            time_factor = max(0, 1 - (total_time / 300))  # Penalizar si toma más de 5 minutos
            success_factor = self.stats['predictions_generated'] / max(1, self.stats['models_trained'])
            
            return (time_factor + success_factor) / 2
        except:
            return 0.5


def main():
    """Main function for single command predictor."""
    parser = argparse.ArgumentParser(
        description="Single Command Predictor - Train models and generate predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - train and predict for today
    python scripts/train_and_predict.py
    
    # Specify date and lottery types
    python scripts/train_and_predict.py --date 2025-08-23 --lottery-types 1,2
    
    # Force retraining and use optimal data window
    python scripts/train_and_predict.py --retrain --use-optimal-window
    
    # Analysis only mode
    python scripts/train_and_predict.py --analyze-only
        """
    )
    
    parser.add_argument('--date', type=str, default=None,
                       help='Prediction date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--lottery-types', type=str, default='1,2',
                       help='Comma-separated lottery type IDs (default: 1,2)')
    parser.add_argument('--retrain', action='store_true',
                       help='Force model retraining even if recent models exist')
    parser.add_argument('--use-optimal-window', action='store_true',
                       help='Use optimal data window based on historical analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only perform data analysis without training/predicting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse date
    if args.date:
        try:
            prediction_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    else:
        prediction_date = date.today()
    
    # Parse lottery types
    try:
        lottery_types = [int(x.strip()) for x in args.lottery_types.split(',')]
    except ValueError:
        logger.error(f"Invalid lottery types: {args.lottery_types}")
        return 1
    
    # Check database connection
    logger.info("🔍 Checking database connection...")
    if not check_database_connection():
        logger.error("❌ Database connection failed")
        return 1
    
    # Initialize predictor
    logger.info("🚀 Initializing Optimal Predictor System")
    logger.info(f"📅 Target date: {prediction_date}")
    logger.info(f"🎲 Lottery types: {lottery_types}")
    
    predictor = OptimalPredictor()
    
    try:
        # Step 1: Analyze historical data for each lottery type
        all_analyses = {}
        for lottery_type_id in lottery_types:
            logger.info(f"\n📊 ANALYZING LOTTERY TYPE {lottery_type_id}")
            analysis = predictor.analyze_historical_data(lottery_type_id)
            all_analyses[lottery_type_id] = analysis
            
            if args.analyze_only:
                continue
            
            # Step 2: Train optimized models
            logger.info(f"\n🧠 TRAINING MODELS FOR LOTTERY TYPE {lottery_type_id}")
            training_result = predictor.train_optimized_models(
                lottery_type_id, analysis, args.retrain
            )
            
            if training_result['status'] != 'success' and training_result['status'] != 'skipped':
                logger.error(f"❌ Training failed for lottery type {lottery_type_id}")
                continue
        
        if args.analyze_only:
            logger.info("\n📈 ANALYSIS COMPLETE")
            return 0
        
        # Step 3: Generate predictions
        logger.info(f"\n🔮 GENERATING PREDICTIONS FOR {prediction_date}")
        prediction_results = predictor.generate_predictions(prediction_date, lottery_types)
        
        # Step 4: Show final results
        logger.info("\n" + "="*60)
        logger.info("🎉 EXECUTION COMPLETED")
        logger.info("="*60)
        
        stats = predictor.get_performance_stats()
        
        logger.info(f"📊 Performance Statistics:")
        logger.info(f"  ⏱️  Total time: {stats['total_execution_time']:.1f}s")
        logger.info(f"  📈 Records analyzed: {stats['total_records_analyzed']:,}")
        logger.info(f"  🧠 Models trained: {stats['models_trained']}")
        logger.info(f"  🔮 Predictions generated: {stats['predictions_generated']}")
        logger.info(f"  ⚡ Efficiency score: {stats['efficiency_score']:.2f}")
        
        # Show prediction summary
        successful_lotteries = [k for k, v in prediction_results.items() if v['status'] == 'success']
        logger.info(f"\n✅ Successfully generated predictions for {len(successful_lotteries)} lottery types")
        
        # API endpoints
        logger.info(f"\n🌐 View predictions at:")
        for lottery_type_id in lottery_types:
            logger.info(f"  http://localhost:8000/predicciones/hoy/{lottery_type_id}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Execution failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    sys.exit(main())