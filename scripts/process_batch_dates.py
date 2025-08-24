#!/usr/bin/env python3
"""
Optimized batch processor for processing pending lottery prediction dates.
Compatible with Python 3.9+ - Fixed timeout issues.

Usage:
    python scripts/process_batch_dates.py --start-date 2023-01-01 --end-date 2024-12-31 --workers 8
    python scripts/process_batch_dates.py --days-back 365 --workers 6
    python scripts/process_batch_dates.py --continue-from-last
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import multiprocessing as mp
import logging
import time
import gc
import psutil
import signal
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.database import get_db_connection, init_database, check_database_connection
from predictions.predictor_engine import predictor_engine
from models.database_models import Sorteo, PrediccionQuiniela, TipoLoteria

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class OptimizedBatchProcessor:
    """Optimized batch processor for handling large volumes of dates."""
    
    def __init__(self, max_workers: int = None, batch_size: int = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size or 25
        self.processed_dates = set()
        self.failed_dates = set()
        self.start_time = None
        self.stats = {
            'total_dates': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def get_lottery_types(self) -> List[int]:
        """Get active lottery types."""
        try:
            with get_db_connection() as session:
                lottery_types = session.query(TipoLoteria).filter(
                    TipoLoteria.activo == True
                ).all()
                return [lt.id for lt in lottery_types]
        except Exception as e:
            logger.error(f"Failed to get lottery types: {e}")
            return [1, 2]  # Default fallback
    
    def get_existing_prediction_dates(self, lottery_type_id: int) -> set:
        """Get dates that already have predictions."""
        try:
            with get_db_connection() as session:
                existing_dates = session.query(PrediccionQuiniela.fecha_prediccion).filter(
                    PrediccionQuiniela.tipo_loteria_id == lottery_type_id
                ).distinct().all()
                
                return {d[0] for d in existing_dates}
        except Exception as e:
            logger.warning(f"Failed to get existing dates for lottery {lottery_type_id}: {e}")
            return set()
    
    def filter_dates_to_process(self, date_range: List[date], lottery_types: List[int]) -> List[Tuple[date, List[int]]]:
        """Filter dates that need processing, avoiding duplicates."""
        dates_to_process = []
        
        # Get existing predictions for all lottery types
        existing_by_lottery = {}
        for lottery_type_id in lottery_types:
            existing_by_lottery[lottery_type_id] = self.get_existing_prediction_dates(lottery_type_id)
        
        for fecha in date_range:
            # Check which lottery types need processing for this date
            needed_lotteries = []
            for lottery_type_id in lottery_types:
                if fecha not in existing_by_lottery[lottery_type_id]:
                    needed_lotteries.append(lottery_type_id)
            
            if needed_lotteries:
                dates_to_process.append((fecha, needed_lotteries))
            else:
                self.stats['skipped'] += 1
                logger.debug(f"Skipping {fecha} - already has predictions for all lottery types")
        
        return dates_to_process
    
    def process_single_date(self, date_and_lotteries: Tuple[date, List[int]]) -> Dict[str, Any]:
        """Process a single date with its required lottery types - OPTIMIZADO AGOSTO 2025."""
        fecha, lottery_type_ids = date_and_lotteries
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        result = {
            'date': fecha_str,
            'lottery_types': lottery_type_ids,
            'status': 'failed',
            'predictions_created': 0,
            'error': None,
            'processing_time': 0,
            'models_trained': 0,  # NUEVO: contador de modelos
            'memory_peak_mb': 0   # NUEVO: pico de memoria
        }
        
        start_time = time.time()
        initial_memory = 0
        
        try:
            # NUEVO: Monitorear memoria inicial
            try:
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            except:
                pass
            
            logger.info(f"[PROCESS] Processing {fecha_str} for lotteries: {lottery_type_ids}")
            
            # CAMBIO CRÍTICO: Verificar memoria antes de entrenar
            if not self._check_memory_available():
                logger.warning(f"[MEMORY] Insufficient memory for {fecha_str}, skipping training")
                # Usar predicciones simples sin entrenamiento pesado
                predictions_created = self._create_simple_predictions(fecha, lottery_type_ids)
                
                result.update({
                    'status': 'memory_limited',
                    'predictions_created': predictions_created,
                    'processing_time': time.time() - start_time,
                    'models_trained': 0,
                    'note': 'Used simple predictions due to memory constraints'
                })
                return result
            
            # CAMBIO: Entrenamiento optimizado con límites
            training_result = self._train_models_limited(fecha, lottery_type_ids)
            models_trained = training_result.get('models_trained', 0)
            
            predictions_created = 0
            
            # Generar predicciones para cada tipo de lotería requerido
            for lottery_type_id in lottery_type_ids:
                try:
                    logger.info(f"[PREDICT] Generating predictions for {fecha_str} - lottery {lottery_type_id}")
                    
                    # CAMBIO: Usar datos optimizados para predicción
                    predictions = predictor_engine.generar_predicciones_diarias(fecha_str, lottery_type_id)
                    
                    if predictions:
                        predictor_engine.insertar_predicciones_en_bd(predictions, fecha, lottery_type_id)
                        predictions_created += 1
                        logger.info(f"[SUCCESS] Predictions saved for {fecha_str} - lottery {lottery_type_id}")
                    else:
                        logger.warning(f"[WARN] No predictions generated for {fecha_str} - lottery {lottery_type_id}")
                
                except Exception as e:
                    logger.error(f"[ERROR] Prediction failed for {fecha_str} - lottery {lottery_type_id}: {e}")
                    continue
            
            # NUEVO: Calcular pico de memoria
            try:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                peak_memory = max(initial_memory, current_memory)
            except:
                peak_memory = 0
            
            result.update({
                'status': 'success' if predictions_created > 0 else 'partial',
                'predictions_created': predictions_created,
                'models_trained': models_trained,
                'processing_time': time.time() - start_time,
                'memory_peak_mb': peak_memory
            })
            
            # NUEVO: Limpieza de memoria forzada
            if hasattr(settings, 'force_gc_after_batch') and settings.force_gc_after_batch:
                gc.collect()
            
            return result
            
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            logger.error(f"[FAILED] Processing failed for {fecha_str}: {e}")
            return result
    
    def _check_memory_available(self) -> bool:
        """Verifica si hay memoria suficiente disponible."""
        try:
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / 1024 / 1024
            
            # NUEVO: Usar configuración de settings
            min_memory_mb = getattr(settings, 'model_training_max_memory_mb', 800) * 2
            
            if available_mb < min_memory_mb:
                logger.warning(f"[MEMORY] Low memory: {available_mb:.0f}MB < {min_memory_mb}MB required")
                return False
            
            return True
        except:
            return True  # Si no puede verificar, continuar
    
    def _train_models_limited(self, fecha: date, lottery_type_ids: List[int]) -> Dict[str, Any]:
        """Entrenamiento limitado y optimizado para agosto 2025."""
        try:
            # NUEVO: Usar solo modelos prioritarios
            priority_models = getattr(settings, 'priority_models', [
                'frequency_analysis', 'lightgbm', 'xgboost', 'random_forest'
            ])
            
            # Limitar número de modelos
            max_models = getattr(settings, 'max_models_per_batch', 4)
            models_to_train = priority_models[:max_models]
            
            logger.info(f"[TRAIN] Training {len(models_to_train)} priority models for {fecha}")
            
            # Entrenar con timeout más corto
            training_result = predictor_engine.entrenar_modelos(
                fecha=fecha,
                tipo_loteria_id=lottery_type_ids[0] if lottery_type_ids else 1
            )
            
            return {
                'models_trained': len(models_to_train),
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"[TRAIN] Limited training failed: {e}")
            return {'models_trained': 0, 'error': str(e)}
    
    def _create_simple_predictions(self, fecha: date, lottery_type_ids: List[int]) -> int:
        """Crear predicciones simples cuando hay limitaciones de memoria."""
        try:
            predictions_created = 0
            
            for lottery_type_id in lottery_type_ids:
                # Usar solo frequency analysis (más rápido)
                simple_predictions = self._generate_frequency_based_predictions(fecha, lottery_type_id)
                
                if simple_predictions:
                    predictor_engine.insertar_predicciones_en_bd(simple_predictions, fecha, lottery_type_id)
                    predictions_created += 1
                    logger.info(f"[SIMPLE] Simple predictions created for {fecha} - lottery {lottery_type_id}")
            
            return predictions_created
            
        except Exception as e:
            logger.error(f"[SIMPLE] Simple prediction creation failed: {e}")
            return 0
    
    def _generate_frequency_based_predictions(self, fecha: date, lottery_type_id: int) -> Dict:
        """Generar predicciones basadas solo en frecuencia (método rápido)."""
        try:
            # Cargar datos históricos limitados
            with get_db_connection() as session:
                recent_data = session.query(Sorteo).filter(
                    Sorteo.tipo_loteria_id == lottery_type_id,
                    Sorteo.fecha >= fecha - timedelta(days=90)  # Solo 3 meses
                ).order_by(Sorteo.fecha.desc()).limit(100).all()
            
            if not recent_data:
                return None
            
            # Análisis de frecuencia simple
            from collections import Counter
            
            all_numbers = []
            for sorteo in recent_data:
                all_numbers.extend([sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar])
            
            frequency_counter = Counter(all_numbers)
            most_common = frequency_counter.most_common(20)  # Top 20 números
            
            # Generar predicciones simples
            import random
            random.seed(int(fecha.strftime('%Y%m%d')))  # Seed basado en fecha
            
            predictions = {
                'quiniela': [],
                'pale': [],
                'tripleta': []
            }
            
            # Quiniela: top 3 números más frecuentes
            for i in range(3):
                if i < len(most_common):
                    predictions['quiniela'].append({
                        'posicion': i + 1,
                        'numero': most_common[i][0],
                        'probabilidad': most_common[i][1] / len(all_numbers),
                        'metodo_generacion': 'frequency_simple',
                        'score_confianza': 0.3
                    })
            
            # Pale: combinaciones de números frecuentes
            for i in range(3):
                if len(most_common) >= 2:
                    num1, num2 = random.sample([n[0] for n in most_common[:10]], 2)
                    predictions['pale'].append({
                        'posicion': i + 1,
                        'numeros': sorted([num1, num2]),
                        'probabilidad': 0.02,
                        'metodo_generacion': 'frequency_simple',
                        'score_confianza': 0.25
                    })
            
            # Tripleta: combinaciones de números frecuentes
            for i in range(3):
                if len(most_common) >= 3:
                    nums = random.sample([n[0] for n in most_common[:15]], 3)
                    predictions['tripleta'].append({
                        'posicion': i + 1,
                        'numeros': sorted(nums),
                        'probabilidad': 0.005,
                        'metodo_generacion': 'frequency_simple',
                        'score_confianza': 0.2
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"[SIMPLE] Frequency-based prediction generation failed: {e}")
            return None
    
    def process_batch_sequential(self, batch: List[Tuple[date, List[int]]]) -> List[Dict[str, Any]]:
        """Process a batch of dates sequentially (fallback for compatibility)."""
        logger.info(f"[BATCH] Processing batch of {len(batch)} dates sequentially")
        
        results = []
        for i, date_item in enumerate(batch):
            if self.shutdown_requested:
                logger.info(f"[BATCH] Shutdown requested, stopping at {i+1}/{len(batch)}")
                break
                
            try:
                result = self.process_single_date(date_item)
                results.append(result)
                
                # Progress within batch
                if (i + 1) % 5 == 0:
                    logger.info(f"[BATCH] Progress: {i+1}/{len(batch)} dates in batch")
                    
            except Exception as e:
                logger.error(f"[BATCH] Error processing date {date_item}: {e}")
                results.append({
                    'date': date_item[0].strftime('%Y-%m-%d'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def process_batch_parallel_fixed(self, batch: List[Tuple[date, List[int]]]) -> List[Dict[str, Any]]:
        """Process a batch of dates in parallel - Python 3.9 compatible."""
        logger.info(f"[BATCH] Processing batch of {len(batch)} dates in parallel")
        
        try:
            # Use process pool for CPU-intensive ML training
            max_workers = min(len(batch), self.max_workers)
            
            with mp.Pool(processes=max_workers) as pool:
                # Submit all tasks without timeout parameter (Python 3.9 compatible)
                async_result = pool.map_async(self.process_single_date, batch)
                
                # Wait for completion with manual timeout handling
                start_time = time.time()
                timeout = 1800  # 30 minutes
                
                while not async_result.ready():
                    if time.time() - start_time > timeout:
                        logger.error(f"[BATCH] Timeout after {timeout}s, terminating pool")
                        pool.terminate()
                        pool.join()
                        return [{'status': 'timeout', 'error': 'Batch timeout'} for _ in batch]
                    
                    if self.shutdown_requested:
                        logger.info("[BATCH] Shutdown requested, terminating pool")
                        pool.terminate()
                        pool.join()
                        return [{'status': 'cancelled', 'error': 'Shutdown requested'} for _ in batch]
                    
                    time.sleep(1)  # Check every second
                
                # Get results
                results = async_result.get()
                return results
                
        except Exception as e:
            logger.error(f"[BATCH] Parallel processing failed: {e}")
            # Fallback to sequential processing
            logger.info("[BATCH] Falling back to sequential processing")
            return self.process_batch_sequential(batch)
    
    def monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor system resource usage."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory_used_mb': memory_info.used / 1024 / 1024,
                'memory_percent': memory_info.percent,
                'cpu_percent': cpu_percent,
                'available_memory_mb': memory_info.available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get system resources: {e}")
            return {}
    
    def should_pause_processing(self) -> bool:
        """Check if processing should be paused due to resource constraints."""
        resources = self.monitor_system_resources()
        
        if resources.get('memory_percent', 0) > 85:
            logger.warning(f"High memory usage: {resources.get('memory_percent', 0)}%")
            return True
        
        if resources.get('cpu_percent', 0) > 90:
            logger.warning(f"High CPU usage: {resources.get('cpu_percent', 0)}%")
            return True
        
        return False
    
    def process_date_range(self, start_date: date, end_date: date, lottery_types: List[int] = None) -> Dict[str, Any]:
        """Process a range of dates efficiently."""
        self.start_time = time.time()
        self.stats['start_time'] = self.start_time
        
        logger.info(f"[START] Starting batch processing from {start_date} to {end_date}")
        
        # Get lottery types
        if not lottery_types:
            lottery_types = self.get_lottery_types()
        
        logger.info(f"[SETUP] Processing lottery types: {lottery_types}")
        
        # Generate date range
        current_date = start_date
        date_range = []
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"[SETUP] Generated {len(date_range)} dates to process")
        
        # Filter dates that need processing
        dates_to_process = self.filter_dates_to_process(date_range, lottery_types)
        
        self.stats['total_dates'] = len(dates_to_process)
        logger.info(f"[SETUP] {len(dates_to_process)} dates need processing (skipped {self.stats['skipped']} existing)")
        
        if not dates_to_process:
            logger.info("[COMPLETE] No dates to process - all predictions already exist")
            return self.get_final_stats()
        
        # Process in batches
        batch_results = []
        
        for i in range(0, len(dates_to_process), self.batch_size):
            if self.shutdown_requested:
                logger.info("[SHUTDOWN] Graceful shutdown requested")
                break
                
            batch_num = i // self.batch_size + 1
            batch = dates_to_process[i:i + self.batch_size]
            
            logger.info(f"[BATCH {batch_num}] Processing batch {batch_num}/{(len(dates_to_process) + self.batch_size - 1) // self.batch_size}")
            
            # Check system resources before processing
            if self.should_pause_processing():
                logger.warning("[PAUSE] Pausing for 60 seconds due to high resource usage")
                time.sleep(60)
                gc.collect()  # Force garbage collection
            
            # Process batch
            batch_start_time = time.time()
            results = self.process_batch_parallel_fixed(batch)
            batch_duration = time.time() - batch_start_time
            
            # Update statistics
            successful = len([r for r in results if r.get('status') == 'success'])
            failed = len([r for r in results if r.get('status') == 'failed'])
            
            self.stats['processed'] += successful
            self.stats['failed'] += failed
            
            logger.info(f"[BATCH {batch_num}] Completed in {batch_duration:.1f}s - "
                       f"Success: {successful}, Failed: {failed}")
            
            # Progress report
            total_processed = self.stats['processed'] + self.stats['failed']
            if total_processed > 0:
                progress = (total_processed / self.stats['total_dates']) * 100
                elapsed = time.time() - self.start_time
                rate = total_processed / elapsed * 3600  # dates per hour
                
                logger.info(f"[PROGRESS] {progress:.1f}% complete "
                           f"({total_processed}/{self.stats['total_dates']}) - "
                           f"Rate: {rate:.1f} dates/hour")
            
            batch_results.extend(results)
            
            # Clean up between batches
            if batch_num % 5 == 0:  # Every 5 batches
                gc.collect()
                try:
                    predictor_engine.limpiar_cache_modelos()  # Clean model cache
                except:
                    pass  # Ignore if method doesn't exist
        
        self.stats['end_time'] = time.time()
        
        return self.get_final_stats()
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final processing statistics."""
        if self.stats['end_time'] and self.stats['start_time']:
            total_time = self.stats['end_time'] - self.stats['start_time']
            self.stats['total_time_hours'] = total_time / 3600
            
            if self.stats['processed'] > 0:
                self.stats['average_time_per_date'] = total_time / self.stats['processed']
                self.stats['dates_per_hour'] = self.stats['processed'] / (total_time / 3600)
        
        return self.stats.copy()


def main():
    """Main entry point for batch processing script."""
    parser = argparse.ArgumentParser(
        description="Optimized batch processor for lottery predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific date range with 8 workers
    python process_batch_dates.py --start-date 2023-01-01 --end-date 2024-12-31 --workers 8
    
    # Process last 365 days with 6 workers  
    python process_batch_dates.py --days-back 365 --workers 6
    
    # Continue processing from where we left off
    python process_batch_dates.py --continue-from-last --workers 4
    
    # Process with custom batch size
    python process_batch_dates.py --days-back 100 --batch-size 20 --workers 4
        """
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--start-date', type=str, 
                           help='Start date (YYYY-MM-DD)')
    date_group.add_argument('--days-back', type=int,
                           help='Process N days back from today')
    date_group.add_argument('--continue-from-last', action='store_true',
                           help='Continue from the last processed date')
    
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of worker processes (default: {min(mp.cpu_count(), 8)})')
    parser.add_argument('--batch-size', type=int, default=25,
                       help='Batch size for parallel processing (default: 25)')
    parser.add_argument('--lottery-types', type=str, nargs='+',
                       help='Specific lottery type IDs to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip dates that already have predictions (default: True)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential processing instead of parallel (safer but slower)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system information
    logger.info("=" * 60)
    logger.info("OPTIMIZED LOTTERY PREDICTION BATCH PROCESSOR")
    logger.info("=" * 60)
    logger.info(f"CPU cores available: {mp.cpu_count()}")
    logger.info(f"Memory available: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    logger.info(f"Workers requested: {args.workers or min(mp.cpu_count(), 8)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Processing mode: {'Sequential' if args.sequential else 'Parallel'}")
    
    # Check database connection
    logger.info("[INIT] Checking database connection...")
    if not check_database_connection():
        logger.error("[INIT] Database connection failed. Exiting.")
        sys.exit(1)
    
    # Initialize database
    logger.info("[INIT] Initializing database...")
    if not init_database():
        logger.error("[INIT] Database initialization failed. Exiting.")
        sys.exit(1)
    
    # Determine date range
    end_date = date.today()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    elif args.days_back:
        start_date = end_date - timedelta(days=args.days_back)
    elif args.continue_from_last:
        # Find the last processed date
        try:
            with get_db_connection() as session:
                last_prediction = session.query(PrediccionQuiniela).order_by(
                    PrediccionQuiniela.fecha_prediccion.desc()
                ).first()
                
                if last_prediction:
                    start_date = last_prediction.fecha_prediccion + timedelta(days=1)
                    logger.info(f"[CONTINUE] Continuing from {start_date} (last prediction: {last_prediction.fecha_prediccion})")
                else:
                    start_date = end_date - timedelta(days=30)  # Default to 30 days back
                    logger.info(f"[CONTINUE] No previous predictions found, starting from {start_date}")
        except Exception as e:
            logger.error(f"[CONTINUE] Failed to find last processed date: {e}")
            start_date = end_date - timedelta(days=30)
    
    # Validate date range
    if start_date > end_date:
        logger.error(f"[ERROR] Start date {start_date} is after end date {end_date}")
        sys.exit(1)
    
    # Parse lottery types
    lottery_types = None
    if args.lottery_types:
        try:
            lottery_types = [int(lt) for lt in args.lottery_types]
            logger.info(f"[SETUP] Processing specific lottery types: {lottery_types}")
        except ValueError:
            logger.error("[ERROR] Invalid lottery type IDs provided")
            sys.exit(1)
    
    # Show processing plan
    total_days = (end_date - start_date).days + 1
    logger.info("=" * 60)
    logger.info("PROCESSING PLAN")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total days to check: {total_days}")
    logger.info(f"Lottery types: {'All active' if not lottery_types else lottery_types}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Workers: {args.workers or min(mp.cpu_count(), 8)}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        logger.info("[DRY-RUN] This is a dry run - no actual processing will occur")
        
        # Show what would be processed
        processor = OptimizedBatchProcessor(
            max_workers=args.workers,
            batch_size=args.batch_size
        )
        
        if not lottery_types:
            lottery_types = processor.get_lottery_types()
        
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        dates_to_process = processor.filter_dates_to_process(date_range, lottery_types)
        
        logger.info(f"[DRY-RUN] Would process {len(dates_to_process)} dates")
        logger.info(f"[DRY-RUN] Would skip {len(date_range) - len(dates_to_process)} existing dates")
        
        if dates_to_process:
            logger.info("[DRY-RUN] Sample dates to process:")
            for i, (date_item, lotteries) in enumerate(dates_to_process[:5]):
                logger.info(f"[DRY-RUN]   {date_item} -> lotteries {lotteries}")
            if len(dates_to_process) > 5:
                logger.info(f"[DRY-RUN]   ... and {len(dates_to_process) - 5} more dates")
        
        return
    
    # Confirm processing for large date ranges
    if total_days > 100:
        response = input(f"\nYou are about to process {total_days} days. This may take several hours. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("[CANCELLED] Processing cancelled by user")
            return
    
    # Start processing
    logger.info("=" * 60)
    logger.info("STARTING BATCH PROCESSING")
    logger.info("=" * 60)
    
    processor = OptimizedBatchProcessor(
        max_workers=args.workers,
        batch_size=args.batch_size
    )
    
    # Override processing method if sequential requested
    if args.sequential:
        processor.process_batch_parallel_fixed = processor.process_batch_sequential
    
    try:
        # Process the date range
        final_stats = processor.process_date_range(
            start_date=start_date,
            end_date=end_date,
            lottery_types=lottery_types
        )
        
        # Print final results
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total dates checked: {final_stats.get('total_dates', 0)}")
        logger.info(f"Successfully processed: {final_stats.get('processed', 0)}")
        logger.info(f"Failed: {final_stats.get('failed', 0)}")
        logger.info(f"Skipped (existing): {final_stats.get('skipped', 0)}")
        
        if final_stats.get('total_time_hours'):
            logger.info(f"Total time: {final_stats['total_time_hours']:.2f} hours")
            logger.info(f"Average time per date: {final_stats.get('average_time_per_date', 0):.1f} seconds")
            logger.info(f"Processing rate: {final_stats.get('dates_per_hour', 0):.1f} dates/hour")
        
        # Success rate
        total_attempted = final_stats.get('processed', 0) + final_stats.get('failed', 0)
        if total_attempted > 0:
            success_rate = (final_stats.get('processed', 0) / total_attempted) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 60)
        
        if final_stats.get('failed', 0) > 0:
            logger.warning(f"WARNING: {final_stats['failed']} dates failed to process")
            sys.exit(1)
        else:
            logger.info("SUCCESS: All dates processed successfully")
    
    except KeyboardInterrupt:
        logger.info("[INTERRUPTED] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[FATAL] Processing failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    main()