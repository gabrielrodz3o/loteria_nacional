"""Scheduler for automated lottery prediction tasks."""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import logging
from datetime import datetime, date
from typing import Dict, Any

from config.settings import settings
from config.database import get_db_connection
from models.database_models import TipoLoteria
from scraping.scraper import lottery_scraper
from scraping.data_cleaner import clean_lottery_data
from utils.cache import cache_manager

logger = logging.getLogger(__name__)


class LotteryScheduler:
    """Scheduler for lottery prediction system tasks."""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone='America/Santo_Domingo')
        self.is_running = False
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """Setup event listeners for job monitoring."""
        def job_executed(event):
            logger.info(f"Job {event.job_id} executed successfully")
        
        def job_error(event):
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        
        self.scheduler.add_listener(job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(job_error, EVENT_JOB_ERROR)
    
    def start(self):
        """Start the scheduler with all configured jobs."""
        if not settings.enable_scheduler:
            logger.info("Scheduler disabled in configuration")
            return
        
        try:
            # Add daily scraping job
            self.scheduler.add_job(
                func=self.daily_scraping_job,
                trigger=CronTrigger.from_crontab(settings.scraping_schedule),
                id='daily_scraping',
                name='Daily Lottery Results Scraping',
                replace_existing=True
            )
            
            # Add daily prediction job
            self.scheduler.add_job(
                func=self.daily_prediction_job,
                trigger=CronTrigger.from_crontab(settings.prediction_schedule),
                id='daily_predictions',
                name='Daily Prediction Generation',
                replace_existing=True
            )
            
            # Add weekly cleanup job
            self.scheduler.add_job(
                func=self.weekly_cleanup_job,
                trigger=CronTrigger.from_crontab(settings.cleanup_schedule),
                id='weekly_cleanup',
                name='Weekly Data Cleanup',
                replace_existing=True
            )
            
            # Add model evaluation job (daily at midnight)
            self.scheduler.add_job(
                func=self.model_evaluation_job,
                trigger=CronTrigger(hour=0, minute=30),
                id='model_evaluation',
                name='Daily Model Evaluation',
                replace_existing=True
            )
            
            # Add cache maintenance job (every 6 hours)
            self.scheduler.add_job(
                func=self.cache_maintenance_job,
                trigger=CronTrigger(hour='*/6'),
                id='cache_maintenance',
                name='Cache Maintenance',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Lottery scheduler started successfully")
            
            # Log scheduled jobs
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                logger.info(f"Scheduled job: {job.name} - Next run: {job.next_run_time}")
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler."""
        try:
            if self.is_running:
                self.scheduler.shutdown(wait=True)
                self.is_running = False
                logger.info("Lottery scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def daily_scraping_job(self):
        """Daily job to scrape lottery results."""
        try:
            logger.info("Starting daily scraping job")
            
            # Scrape results for today
            results = lottery_scraper.scrape_all_results(date.today())
            
            if not results:
                logger.warning("No lottery results found for today")
                return
            
            # Clean and validate results
            cleaned_results, quality_report = clean_lottery_data(results)
            
            logger.info(f"Scraping completed: {len(cleaned_results)} valid results")
            logger.info(f"Quality report: {quality_report.get('success_rate', 0):.2%} success rate")
            
            # Store results in database
            self._store_scraping_results(cleaned_results)
            
            # Clear prediction cache since we have new results
            cache_manager.delete_pattern("predicciones_*")
            
        except Exception as e:
            logger.error(f"Daily scraping job failed: {e}")
    
    def daily_prediction_job(self):
        """Daily job to generate predictions."""
        try:
            from predictions.predictor_engine import predictor_engine
            logger.info("Starting daily prediction job")
            
            # Get all active lottery types
            with get_db_connection() as session:
                lottery_types = session.query(TipoLoteria).filter(
                    TipoLoteria.activo == True
                ).all()
            
            today_str = date.today().strftime('%Y-%m-%d')
            
            for lottery_type in lottery_types:
                try:
                    logger.info(f"Generating predictions for {lottery_type.nombre}")
                    
                    # Train models
                    training_results = predictor_engine.entrenar_modelos(
                        fecha=date.today(),
                        tipo_loteria_id=lottery_type.id
                    )
                    
                    # Generate predictions
                    predictions = predictor_engine.generar_predicciones_diarias(
                        fecha=today_str,
                        tipo_loteria_id=lottery_type.id
                    )
                    
                    # Store predictions
                    predictor_engine.insertar_predicciones_en_bd(
                        predictions,
                        date.today(),
                        lottery_type.id
                    )
                    
                    logger.info(f"Predictions generated for {lottery_type.nombre}")
                    
                except Exception as e:
                    logger.error(f"Prediction generation failed for {lottery_type.nombre}: {e}")
            
            logger.info("Daily prediction job completed")
            
        except Exception as e:
            logger.error(f"Daily prediction job failed: {e}")
    
    def weekly_cleanup_job(self):
        """Weekly job for data cleanup."""
        try:
            from predictions.predictor_engine import predictor_engine
            logger.info("Starting weekly cleanup job")
            
            # Clean old model cache files
            removed_files = predictor_engine.limpiar_cache_modelos()
            logger.info(f"Removed {removed_files} old model cache files")
            
            # Clean old predictions from database
            with get_db_connection() as session:
                result = session.execute(
                    "SELECT limpiar_predicciones_antiguas(90)"
                ).scalar()
                logger.info(f"Removed {result} old prediction records")
            
            # Clear old cache entries
            cleared_keys = cache_manager.clear_all()
            logger.info(f"Cleared {cleared_keys} cache entries")
            
            logger.info("Weekly cleanup completed")
            
        except Exception as e:
            logger.error(f"Weekly cleanup job failed: {e}")
    
    def model_evaluation_job(self):
        """Daily job to evaluate model performance."""
        try:
            from predictions.predictor_engine import predictor_engine
            logger.info("Starting model evaluation job")
            
            # Evaluate models
            evaluation_results = predictor_engine.evaluar_modelos(date.today())
            
            if evaluation_results:
                logger.info("Model evaluation completed")
                
                # Log performance summary
                for model_name, metrics in evaluation_results.items():
                    avg_score = metrics.get('avg_score', 0)
                    evaluations = metrics.get('evaluations', 0)
                    logger.info(f"{model_name}: {avg_score:.4f} avg score ({evaluations} evaluations)")
            else:
                logger.warning("No evaluation results available")
            
        except Exception as e:
            logger.error(f"Model evaluation job failed: {e}")
    
    def cache_maintenance_job(self):
        """Periodic cache maintenance job."""
        try:
            logger.info("Starting cache maintenance job")
            
            # Get cache info
            cache_info = cache_manager.get_cache_info()
            
            if cache_info.get('connected'):
                total_keys = cache_info.get('total_keys', 0)
                memory_used = cache_info.get('memory_used', 'unknown')
                
                logger.info(f"Cache status: {total_keys} keys, {memory_used} memory used")
                
                # Clean expired keys (Redis should do this automatically, but we can be explicit)
                # This is mainly for monitoring purposes
                
                # Remove very old cache entries if memory usage is high
                if total_keys > 10000:
                    # Clean old statistics cache
                    old_stats_pattern = f"estadisticas_*"
                    cleaned = cache_manager.delete_pattern(old_stats_pattern)
                    logger.info(f"Cleaned {cleaned} old statistics cache entries")
            else:
                logger.warning("Cache not connected during maintenance")
            
        except Exception as e:
            logger.error(f"Cache maintenance job failed: {e}")
    
    def _store_scraping_results(self, results):
        """Store scraping results in database."""
        try:
            from models.database_models import Sorteo
            
            with get_db_connection() as session:
                stored_count = 0
                
                for result in results:
                    # Get lottery type ID
                    lottery_type = session.query(TipoLoteria).filter(
                        TipoLoteria.nombre == result.tipo_loteria
                    ).first()
                    
                    if not lottery_type:
                        logger.warning(f"Unknown lottery type: {result.tipo_loteria}")
                        continue
                    
                    # Check if already exists
                    existing = session.query(Sorteo).filter(
                        Sorteo.fecha == result.fecha,
                        Sorteo.tipo_loteria_id == lottery_type.id
                    ).first()
                    
                    if existing:
                        logger.info(f"Sorteo already exists: {result.fecha} - {result.tipo_loteria}")
                        continue
                    
                    # Create new sorteo
                    sorteo = Sorteo(
                        fecha=result.fecha,
                        tipo_loteria_id=lottery_type.id,
                        primer_lugar=result.primer_lugar,
                        segundo_lugar=result.segundo_lugar,
                        tercer_lugar=result.tercer_lugar,
                        fuente_scraping=result.fuente
                    )
                    
                    session.add(sorteo)
                    stored_count += 1
                
                session.commit()
                logger.info(f"Stored {stored_count} new sorteo records")
                
        except Exception as e:
            logger.error(f"Failed to store scraping results: {e}")
    
    def add_one_time_job(self, func, run_date, job_id=None, **kwargs):
        """Add a one-time job to the scheduler."""
        try:
            job = self.scheduler.add_job(
                func=func,
                trigger='date',
                run_date=run_date,
                id=job_id,
                replace_existing=True,
                **kwargs
            )
            logger.info(f"Added one-time job: {job.id} at {run_date}")
            return job
        except Exception as e:
            logger.error(f"Failed to add one-time job: {e}")
            return None
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all scheduled jobs."""
        try:
            jobs = self.scheduler.get_jobs()
            
            job_status = {
                'scheduler_running': self.is_running,
                'total_jobs': len(jobs),
                'jobs': []
            }
            
            for job in jobs:
                job_info = {
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger),
                    'func_name': job.func.__name__ if hasattr(job.func, '__name__') else str(job.func)
                }
                job_status['jobs'].append(job_info)
            
            return job_status
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {'error': str(e)}
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a specific job."""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a specific job."""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    def run_job_now(self, job_id: str) -> bool:
        """Run a job immediately."""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.func()
                logger.info(f"Executed job immediately: {job_id}")
                return True
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to run job {job_id}: {e}")
            return False


def setup_scheduler() -> LotteryScheduler:
    """Setup and configure the lottery scheduler."""
    try:
        scheduler = LotteryScheduler()
        
        if settings.enable_scheduler:
            logger.info("Setting up lottery scheduler")
            scheduler.start()
        else:
            logger.info("Scheduler disabled in configuration")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"Scheduler setup failed: {e}")
        raise


def manual_scraping_task():
    """Manual task to run scraping immediately."""
    try:
        logger.info("Running manual scraping task")
        scheduler.daily_scraping_job()
        return {"status": "success", "message": "Scraping completed"}
    except Exception as e:
        logger.error(f"Manual scraping failed: {e}")
        return {"status": "error", "message": str(e)}


def manual_prediction_task():
    """Manual task to generate predictions immediately."""
    try:
        logger.info("Running manual prediction task")
        scheduler.daily_prediction_job()
        return {"status": "success", "message": "Predictions generated"}
    except Exception as e:
        logger.error(f"Manual prediction failed: {e}")
        return {"status": "error", "message": str(e)}


def manual_cleanup_task():
    """Manual task to run cleanup immediately."""
    try:
        logger.info("Running manual cleanup task")
        scheduler.weekly_cleanup_job()
        return {"status": "success", "message": "Cleanup completed"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        return {"status": "error", "message": str(e)}


# Global scheduler instance
scheduler = LotteryScheduler()


# Utility functions for external use
def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status."""
    return scheduler.get_job_status()


def pause_scheduler_job(job_id: str) -> bool:
    """Pause a scheduler job."""
    return scheduler.pause_job(job_id)


def resume_scheduler_job(job_id: str) -> bool:
    """Resume a scheduler job."""
    return scheduler.resume_job(job_id)


def run_scheduler_job_now(job_id: str) -> bool:
    """Run a scheduler job immediately."""
    return scheduler.run_job_now(job_id)


def add_custom_job(func, cron_expression: str, job_id: str, name: str = None):
    """Add a custom job to the scheduler."""
    try:
        job = scheduler.scheduler.add_job(
            func=func,
            trigger=CronTrigger.from_crontab(cron_expression),
            id=job_id,
            name=name or job_id,
            replace_existing=True
        )
        logger.info(f"Added custom job: {job_id}")
        return job
    except Exception as e:
        logger.error(f"Failed to add custom job {job_id}: {e}")
        return None


# Health check function
def scheduler_health_check() -> Dict[str, Any]:
    """Check scheduler health and status."""
    try:
        health = {
            'scheduler_running': scheduler.is_running,
            'enabled_in_config': settings.enable_scheduler,
            'jobs_count': 0,
            'next_jobs': [],
            'last_error': None
        }
        
        if scheduler.is_running:
            jobs = scheduler.scheduler.get_jobs()
            health['jobs_count'] = len(jobs)
            
            # Get next 3 jobs to run
            upcoming_jobs = sorted(
                [j for j in jobs if j.next_run_time], 
                key=lambda x: x.next_run_time
            )[:3]
            
            health['next_jobs'] = [
                {
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat()
                }
                for job in upcoming_jobs
            ]
        
        return health
        
    except Exception as e:
        logger.error(f"Scheduler health check failed: {e}")
        return {
            'scheduler_running': False,
            'enabled_in_config': settings.enable_scheduler,
            'error': str(e)
        }