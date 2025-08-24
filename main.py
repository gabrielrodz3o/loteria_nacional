"""Main application entry point for the lottery prediction system."""

import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.database import init_database, check_database_connection
from api.routes import app
from utils.scheduler import scheduler
import structlog

# Configure structured logging
def setup_logging():
    """Setup structured logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if settings.log_format == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def main():
    """Main application entry point."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Lottery Prediction System")
    logger.info(f"Environment: {settings.log_level}")
    
    try:
        # Check database connection
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Database connection failed. Please check your configuration.")
            sys.exit(1)
        
        # Initialize database tables
        logger.info("Initializing database...")
        init_database()
        
        # Start scheduler if enabled
        if settings.enable_scheduler:
            logger.info("Starting background scheduler...")
            scheduler.start()
        
        # Start the FastAPI application
        logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")
        
        if settings.api_reload:
            # Use import string for reload mode
            uvicorn.run(
                "api.routes:app",
                host=settings.api_host,
                port=settings.api_port,
                reload=True,
                log_level=settings.log_level.lower(),
                access_log=True
            )
        else:
            uvicorn.run(
                app,
                host=settings.api_host,
                port=settings.api_port,
                reload=False,
                log_level=settings.log_level.lower(),
                access_log=True
            )
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if settings.enable_scheduler and scheduler.is_running:
            logger.info("Shutting down scheduler...")
            scheduler.stop()
        
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()