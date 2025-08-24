"""Database configuration with SQLAlchemy and retry logic."""

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator
import time
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Metadata for schema operations
metadata = MetaData()


class DatabaseError(Exception):
    """Custom database exception."""
    pass


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise DatabaseError(f"Database operation failed: {e}")
    finally:
        session.close()


@contextmanager
def get_db_connection():
    """Context manager for database connections with retry logic."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with get_database_session() as session:
                yield session
                return  # Exit successfully after yielding
        except SQLAlchemyError as e:
            if attempt == max_retries - 1:
                logger.error(f"Database connection failed after {max_retries} attempts: {e}")
                raise DatabaseError(f"Database connection failed: {e}")
            
            logger.warning(f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2


def init_database():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Database initialization failed: {e}")


def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def execute_raw_sql(query: str, params: dict = None) -> list:
    """Execute raw SQL query with parameters."""
    try:
        with get_db_connection() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()
    except Exception as e:
        logger.error(f"Raw SQL execution failed: {e}")
        raise DatabaseError(f"SQL execution failed: {e}")


class DatabaseManager:
    """Database operations manager with transaction support."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert(self, model_class, data_list: list) -> int:
        """Perform bulk insert operation."""
        try:
            with self.transaction() as session:
                session.bulk_insert_mappings(model_class, data_list)
                return len(data_list)
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise DatabaseError(f"Bulk insert failed: {e}")
    
    def bulk_update(self, model_class, data_list: list) -> int:
        """Perform bulk update operation."""
        try:
            with self.transaction() as session:
                session.bulk_update_mappings(model_class, data_list)
                return len(data_list)
        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
            raise DatabaseError(f"Bulk update failed: {e}")


# Global database manager instance
db_manager = DatabaseManager()