"""Data cleaning and validation utilities for lottery scraping."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import re
import logging
from dataclasses import dataclass

from config.database import get_db_connection
from models.database_models import Sorteo, TipoLoteria, TipoJuego
from scraping.scraper import LotteryResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[Any] = None


class DataCleaner:
    """Clean and validate lottery data from scraping."""
    
    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for lottery data."""
        return {
            'number_range': {'min': 0, 'max': 99},
            'required_fields': ['fecha', 'tipo_loteria', 'primer_lugar', 'segundo_lugar', 'tercer_lugar'],
            'date_range': {
                'min_date': date(2020, 1, 1),
                'max_date': date.today() + timedelta(days=1)  # Allow tomorrow
            },
            'lottery_types': ['Lotería Nacional', 'Gana Más'],
            'duplicate_window_days': 1  # Check for duplicates within 1 day
        }
    
    def validate_lottery_result(self, result: LotteryResult) -> ValidationResult:
        """Validate a single lottery result."""
        errors = []
        warnings = []
        
        try:
            # Check required fields
            if not result.fecha:
                errors.append("Missing fecha")
            if not result.tipo_loteria:
                errors.append("Missing tipo_loteria")
            
            # Validate date
            if result.fecha:
                if result.fecha < self.validation_rules['date_range']['min_date']:
                    errors.append(f"Date too old: {result.fecha}")
                if result.fecha > self.validation_rules['date_range']['max_date']:
                    errors.append(f"Date too recent: {result.fecha}")
                
                # Check if date is weekend for Lotería Nacional
                if result.tipo_loteria == 'Lotería Nacional' and result.fecha.weekday() == 6:  # Sunday
                    warnings.append("Lotería Nacional on Sunday - verify time")
            
            # Validate lottery type
            if result.tipo_loteria not in self.validation_rules['lottery_types']:
                errors.append(f"Invalid lottery type: {result.tipo_loteria}")
            
            # Validate numbers
            numbers = [result.primer_lugar, result.segundo_lugar, result.tercer_lugar]
            for i, num in enumerate(numbers):
                if num is None:
                    errors.append(f"Missing number at position {i+1}")
                elif not (self.validation_rules['number_range']['min'] <= num <= self.validation_rules['number_range']['max']):
                    errors.append(f"Number {num} out of range at position {i+1}")
            
            # Check for duplicate numbers
            valid_numbers = [n for n in numbers if n is not None]
            if len(valid_numbers) != len(set(valid_numbers)):
                warnings.append(f"Duplicate numbers found: {numbers}")
            
            # Check for suspicious patterns
            if len(valid_numbers) == 3:
                # All same number
                if len(set(valid_numbers)) == 1:
                    warnings.append(f"All numbers are the same: {valid_numbers[0]}")
                
                # Sequential numbers
                sorted_nums = sorted(valid_numbers)
                if sorted_nums[1] == sorted_nums[0] + 1 and sorted_nums[2] == sorted_nums[1] + 1:
                    warnings.append(f"Sequential numbers: {sorted_nums}")
                
                # All multiples of 10
                if all(n % 10 == 0 for n in valid_numbers):
                    warnings.append(f"All multiples of 10: {valid_numbers}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                cleaned_data=result if len(errors) == 0 else None
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {e}"],
                warnings=warnings
            )
    
    def clean_text_data(self, text: str) -> str:
        """Clean raw text data from scraping."""
        if not text:
            return ""
        
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove special characters but keep Spanish characters
            text = re.sub(r'[^\w\s\-ñÑáéíóúÁÉÍÓÚ]', ' ', text)
            
            # Normalize lottery type names
            text = text.replace('Gana+', 'Gana Más')
            text = text.replace('GanaMas', 'Gana Más')
            text = text.replace('gana mas', 'Gana Más')
            text = text.replace('loteria nacional', 'Lotería Nacional')
            text = text.replace('LOTERIA NACIONAL', 'Lotería Nacional')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return text
    
    def extract_numbers_from_text(self, text: str) -> List[int]:
        """Extract lottery numbers from text with validation."""
        try:
            # Clean text first
            clean_text = self.clean_text_data(text)
            
            # Multiple patterns for number extraction
            patterns = [
                r'\b(\d{2})\s*[-,]\s*(\d{2})\s*[-,]\s*(\d{2})\b',  # 23-45-67
                r'\b(\d{2})\s+(\d{2})\s+(\d{2})\b',                 # 23 45 67
                r'(\d{2}),\s*(\d{2}),\s*(\d{2})',                   # 23, 45, 67
                r'(\d{2})(\d{2})(\d{2})',                           # 234567 (concatenated)
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, clean_text)
                for match in matches:
                    try:
                        numbers = [int(num) for num in match]
                        # Validate each number
                        if all(0 <= num <= 99 for num in numbers):
                            return numbers
                    except ValueError:
                        continue
            
            # Fallback: find any 2-digit numbers
            two_digit_matches = re.findall(r'\b\d{2}\b', clean_text)
            if len(two_digit_matches) >= 3:
                try:
                    numbers = [int(num) for num in two_digit_matches[:3]]
                    if all(0 <= num <= 99 for num in numbers):
                        return numbers
                except ValueError:
                    pass
            
            # Final fallback: single digit pairs
            single_digits = re.findall(r'\b\d\b', clean_text)
            if len(single_digits) >= 6:
                try:
                    # Group single digits into pairs
                    numbers = []
                    for i in range(0, min(6, len(single_digits)), 2):
                        num = int(single_digits[i] + single_digits[i+1])
                        if 0 <= num <= 99:
                            numbers.append(num)
                        if len(numbers) == 3:
                            break
                    
                    if len(numbers) == 3:
                        return numbers
                except (ValueError, IndexError):
                    pass
            
            return []
            
        except Exception as e:
            logger.error(f"Number extraction error: {e}")
            return []
    
    def normalize_lottery_type(self, lottery_type: str) -> Optional[str]:
        """Normalize lottery type string."""
        if not lottery_type:
            return None
        
        try:
            clean_type = self.clean_text_data(lottery_type.lower())
            
            # Mapping patterns to standard names
            if any(pattern in clean_type for pattern in ['gana', 'mas', '2:30', '230', '14:30']):
                return 'Gana Más'
            elif any(pattern in clean_type for pattern in ['nacional', '9:00', '900', '21:00', '6:00', '18:00']):
                return 'Lotería Nacional'
            
            return None
            
        except Exception as e:
            logger.error(f"Lottery type normalization error: {e}")
            return None
    
    def check_for_duplicates(self, result: LotteryResult) -> bool:
        """Check if result already exists in database."""
        try:
            with get_db_connection() as session:
                # Get lottery type ID
                lottery_type = session.query(TipoLoteria).filter(
                    TipoLoteria.nombre == result.tipo_loteria
                ).first()
                
                if not lottery_type:
                    return False
                
                # Check for existing sorteo
                existing = session.query(Sorteo).filter(
                    Sorteo.fecha == result.fecha,
                    Sorteo.tipo_loteria_id == lottery_type.id
                ).first()
                
                return existing is not None
                
        except Exception as e:
            logger.error(f"Duplicate check error: {e}")
            return False
    
    def clean_batch_results(self, results: List[LotteryResult]) -> Tuple[List[LotteryResult], List[ValidationResult]]:
        """Clean and validate a batch of lottery results."""
        cleaned_results = []
        validation_results = []
        
        logger.info(f"Cleaning batch of {len(results)} results")
        
        for result in results:
            # Validate result
            validation = self.validate_lottery_result(result)
            validation_results.append(validation)
            
            if validation.is_valid:
                # Check for duplicates
                if not self.check_for_duplicates(result):
                    cleaned_results.append(result)
                else:
                    logger.info(f"Skipping duplicate result: {result.fecha} - {result.tipo_loteria}")
            else:
                logger.warning(f"Invalid result: {validation.errors}")
        
        logger.info(f"Cleaned batch: {len(cleaned_results)} valid results")
        return cleaned_results, validation_results
    
    def generate_quality_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate data quality report."""
        try:
            total_results = len(validation_results)
            valid_results = sum(1 for v in validation_results if v.is_valid)
            
            # Count error types
            error_counts = {}
            warning_counts = {}
            
            for validation in validation_results:
                for error in validation.errors:
                    error_counts[error] = error_counts.get(error, 0) + 1
                for warning in validation.warnings:
                    warning_counts[warning] = warning_counts.get(warning, 0) + 1
            
            return {
                'timestamp': datetime.now(),
                'total_results': total_results,
                'valid_results': valid_results,
                'invalid_results': total_results - valid_results,
                'success_rate': valid_results / total_results if total_results > 0 else 0,
                'error_summary': error_counts,
                'warning_summary': warning_counts,
                'top_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'top_warnings': sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            return {'error': str(e)}


def clean_lottery_data(raw_results: List[LotteryResult]) -> Tuple[List[LotteryResult], Dict[str, Any]]:
    """Clean lottery data and return results with quality report."""
    cleaner = DataCleaner()
    cleaned_results, validation_results = cleaner.clean_batch_results(raw_results)
    quality_report = cleaner.generate_quality_report(validation_results)
    
    return cleaned_results, quality_report


def seed_sample_data(num_days: int = 30) -> None:
    """Seed database with sample lottery data for testing."""
    try:
        logger.info(f"Seeding database with {num_days} days of sample data")
        
        with get_db_connection() as session:
            # Get lottery types
            gana_mas = session.query(TipoLoteria).filter(TipoLoteria.nombre == 'Gana Más').first()
            loteria_nacional = session.query(TipoLoteria).filter(TipoLoteria.nombre == 'Lotería Nacional').first()
            
            if not gana_mas or not loteria_nacional:
                logger.error("Lottery types not found in database")
                return
            
            # Generate sample data
            start_date = date.today() - timedelta(days=num_days)
            
            for i in range(num_days):
                current_date = start_date + timedelta(days=i)
                
                # Generate Gana Más result (daily)
                gana_mas_numbers = list(np.random.choice(100, 3, replace=False))
                gana_mas_sorteo = Sorteo(
                    fecha=current_date,
                    tipo_loteria_id=gana_mas.id,
                    primer_lugar=gana_mas_numbers[0],
                    segundo_lugar=gana_mas_numbers[1],
                    tercer_lugar=gana_mas_numbers[2],
                    fuente_scraping='sample_data'
                )
                session.add(gana_mas_sorteo)
                
                # Generate Lotería Nacional result (Monday to Saturday)
                if current_date.weekday() < 6:  # Monday=0, Sunday=6
                    ln_numbers = list(np.random.choice(100, 3, replace=False))
                    ln_sorteo = Sorteo(
                        fecha=current_date,
                        tipo_loteria_id=loteria_nacional.id,
                        primer_lugar=ln_numbers[0],
                        segundo_lugar=ln_numbers[1],
                        tercer_lugar=ln_numbers[2],
                        fuente_scraping='sample_data'
                    )
                    session.add(ln_sorteo)
            
            session.commit()
            logger.info(f"Successfully seeded {num_days} days of sample data")
            
    except Exception as e:
        logger.error(f"Sample data seeding failed: {e}")
        raise


def validate_scraped_text(text: str) -> Dict[str, Any]:
    """Validate raw scraped text and extract information."""
    cleaner = DataCleaner()
    
    try:
        # Clean text
        clean_text = cleaner.clean_text_data(text)
        
        # Extract numbers
        numbers = cleaner.extract_numbers_from_text(text)
        
        # Try to identify lottery type
        lottery_type = cleaner.normalize_lottery_type(text)
        
        # Extract date patterns
        date_patterns = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        
        return {
            'original_text': text,
            'cleaned_text': clean_text,
            'extracted_numbers': numbers,
            'lottery_type': lottery_type,
            'date_patterns': date_patterns,
            'is_valid': len(numbers) == 3 and lottery_type is not None
        }
        
    except Exception as e:
        logger.error(f"Text validation failed: {e}")
        return {
            'original_text': text,
            'error': str(e),
            'is_valid': False
        }


# Global data cleaner instance
data_cleaner = DataCleaner()