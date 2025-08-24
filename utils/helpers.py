"""Helper utilities for lottery prediction system."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, time
import re
import logging
from itertools import combinations
from config.settings import settings

logger = logging.getLogger(__name__)


def validate_number_range(number: int, min_val: int = None, max_val: int = None) -> bool:
    """Validate if number is within valid lottery range."""
    min_val = min_val or settings.number_range_min
    max_val = max_val or settings.number_range_max
    return min_val <= number <= max_val


def validate_lottery_numbers(numbers: List[int], game_type: str) -> Dict[str, Any]:
    """Validate lottery numbers based on game type."""
    try:
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check number count
        expected_counts = {'quiniela': 1, 'pale': 2, 'tripleta': 3}
        expected_count = expected_counts.get(game_type, 1)
        
        if len(numbers) != expected_count:
            validation_result['errors'].append(
                f"{game_type} requires {expected_count} numbers, got {len(numbers)}"
            )
            validation_result['is_valid'] = False
        
        # Check number ranges
        for i, num in enumerate(numbers):
            if not validate_number_range(num):
                validation_result['errors'].append(
                    f"Number {num} at position {i+1} is out of range (0-99)"
                )
                validation_result['is_valid'] = False
        
        # Check for duplicates (not allowed in pale/tripleta)
        if game_type in ['pale', 'tripleta'] and len(set(numbers)) != len(numbers):
            validation_result['errors'].append(
                f"Duplicate numbers not allowed in {game_type}: {numbers}"
            )
            validation_result['is_valid'] = False
        
        # Warnings for suspicious patterns
        if len(numbers) >= 2:
            # All same number
            if len(set(numbers)) == 1:
                validation_result['warnings'].append(
                    f"All numbers are identical: {numbers[0]}"
                )
            
            # Sequential numbers
            sorted_nums = sorted(numbers)
            if len(sorted_nums) >= 2:
                is_sequential = all(
                    sorted_nums[i+1] == sorted_nums[i] + 1 
                    for i in range(len(sorted_nums) - 1)
                )
                if is_sequential:
                    validation_result['warnings'].append(
                        f"Sequential numbers: {sorted_nums}"
                    )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Number validation failed: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation error: {e}"],
            'warnings': []
        }


def generate_combinations(game_type: str, pool_size: int = 100) -> List[List[int]]:
    """Generate all possible combinations for a game type."""
    try:
        numbers = list(range(pool_size))
        
        if game_type == 'quiniela':
            return [[num] for num in numbers]
        elif game_type == 'pale':
            return [list(combo) for combo in combinations(numbers, 2)]
        elif game_type == 'tripleta':
            return [list(combo) for combo in combinations(numbers, 3)]
        else:
            raise ValueError(f"Unknown game type: {game_type}")
            
    except Exception as e:
        logger.error(f"Combination generation failed: {e}")
        return []


def calculate_combination_probability(numbers: List[int], game_type: str) -> float:
    """Calculate theoretical probability of a combination."""
    try:
        total_combinations = len(generate_combinations(game_type))
        if total_combinations > 0:
            return 1.0 / total_combinations
        return 0.0
        
    except Exception as e:
        logger.error(f"Probability calculation failed: {e}")
        return 0.0


def format_prediction_output(predictions: Dict[str, List[Dict]], 
                           tipo_loteria_id: int) -> Dict[str, Any]:
    """Format prediction output for API response."""
    try:
        formatted = {
            'fecha': date.today().strftime('%Y-%m-%d'),
            'tipo_loteria_id': tipo_loteria_id,
            'timestamp': datetime.now().isoformat(),
            'quiniela': [],
            'pale': [],
            'tripleta': []
        }
        
        for game_type in ['quiniela', 'pale', 'tripleta']:
            if game_type in predictions:
                for pred in predictions[game_type]:
                    formatted_pred = {
                        'posicion': pred.get('posicion', 0),
                        'probabilidad': round(pred.get('probabilidad', 0.0), 4),
                        'metodo_generacion': pred.get('metodo_generacion', 'unknown'),
                        'score_confianza': round(pred.get('score_confianza', 0.0), 4)
                    }
                    
                    # Add numbers based on game type
                    if game_type == 'quiniela':
                        formatted_pred['numero'] = pred.get('numero', pred.get('numeros', [0])[0])
                    else:
                        formatted_pred['numeros'] = pred.get('numeros', [])
                    
                    formatted[game_type].append(formatted_pred)
        
        return formatted
        
    except Exception as e:
        logger.error(f"Prediction formatting failed: {e}")
        return {
            'error': str(e),
            'fecha': date.today().strftime('%Y-%m-%d'),
            'tipo_loteria_id': tipo_loteria_id
        }


def parse_lottery_schedule(schedule_str: str) -> Optional[time]:
    """Parse lottery schedule string to time object."""
    try:
        # Handle different time formats
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(AM|PM)?',
            r'(\d{1,2}):(\d{2}):(\d{2})',
            r'(\d{1,2})(\d{2})',  # HHMM format
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, schedule_str.upper())
            if match:
                groups = match.groups()
                
                if len(groups) >= 2:
                    hour = int(groups[0])
                    minute = int(groups[1])
                    
                    # Handle AM/PM
                    if len(groups) > 2 and groups[2]:
                        if groups[2] == 'PM' and hour != 12:
                            hour += 12
                        elif groups[2] == 'AM' and hour == 12:
                            hour = 0
                    
                    # Handle seconds
                    second = int(groups[2]) if len(groups) > 2 and groups[2].isdigit() else 0
                    
                    return time(hour, minute, second)
        
        return None
        
    except Exception as e:
        logger.error(f"Schedule parsing failed: {e}")
        return None


def calculate_lottery_stats(numbers: List[int]) -> Dict[str, Any]:
    """Calculate comprehensive statistics for lottery numbers."""
    try:
        if not numbers:
            return {'error': 'No numbers provided'}
        
        nums = np.array(numbers)
        
        stats = {
            # Basic statistics
            'count': len(numbers),
            'mean': float(np.mean(nums)),
            'median': float(np.median(nums)),
            'std': float(np.std(nums)),
            'min': int(np.min(nums)),
            'max': int(np.max(nums)),
            'range': int(np.max(nums) - np.min(nums)),
            
            # Distribution
            'unique_count': len(np.unique(nums)),
            'unique_ratio': len(np.unique(nums)) / len(numbers),
            
            # Parity
            'even_count': int(np.sum(nums % 2 == 0)),
            'odd_count': int(np.sum(nums % 2 == 1)),
            'even_ratio': float(np.sum(nums % 2 == 0) / len(numbers)),
            
            # Range distribution
            'low_numbers': int(np.sum(nums < 50)),
            'high_numbers': int(np.sum(nums >= 50)),
            'low_ratio': float(np.sum(nums < 50) / len(numbers)),
            
            # Decade distribution
            'decade_distribution': {}
        }
        
        # Calculate decade distribution
        for decade in range(10):
            count = int(np.sum((nums >= decade * 10) & (nums < (decade + 1) * 10)))
            stats['decade_distribution'][f'{decade}0s'] = count
        
        return stats
        
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        return {'error': str(e)}


def normalize_lottery_type_name(name: str) -> Optional[str]:
    """Normalize lottery type name to standard format."""
    if not name:
        return None
    
    try:
        name_lower = name.lower().strip()
        
        # Gana Más variations
        gana_mas_patterns = [
            'gana mas', 'gana+', 'ganamas', 'gana_mas', 'gana-mas'
        ]
        
        for pattern in gana_mas_patterns:
            if pattern in name_lower:
                return 'Gana Más'
        
        # Lotería Nacional variations
        nacional_patterns = [
            'loteria nacional', 'nacional', 'loto nacional', 'lotería nacional'
        ]
        
        for pattern in nacional_patterns:
            if pattern in name_lower:
                return 'Lotería Nacional'
        
        return None
        
    except Exception as e:
        logger.error(f"Lottery type normalization failed: {e}")
        return None


def generate_random_predictions(game_type: str, count: int = 3) -> List[Dict[str, Any]]:
    """Generate random predictions for testing purposes."""
    try:
        predictions = []
        
        for i in range(count):
            if game_type == 'quiniela':
                numbers = [np.random.randint(0, 100)]
            elif game_type == 'pale':
                numbers = list(np.random.choice(100, 2, replace=False))
            elif game_type == 'tripleta':
                numbers = list(np.random.choice(100, 3, replace=False))
            else:
                continue
            
            prediction = {
                'posicion': i + 1,
                'numeros': numbers,
                'probabilidad': np.random.uniform(0.01, 0.1),
                'metodo_generacion': 'random_fallback',
                'score_confianza': np.random.uniform(0.1, 0.3)
            }
            
            if game_type == 'quiniela':
                prediction['numero'] = numbers[0]
                del prediction['numeros']
            
            predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Random prediction generation failed: {e}")
        return []


def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements for lottery prediction."""
    try:
        requirements_check = {
            'python_version': True,
            'required_packages': {},
            'database_connection': False,
            'redis_connection': False,
            'memory_available': True,
            'disk_space': True
        }
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            requirements_check['python_version'] = False
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'sqlalchemy', 
            'fastapi', 'redis', 'psycopg2', 'pgvector'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                requirements_check['required_packages'][package] = True
            except ImportError:
                requirements_check['required_packages'][package] = False
        
        # Check database connection
        try:
            from config.database import check_database_connection
            requirements_check['database_connection'] = check_database_connection()
        except:
            requirements_check['database_connection'] = False
        
        # Check Redis connection
        try:
            from utils.cache import cache_manager
            requirements_check['redis_connection'] = cache_manager.is_connected
        except:
            requirements_check['redis_connection'] = False
        
        return requirements_check
        
    except Exception as e:
        logger.error(f"System requirements check failed: {e}")
        return {'error': str(e)}


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def format_number_with_leading_zeros(number: int, digits: int = 2) -> str:
    """Format number with leading zeros."""
    return str(number).zfill(digits)


def parse_numbers_from_string(text: str) -> List[int]:
    """Parse lottery numbers from text string."""
    try:
        # Find all 2-digit numbers
        numbers = re.findall(r'\b\d{2}\b', text)
        
        # Convert to integers and validate
        valid_numbers = []
        for num_str in numbers:
            num = int(num_str)
            if validate_number_range(num):
                valid_numbers.append(num)
        
        return valid_numbers
        
    except Exception as e:
        logger.error(f"Number parsing failed: {e}")
        return []


def get_business_days_between(start_date: date, end_date: date) -> int:
    """Get number of business days between two dates."""
    try:
        business_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Monday = 0, Sunday = 6
            if current_date.weekday() < 6:  # Monday to Saturday
                business_days += 1
            current_date = date.fromordinal(current_date.toordinal() + 1)
        
        return business_days
        
    except Exception as e:
        logger.error(f"Business days calculation failed: {e}")
        return 0


# Configuration validation
def validate_configuration() -> Dict[str, Any]:
    """Validate system configuration."""
    try:
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required settings
        required_settings = [
            'database_url', 'redis_url', 'api_host', 'api_port'
        ]
        
        for setting in required_settings:
            if not hasattr(settings, setting) or not getattr(settings, setting):
                validation['errors'].append(f"Missing required setting: {setting}")
                validation['valid'] = False
        
        # Validate number ranges
        if settings.number_range_min < 0 or settings.number_range_max > 99:
            validation['errors'].append("Invalid number range configuration")
            validation['valid'] = False
        
        if settings.number_range_min >= settings.number_range_max:
            validation['errors'].append("Min range must be less than max range")
            validation['valid'] = False
        
        # Check cache TTL values
        if settings.cache_ttl_predictions <= 0:
            validation['warnings'].append("Cache TTL for predictions should be positive")
        
        return validation
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {e}"],
            'warnings': []
        }