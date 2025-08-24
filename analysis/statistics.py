"""Statistical analysis for lottery prediction performance and data."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter
import logging

from config.database import get_db_connection
from models.database_models import (
    Sorteo, PrediccionQuiniela, PrediccionPale, PrediccionTripleta,
    ResultadoPrediccion, TipoLoteria, TipoJuego
)

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Comprehensive statistical analysis for lottery system."""
    
    def __init__(self):
        self.cache = {}
    
    def generate_performance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            cutoff_date = date.today() - timedelta(days=days_back)
            
            with get_db_connection() as session:
                # Get prediction results
                results = session.query(ResultadoPrediccion).filter(
                    ResultadoPrediccion.fecha_sorteo >= cutoff_date
                ).all()
                
                if not results:
                    return {'error': 'No prediction results found'}
                
                # Performance by game type
                game_performance = self._analyze_game_performance(results)
                
                # Performance by method
                method_performance = self._analyze_method_performance(results)
                
                # Accuracy trends
                accuracy_trends = self._analyze_accuracy_trends(results)
                
                # Overall statistics
                overall_stats = self._calculate_overall_stats(results)
                
                return {
                    'timestamp': datetime.now(),
                    'period_days': days_back,
                    'total_predictions': len(results),
                    'overall_statistics': overall_stats,
                    'game_performance': game_performance,
                    'method_performance': method_performance,
                    'accuracy_trends': accuracy_trends
                }
                
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_game_performance(self, results: List[ResultadoPrediccion]) -> Dict[str, Any]:
        """Analyze performance by game type."""
        game_stats = defaultdict(lambda: {
            'total': 0, 'aciertos': 0, 'puntos': 0, 'accuracy': 0.0
        })
        
        for result in results:
            game_id = result.tipo_juego_id
            game_stats[game_id]['total'] += 1
            if result.acierto:
                game_stats[game_id]['aciertos'] += 1
            game_stats[game_id]['puntos'] += result.puntos_obtenidos or 0
        
        # Calculate accuracies
        for game_id, stats in game_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['aciertos'] / stats['total']
        
        return dict(game_stats)
    
    def _analyze_method_performance(self, results: List[ResultadoPrediccion]) -> Dict[str, Any]:
        """Analyze performance by prediction method."""
        try:
            with get_db_connection() as session:
                method_stats = defaultdict(lambda: {
                    'total': 0, 'aciertos': 0, 'accuracy': 0.0, 'avg_confidence': 0.0
                })
                
                for result in results:
                    # Get method from prediction tables
                    method = self._get_prediction_method(session, result)
                    if method:
                        method_stats[method]['total'] += 1
                        if result.acierto:
                            method_stats[method]['aciertos'] += 1
                
                # Calculate accuracies
                for method, stats in method_stats.items():
                    if stats['total'] > 0:
                        stats['accuracy'] = stats['aciertos'] / stats['total']
                
                return dict(method_stats)
                
        except Exception as e:
            logger.error(f"Method performance analysis failed: {e}")
            return {}
    
    def _get_prediction_method(self, session, result: ResultadoPrediccion) -> Optional[str]:
        """Get prediction method for a result."""
        try:
            if result.tipo_juego_id == 1:  # Quiniela
                pred = session.query(PrediccionQuiniela).filter(
                    PrediccionQuiniela.id == result.prediccion_id
                ).first()
            elif result.tipo_juego_id == 2:  # Pale
                pred = session.query(PrediccionPale).filter(
                    PrediccionPale.id == result.prediccion_id
                ).first()
            elif result.tipo_juego_id == 3:  # Tripleta
                pred = session.query(PrediccionTripleta).filter(
                    PrediccionTripleta.id == result.prediccion_id
                ).first()
            else:
                return None
            
            return pred.metodo_generacion if pred else None
            
        except Exception as e:
            logger.warning(f"Failed to get prediction method: {e}")
            return None
    
    def _analyze_accuracy_trends(self, results: List[ResultadoPrediccion]) -> Dict[str, Any]:
        """Analyze accuracy trends over time."""
        try:
            # Group by date
            daily_accuracy = defaultdict(lambda: {'total': 0, 'aciertos': 0})
            
            for result in results:
                date_key = result.fecha_sorteo.strftime('%Y-%m-%d')
                daily_accuracy[date_key]['total'] += 1
                if result.acierto:
                    daily_accuracy[date_key]['aciertos'] += 1
            
            # Calculate daily accuracies
            trend_data = []
            for date_str, stats in sorted(daily_accuracy.items()):
                accuracy = stats['aciertos'] / stats['total'] if stats['total'] > 0 else 0
                trend_data.append({
                    'date': date_str,
                    'accuracy': accuracy,
                    'total_predictions': stats['total']
                })
            
            # Calculate trend
            if len(trend_data) >= 2:
                accuracies = [d['accuracy'] for d in trend_data]
                trend_slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
                trend_direction = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
            else:
                trend_direction = 'insufficient_data'
                trend_slope = 0
            
            return {
                'daily_data': trend_data,
                'trend_direction': trend_direction,
                'trend_slope': float(trend_slope),
                'best_day': max(trend_data, key=lambda x: x['accuracy']) if trend_data else None,
                'worst_day': min(trend_data, key=lambda x: x['accuracy']) if trend_data else None
            }
            
        except Exception as e:
            logger.error(f"Accuracy trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_stats(self, results: List[ResultadoPrediccion]) -> Dict[str, Any]:
        """Calculate overall statistics."""
        if not results:
            return {}
        
        total_predictions = len(results)
        total_aciertos = sum(1 for r in results if r.acierto)
        total_puntos = sum(r.puntos_obtenidos or 0 for r in results)
        
        # Accuracy by position
        position_stats = defaultdict(lambda: {'total': 0, 'aciertos': 0})
        
        # This would require position info from predictions - simplified for now
        overall_accuracy = total_aciertos / total_predictions if total_predictions > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'total_aciertos': total_aciertos,
            'overall_accuracy': overall_accuracy,
            'total_puntos': total_puntos,
            'avg_puntos_per_prediction': total_puntos / total_predictions if total_predictions > 0 else 0
        }
    
    def analyze_number_frequency(self, tipo_loteria_id: int, days_back: int = 90) -> Dict[str, Any]:
        """Analyze number frequency patterns."""
        try:
            cutoff_date = date.today() - timedelta(days=days_back)
            
            with get_db_connection() as session:
                sorteos = session.query(Sorteo).filter(
                    Sorteo.tipo_loteria_id == tipo_loteria_id,
                    Sorteo.fecha >= cutoff_date
                ).all()
                
                if not sorteos:
                    return {'error': 'No sorteos found'}
                
                # Collect all numbers
                all_numbers = []
                position_numbers = {'first': [], 'second': [], 'third': []}
                
                for sorteo in sorteos:
                    numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                    all_numbers.extend(numbers)
                    position_numbers['first'].append(sorteo.primer_lugar)
                    position_numbers['second'].append(sorteo.segundo_lugar)
                    position_numbers['third'].append(sorteo.tercer_lugar)
                
                # Overall frequency analysis
                frequency_counter = Counter(all_numbers)
                total_numbers = len(all_numbers)
                expected_freq = total_numbers / 100
                
                # Hot and cold numbers
                frequencies = list(frequency_counter.values())
                avg_freq = np.mean(frequencies)
                std_freq = np.std(frequencies)
                
                hot_numbers = [
                    {'number': num, 'frequency': freq, 'percentage': freq/total_numbers*100}
                    for num, freq in frequency_counter.most_common(10)
                ]
                
                cold_numbers = [
                    {'number': num, 'frequency': freq, 'percentage': freq/total_numbers*100}
                    for num, freq in frequency_counter.most_common()[-10:]
                ]
                
                # Position-specific analysis
                position_analysis = {}
                for pos, numbers in position_numbers.items():
                    pos_counter = Counter(numbers)
                    position_analysis[pos] = {
                        'most_frequent': [
                            {'number': num, 'frequency': freq}
                            for num, freq in pos_counter.most_common(5)
                        ],
                        'average': np.mean(numbers),
                        'std': np.std(numbers)
                    }
                
                # Statistical tests
                chi_square = sum(
                    (freq - expected_freq)**2 / expected_freq
                    for freq in frequencies
                )
                
                return {
                    'period_days': days_back,
                    'total_sorteos': len(sorteos),
                    'total_numbers': total_numbers,
                    'expected_frequency': expected_freq,
                    'average_frequency': avg_freq,
                    'frequency_std': std_freq,
                    'chi_square_statistic': chi_square,
                    'hot_numbers': hot_numbers,
                    'cold_numbers': cold_numbers,
                    'position_analysis': position_analysis
                }
                
        except Exception as e:
            logger.error(f"Number frequency analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_prediction_accuracy_by_position(self, days_back: int = 30) -> Dict[str, Any]:
        """Calculate prediction accuracy by position."""
        try:
            cutoff_date = date.today() - timedelta(days=days_back)
            
            with get_db_connection() as session:
                position_stats = {1: {'total': 0, 'aciertos': 0}, 
                                2: {'total': 0, 'aciertos': 0}, 
                                3: {'total': 0, 'aciertos': 0}}
                
                # Analyze quiniela predictions
                quinielas = session.query(PrediccionQuiniela).filter(
                    PrediccionQuiniela.fecha_prediccion >= cutoff_date
                ).all()
                
                for pred in quinielas:
                    # Get corresponding sorteo
                    sorteo = session.query(Sorteo).filter(
                        Sorteo.fecha == pred.fecha_prediccion,
                        Sorteo.tipo_loteria_id == pred.tipo_loteria_id
                    ).first()
                    
                    if sorteo:
                        position_stats[pred.posicion]['total'] += 1
                        
                        # Check if prediction matches any position in sorteo
                        sorteo_numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                        if pred.numero_predicho in sorteo_numbers:
                            position_stats[pred.posicion]['aciertos'] += 1
                
                # Calculate accuracies
                for pos in position_stats:
                    total = position_stats[pos]['total']
                    if total > 0:
                        position_stats[pos]['accuracy'] = position_stats[pos]['aciertos'] / total
                    else:
                        position_stats[pos]['accuracy'] = 0.0
                
                return {
                    'period_days': days_back,
                    'position_statistics': position_stats,
                    'best_position': max(position_stats.keys(), 
                                       key=lambda x: position_stats[x]['accuracy']),
                    'overall_accuracy': sum(s['aciertos'] for s in position_stats.values()) / 
                                      max(1, sum(s['total'] for s in position_stats.values()))
                }
                
        except Exception as e:
            logger.error(f"Position accuracy analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_monthly_summary(self, year: int, month: int) -> Dict[str, Any]:
        """Generate monthly performance summary."""
        try:
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
            
            with get_db_connection() as session:
                # Get sorteos for the month
                sorteos = session.query(Sorteo).filter(
                    Sorteo.fecha >= start_date,
                    Sorteo.fecha <= end_date
                ).all()
                
                # Get predictions for the month
                predictions = {
                    'quiniela': session.query(PrediccionQuiniela).filter(
                        PrediccionQuiniela.fecha_prediccion >= start_date,
                        PrediccionQuiniela.fecha_prediccion <= end_date
                    ).count(),
                    'pale': session.query(PrediccionPale).filter(
                        PrediccionPale.fecha_prediccion >= start_date,
                        PrediccionPale.fecha_prediccion <= end_date
                    ).count(),
                    'tripleta': session.query(PrediccionTripleta).filter(
                        PrediccionTripleta.fecha_prediccion >= start_date,
                        PrediccionTripleta.fecha_prediccion <= end_date
                    ).count()
                }
                
                # Get results for the month
                results = session.query(ResultadoPrediccion).filter(
                    ResultadoPrediccion.fecha_sorteo >= start_date,
                    ResultadoPrediccion.fecha_sorteo <= end_date
                ).all()
                
                # Calculate summary statistics
                total_sorteos = len(sorteos)
                total_predictions = sum(predictions.values())
                total_results = len(results)
                total_aciertos = sum(1 for r in results if r.acierto)
                
                # Number analysis for the month
                all_numbers = []
                for sorteo in sorteos:
                    all_numbers.extend([sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar])
                
                number_stats = {}
                if all_numbers:
                    number_stats = {
                        'most_frequent': Counter(all_numbers).most_common(5),
                        'average': np.mean(all_numbers),
                        'std': np.std(all_numbers),
                        'range': [min(all_numbers), max(all_numbers)]
                    }
                
                return {
                    'year': year,
                    'month': month,
                    'period': f"{start_date} to {end_date}",
                    'total_sorteos': total_sorteos,
                    'total_predictions': total_predictions,
                    'predictions_by_game': predictions,
                    'total_results_evaluated': total_results,
                    'total_aciertos': total_aciertos,
                    'monthly_accuracy': total_aciertos / max(1, total_results),
                    'number_statistics': number_stats
                }
                
        except Exception as e:
            logger.error(f"Monthly summary generation failed: {e}")
            return {'error': str(e)}


def calculate_prediction_metrics(predictions: List[Dict], actual_results: List[Dict]) -> Dict[str, float]:
    """Calculate prediction metrics."""
    try:
        if not predictions or not actual_results:
            return {'error': 'Insufficient data'}
        
        # This is a simplified implementation
        # In practice, you'd match predictions with actual results
        
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }
        
        # Placeholder calculation
        correct_predictions = 0
        total_predictions = len(predictions)
        
        # Simple accuracy calculation
        for pred in predictions:
            # Check if predicted numbers match any actual results
            pred_numbers = pred.get('numeros', [pred.get('numero')])
            for actual in actual_results:
                actual_numbers = actual.get('numeros', [actual.get('numero')])
                if any(num in actual_numbers for num in pred_numbers):
                    correct_predictions += 1
                    break
        
        if total_predictions > 0:
            metrics['accuracy'] = correct_predictions / total_predictions
            metrics['precision'] = metrics['accuracy']  # Simplified
            metrics['recall'] = metrics['accuracy']     # Simplified
            metrics['f1_score'] = metrics['accuracy']   # Simplified
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {'error': str(e)}


def generate_performance_report(days_back: int = 30) -> Dict[str, Any]:
    """Generate performance report."""
    analyzer = StatisticsAnalyzer()
    return analyzer.generate_performance_report(days_back)


def get_lottery_statistics(tipo_loteria_id: int, days_back: int = 90) -> Dict[str, Any]:
    """Get comprehensive lottery statistics."""
    analyzer = StatisticsAnalyzer()
    return analyzer.analyze_number_frequency(tipo_loteria_id, days_back)


def get_monthly_performance(year: int, month: int) -> Dict[str, Any]:
    """Get monthly performance summary."""
    analyzer = StatisticsAnalyzer()
    return analyzer.generate_monthly_summary(year, month)


# Global analyzer instance
statistics_analyzer = StatisticsAnalyzer()