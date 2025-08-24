"""Pattern analysis for lottery data using vector similarity and clustering."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict, Counter
import logging

from config.database import get_db_connection
from models.database_models import Sorteo, Vector, TipoLoteria
from utils.embeddings import create_embedding, cosine_similarity

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Analyze patterns in lottery data using various techniques."""
    
    def __init__(self):
        self.pattern_cache = {}
        self.similarity_threshold = 0.8
    
    def analyze_number_patterns(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze various patterns in lottery numbers."""
        try:
            if not sorteos:
                return {'error': 'No sorteos provided'}
            
            # Extract numbers
            all_numbers = []
            for sorteo in sorteos:
                all_numbers.extend([sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar])
            
            # Basic frequency analysis
            frequency_analysis = self._analyze_frequencies(all_numbers)
            
            # Sequence patterns
            sequence_patterns = self._analyze_sequences(sorteos)
            
            # Position patterns
            position_patterns = self._analyze_positions(sorteos)
            
            # Day patterns
            day_patterns = self._analyze_day_patterns(sorteos)
            
            # Statistical patterns
            statistical_patterns = self._analyze_statistical_patterns(all_numbers)
            
            # Pair and triplet patterns
            pair_patterns = self._analyze_pair_patterns(sorteos)
            
            # Sum patterns
            sum_patterns = self._analyze_sum_patterns(sorteos)
            
            # Gap patterns
            gap_patterns = self._analyze_gap_patterns(sorteos)
            
            return {
                'timestamp': datetime.now(),
                'total_sorteos': len(sorteos),
                'total_numbers': len(all_numbers),
                'frequency_analysis': frequency_analysis,
                'sequence_patterns': sequence_patterns,
                'position_patterns': position_patterns,
                'day_patterns': day_patterns,
                'statistical_patterns': statistical_patterns,
                'pair_patterns': pair_patterns,
                'sum_patterns': sum_patterns,
                'gap_patterns': gap_patterns
            }
            
        except Exception as e:
            logger.error(f"Number pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_frequencies(self, numbers: List[int]) -> Dict[str, Any]:
        """Analyze number frequencies."""
        try:
            counter = Counter(numbers)
            total = len(numbers)
            
            # Most and least frequent
            most_frequent = counter.most_common(10)
            least_frequent = counter.most_common()[-10:]
            
            # Calculate statistics
            frequencies = list(counter.values())
            expected_freq = total / 100 if total > 0 else 0
            
            # Chi-square test approximation
            chi_square = sum((freq - expected_freq)**2 / expected_freq 
                           for freq in frequencies if expected_freq > 0)
            
            # Hot and cold numbers
            avg_frequency = np.mean(frequencies) if frequencies else 0
            hot_numbers = [num for num, freq in counter.items() if freq > avg_frequency * 1.2]
            cold_numbers = [num for num, freq in counter.items() if freq < avg_frequency * 0.8]
            
            # Frequency distribution
            freq_distribution = {
                'range_0_5': sum(1 for f in frequencies if 0 <= f <= 5),
                'range_6_10': sum(1 for f in frequencies if 6 <= f <= 10),
                'range_11_15': sum(1 for f in frequencies if 11 <= f <= 15),
                'range_16_plus': sum(1 for f in frequencies if f >= 16)
            }
            
            return {
                'most_frequent': most_frequent,
                'least_frequent': least_frequent,
                'hot_numbers': hot_numbers,
                'cold_numbers': cold_numbers,
                'chi_square_stat': chi_square,
                'expected_frequency': expected_freq,
                'variance': np.var(frequencies) if frequencies else 0,
                'frequency_distribution': freq_distribution,
                'total_unique_numbers': len(counter),
                'coverage_percentage': len(counter) / 100 * 100
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_sequences(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze sequential patterns."""
        try:
            patterns = {
                'consecutive_in_draw': 0,
                'ascending_draws': 0,
                'descending_draws': 0,
                'repeated_numbers': 0,
                'mirror_patterns': 0,
                'arithmetic_progressions': 0,
                'sum_patterns': defaultdict(int),
                'digit_sum_patterns': defaultdict(int),
                'reverse_patterns': 0,
                'palindrome_patterns': 0
            }
            
            for sorteo in sorteos:
                numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                sorted_numbers = sorted(numbers)
                
                # Check for consecutive numbers
                consecutive_count = 0
                for i in range(len(sorted_numbers) - 1):
                    if sorted_numbers[i+1] == sorted_numbers[i] + 1:
                        consecutive_count += 1
                
                if consecutive_count >= 1:
                    patterns['consecutive_in_draw'] += 1
                
                # Check if ascending
                if numbers == sorted(numbers):
                    patterns['ascending_draws'] += 1
                
                # Check if descending
                if numbers == sorted(numbers, reverse=True):
                    patterns['descending_draws'] += 1
                
                # Check for repeated numbers
                if len(set(numbers)) < len(numbers):
                    patterns['repeated_numbers'] += 1
                
                # Check for arithmetic progression
                if len(numbers) == 3:
                    diff1 = numbers[1] - numbers[0]
                    diff2 = numbers[2] - numbers[1]
                    if diff1 == diff2 and diff1 != 0:
                        patterns['arithmetic_progressions'] += 1
                
                # Check for mirror patterns (e.g., 12, 34, 21)
                str_numbers = [str(n).zfill(2) for n in numbers]
                reversed_first = str_numbers[0][::-1]
                if reversed_first in str_numbers[1:]:
                    patterns['mirror_patterns'] += 1
                
                # Check for reverse patterns
                if numbers == numbers[::-1]:
                    patterns['reverse_patterns'] += 1
                
                # Check for palindrome patterns in concatenated string
                concat_str = ''.join(str(n).zfill(2) for n in numbers)
                if concat_str == concat_str[::-1]:
                    patterns['palindrome_patterns'] += 1
                
                # Sum patterns
                total_sum = sum(numbers)
                patterns['sum_patterns'][total_sum] += 1
                
                # Digit sum patterns
                digit_sum = sum(int(digit) for num in numbers for digit in str(num).zfill(2))
                patterns['digit_sum_patterns'][digit_sum] += 1
            
            # Convert defaultdicts to regular dicts for JSON serialization
            patterns['sum_patterns'] = dict(patterns['sum_patterns'])
            patterns['digit_sum_patterns'] = dict(patterns['digit_sum_patterns'])
            
            # Add percentages
            total_draws = len(sorteos)
            if total_draws > 0:
                patterns['consecutive_percentage'] = patterns['consecutive_in_draw'] / total_draws * 100
                patterns['ascending_percentage'] = patterns['ascending_draws'] / total_draws * 100
                patterns['descending_percentage'] = patterns['descending_draws'] / total_draws * 100
                patterns['repeated_percentage'] = patterns['repeated_numbers'] / total_draws * 100
                patterns['arithmetic_percentage'] = patterns['arithmetic_progressions'] / total_draws * 100
                patterns['mirror_percentage'] = patterns['mirror_patterns'] / total_draws * 100
            
            # Most common sums
            patterns['most_common_sums'] = sorted(
                patterns['sum_patterns'].items(), 
                key=lambda x: x[1], reverse=True
            )[:10]
            
            patterns['most_common_digit_sums'] = sorted(
                patterns['digit_sum_patterns'].items(),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Sequence analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_positions(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze patterns by position."""
        try:
            position_stats = {
                'first_position': [],
                'second_position': [],
                'third_position': []
            }
            
            for sorteo in sorteos:
                position_stats['first_position'].append(sorteo.primer_lugar)
                position_stats['second_position'].append(sorteo.segundo_lugar)
                position_stats['third_position'].append(sorteo.tercer_lugar)
            
            analysis = {}
            for position, numbers in position_stats.items():
                if numbers:
                    analysis[position] = {
                        'mean': np.mean(numbers),
                        'median': np.median(numbers),
                        'std': np.std(numbers),
                        'min': min(numbers),
                        'max': max(numbers),
                        'most_common': Counter(numbers).most_common(5),
                        'range_distribution': self._calculate_range_distribution(numbers),
                        'even_odd_ratio': sum(1 for n in numbers if n % 2 == 0) / len(numbers),
                        'high_low_ratio': sum(1 for n in numbers if n >= 50) / len(numbers),
                        'decade_preference': self._calculate_decade_preference(numbers)
                    }
            
            # Cross-position analysis
            cross_analysis = self._analyze_cross_positions(sorteos)
            analysis['cross_position_analysis'] = cross_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Position analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_cross_positions(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze relationships between positions."""
        try:
            differences = {
                'first_second': [],
                'second_third': [],
                'first_third': []
            }
            
            correlations = []
            
            for sorteo in sorteos:
                nums = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                
                differences['first_second'].append(abs(nums[1] - nums[0]))
                differences['second_third'].append(abs(nums[2] - nums[1]))
                differences['first_third'].append(abs(nums[2] - nums[0]))
                
                correlations.append(nums)
            
            # Calculate correlation matrix
            if correlations:
                corr_matrix = np.corrcoef(np.array(correlations).T)
                
                return {
                    'average_differences': {
                        k: np.mean(v) for k, v in differences.items()
                    },
                    'correlation_matrix': {
                        'first_second': float(corr_matrix[0, 1]),
                        'first_third': float(corr_matrix[0, 2]),
                        'second_third': float(corr_matrix[1, 2])
                    },
                    'max_differences': {
                        k: max(v) for k, v in differences.items()
                    },
                    'min_differences': {
                        k: min(v) for k, v in differences.items()
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Cross-position analysis failed: {e}")
            return {}
    
    def _analyze_day_patterns(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze patterns by day of week."""
        try:
            day_patterns = defaultdict(list)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for sorteo in sorteos:
                day_name = day_names[sorteo.fecha.weekday()]
                numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                day_patterns[day_name].extend(numbers)
            
            analysis = {}
            for day, numbers in day_patterns.items():
                if numbers:
                    # Group numbers back into draws
                    draws = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
                    
                    analysis[day] = {
                        'total_draws': len(draws),
                        'avg_sum': np.mean([sum(draw) for draw in draws]),
                        'most_frequent': Counter(numbers).most_common(5),
                        'avg_number': np.mean(numbers),
                        'std_number': np.std(numbers),
                        'even_ratio': sum(1 for n in numbers if n % 2 == 0) / len(numbers),
                        'high_ratio': sum(1 for n in numbers if n >= 50) / len(numbers)
                    }
            
            # Find best and worst performing days
            if analysis:
                # Sort by average sum (example metric)
                sorted_days = sorted(analysis.items(), key=lambda x: x[1]['avg_sum'], reverse=True)
                analysis['best_sum_day'] = sorted_days[0][0] if sorted_days else None
                analysis['worst_sum_day'] = sorted_days[-1][0] if sorted_days else None
            
            return dict(analysis)
            
        except Exception as e:
            logger.error(f"Day pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_statistical_patterns(self, numbers: List[int]) -> Dict[str, Any]:
        """Analyze statistical patterns."""
        try:
            if not numbers:
                return {'error': 'No numbers provided'}
            
            # Basic statistics
            stats = {
                'mean': np.mean(numbers),
                'median': np.median(numbers),
                'mode': Counter(numbers).most_common(1)[0] if numbers else (0, 0),
                'std': np.std(numbers),
                'variance': np.var(numbers),
                'skewness': self._calculate_skewness(numbers),
                'kurtosis': self._calculate_kurtosis(numbers),
                'coefficient_of_variation': np.std(numbers) / np.mean(numbers) if np.mean(numbers) != 0 else 0
            }
            
            # Distribution analysis
            stats['distribution'] = {
                'even_count': sum(1 for n in numbers if n % 2 == 0),
                'odd_count': sum(1 for n in numbers if n % 2 == 1),
                'low_numbers': sum(1 for n in numbers if n < 50),
                'high_numbers': sum(1 for n in numbers if n >= 50),
                'decade_distribution': self._calculate_decade_distribution(numbers),
                'quartile_distribution': self._calculate_quartile_distribution(numbers)
            }
            
            # Entropy calculation
            counter = Counter(numbers)
            probabilities = [count / len(numbers) for count in counter.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            stats['entropy'] = entropy
            
            # Runs test for randomness
            stats['runs_test'] = self._runs_test(numbers)
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistical pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_pair_patterns(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze pair patterns in draws."""
        try:
            pair_frequencies = Counter()
            triplet_frequencies = Counter()
            
            for sorteo in sorteos:
                numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                
                # Analyze all pairs
                from itertools import combinations
                for pair in combinations(numbers, 2):
                    pair_frequencies[tuple(sorted(pair))] += 1
                
                # Analyze triplet
                triplet_frequencies[tuple(sorted(numbers))] += 1
            
            # Most common pairs and triplets
            most_common_pairs = pair_frequencies.most_common(10)
            most_common_triplets = triplet_frequencies.most_common(5)
            
            # Pair gap analysis
            pair_gaps = self._analyze_pair_gaps(most_common_pairs)
            
            return {
                'most_common_pairs': most_common_pairs,
                'most_common_triplets': most_common_triplets,
                'total_unique_pairs': len(pair_frequencies),
                'total_unique_triplets': len(triplet_frequencies),
                'pair_gap_analysis': pair_gaps,
                'pair_frequency_stats': {
                    'mean': np.mean(list(pair_frequencies.values())),
                    'std': np.std(list(pair_frequencies.values())),
                    'max': max(pair_frequencies.values()) if pair_frequencies else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Pair pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_sum_patterns(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze sum patterns in draws."""
        try:
            sums = []
            sum_frequencies = Counter()
            
            for sorteo in sorteos:
                numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                total_sum = sum(numbers)
                sums.append(total_sum)
                sum_frequencies[total_sum] += 1
            
            # Sum statistics
            sum_stats = {
                'mean': np.mean(sums),
                'median': np.median(sums),
                'std': np.std(sums),
                'min': min(sums),
                'max': max(sums),
                'range': max(sums) - min(sums)
            }
            
            # Sum ranges analysis
            sum_ranges = {
                'low_sums_0_50': sum(1 for s in sums if 0 <= s <= 50),
                'medium_sums_51_150': sum(1 for s in sums if 51 <= s <= 150),
                'high_sums_151_297': sum(1 for s in sums if 151 <= s <= 297)
            }
            
            # Most/least common sums
            most_common_sums = sum_frequencies.most_common(10)
            least_common_sums = sum_frequencies.most_common()[-10:]
            
            return {
                'sum_statistics': sum_stats,
                'sum_ranges': sum_ranges,
                'most_common_sums': most_common_sums,
                'least_common_sums': least_common_sums,
                'sum_distribution_entropy': self._calculate_entropy(list(sum_frequencies.values())),
                'total_unique_sums': len(sum_frequencies)
            }
            
        except Exception as e:
            logger.error(f"Sum pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_gap_patterns(self, sorteos: List[Sorteo]) -> Dict[str, Any]:
        """Analyze gap patterns between numbers."""
        try:
            gaps = []
            max_gaps = []
            min_gaps = []
            
            for sorteo in sorteos:
                numbers = sorted([sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar])
                
                # Calculate gaps between consecutive numbers
                draw_gaps = []
                for i in range(len(numbers) - 1):
                    gap = numbers[i + 1] - numbers[i]
                    gaps.append(gap)
                    draw_gaps.append(gap)
                
                if draw_gaps:
                    max_gaps.append(max(draw_gaps))
                    min_gaps.append(min(draw_gaps))
            
            # Gap statistics
            gap_stats = {
                'mean_gap': np.mean(gaps) if gaps else 0,
                'median_gap': np.median(gaps) if gaps else 0,
                'std_gap': np.std(gaps) if gaps else 0,
                'min_gap': min(gaps) if gaps else 0,
                'max_gap': max(gaps) if gaps else 0
            }
            
            # Gap frequency analysis
            gap_frequencies = Counter(gaps)
            most_common_gaps = gap_frequencies.most_common(10)
            
            return {
                'gap_statistics': gap_stats,
                'most_common_gaps': most_common_gaps,
                'max_gaps_per_draw': {
                    'mean': np.mean(max_gaps) if max_gaps else 0,
                    'std': np.std(max_gaps) if max_gaps else 0
                },
                'min_gaps_per_draw': {
                    'mean': np.mean(min_gaps) if min_gaps else 0,
                    'std': np.std(min_gaps) if min_gaps else 0
                },
                'gap_distribution': dict(gap_frequencies)
            }
            
        except Exception as e:
            logger.error(f"Gap pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_range_distribution(self, numbers: List[int]) -> Dict[str, int]:
        """Calculate distribution across number ranges."""
        ranges = {
            '0-19': sum(1 for n in numbers if 0 <= n <= 19),
            '20-39': sum(1 for n in numbers if 20 <= n <= 39),
            '40-59': sum(1 for n in numbers if 40 <= n <= 59),
            '60-79': sum(1 for n in numbers if 60 <= n <= 79),
            '80-99': sum(1 for n in numbers if 80 <= n <= 99)
        }
        return ranges
    
    def _calculate_decade_distribution(self, numbers: List[int]) -> Dict[str, int]:
        """Calculate distribution by decades."""
        decades = {}
        for i in range(10):
            decade_key = f'{i*10}-{i*10+9}'
            decades[decade_key] = sum(1 for n in numbers if i*10 <= n <= i*10+9)
        return decades
    
    def _calculate_decade_preference(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate preference score for each decade."""
        decade_counts = self._calculate_decade_distribution(numbers)
        total = len(numbers)
        
        preferences = {}
        for decade, count in decade_counts.items():
            preferences[decade] = count / total if total > 0 else 0
        
        return preferences
    
    def _calculate_quartile_distribution(self, numbers: List[int]) -> Dict[str, int]:
        """Calculate distribution by quartiles."""
        return {
            'Q1_0-24': sum(1 for n in numbers if 0 <= n <= 24),
            'Q2_25-49': sum(1 for n in numbers if 25 <= n <= 49),
            'Q3_50-74': sum(1 for n in numbers if 50 <= n <= 74),
            'Q4_75-99': sum(1 for n in numbers if 75 <= n <= 99)
        }
    
    def _calculate_skewness(self, numbers: List[int]) -> float:
        """Calculate skewness of number distribution."""
        try:
            if len(numbers) < 3:
                return 0.0
            
            mean = np.mean(numbers)
            std = np.std(numbers)
            if std == 0:
                return 0.0
            
            skewness = np.mean([(x - mean)**3 for x in numbers]) / (std**3)
            return skewness
        except:
            return 0.0
    
    def _calculate_kurtosis(self, numbers: List[int]) -> float:
        """Calculate kurtosis of number distribution."""
        try:
            if len(numbers) < 4:
                return 0.0
            
            mean = np.mean(numbers)
            std = np.std(numbers)
            if std == 0:
                return 0.0
            
            kurtosis = np.mean([(x - mean)**4 for x in numbers]) / (std**4) - 3
            return kurtosis
        except:
            return 0.0
    
    def _calculate_entropy(self, frequencies: List[int]) -> float:
        """Calculate entropy of frequency distribution."""
        try:
            total = sum(frequencies)
            if total == 0:
                return 0.0
            
            probabilities = [f / total for f in frequencies if f > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            return entropy
        except:
            return 0.0
    
    def _runs_test(self, numbers: List[int]) -> Dict[str, Any]:
        """Perform runs test for randomness."""
        try:
            if len(numbers) < 2:
                return {'test': 'insufficient_data'}
            
            # Convert to binary sequence (above/below median)
            median = np.median(numbers)
            binary_seq = [1 if n >= median else 0 for n in numbers]
            
            # Count runs
            runs = 1
            for i in range(1, len(binary_seq)):
                if binary_seq[i] != binary_seq[i-1]:
                    runs += 1
            
            # Expected runs and variance
            n1 = sum(binary_seq)
            n2 = len(binary_seq) - n1
            
            if n1 == 0 or n2 == 0:
                return {'test': 'no_variation'}
            
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
            
            # Z-score
            z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
            
            return {
                'observed_runs': runs,
                'expected_runs': expected_runs,
                'z_score': z_score,
                'test_result': 'random' if abs(z_score) < 1.96 else 'non_random'
            }
        except:
            return {'test': 'error'}
    
    def _analyze_pair_gaps(self, pairs: List[Tuple]) -> Dict[str, Any]:
        """Analyze gaps in number pairs."""
        try:
            gaps = []
            for pair, frequency in pairs:
                gap = abs(pair[1] - pair[0])
                gaps.extend([gap] * frequency)
            
            if not gaps:
                return {}
            
            return {
                'mean_gap': np.mean(gaps),
                'most_common_gap': Counter(gaps).most_common(1)[0][0],
                'gap_distribution': dict(Counter(gaps))
            }
        except:
            return {}
    
    def find_similar_historical_patterns(self, target_sequence: List[int], 
                                       tipo_loteria_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Find historically similar patterns using vector similarity."""
        try:
            # Create embedding for target sequence
            target_embedding = create_embedding(target_sequence)
            
            with get_db_connection() as session:
                # Get historical sorteos with embeddings
                query = session.query(Sorteo, Vector).join(
                    Vector, Sorteo.id == Vector.sorteo_id
                ).filter(
                    Sorteo.tipo_loteria_id == tipo_loteria_id
                ).order_by(Sorteo.fecha.desc()).limit(1000)
                
                results = query.all()
                
                if not results:
                    return []
                
                similarities = []
                for sorteo, vector in results:
                    try:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(target_embedding, vector.embedding)
                        
                        if similarity > self.similarity_threshold:
                            similarities.append({
                                'sorteo_id': sorteo.id,
                                'fecha': sorteo.fecha,
                                'numbers': [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar],
                                'similarity': float(similarity),
                                'sum': sorteo.primer_lugar + sorteo.segundo_lugar + sorteo.tercer_lugar
                            })
                    except Exception as e:
                        logger.warning(f"Similarity calculation failed for sorteo {sorteo.id}: {e}")
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"Similar pattern search failed: {e}")
            return []
    
    def detect_anomalies(self, sorteos: List[Sorteo]) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in lottery data."""
        try:
            anomalies = []
            
            if len(sorteos) < 10:
                return anomalies
            
            # Calculate baseline statistics
            all_numbers = []
            sums = []
            gaps = []
            
            for sorteo in sorteos:
                numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                sorted_nums = sorted(numbers)
                all_numbers.extend(numbers)
                sums.append(sum(numbers))
                
                # Calculate gaps
                for i in range(len(sorted_nums) - 1):
                    gaps.append(sorted_nums[i + 1] - sorted_nums[i])
            
            # Calculate baselines
            mean_sum = np.mean(sums)
            std_sum = np.std(sums)
            mean_gap = np.mean(gaps) if gaps else 0
            std_gap = np.std(gaps) if gaps else 0
            
            # Detect anomalies
            for sorteo in sorteos:
                numbers = [sorteo.primer_lugar, sorteo.