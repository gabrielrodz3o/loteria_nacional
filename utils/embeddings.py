"""Vector embeddings utilities for lottery pattern analysis."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from config.settings import settings
from config.database import get_db_connection
from models.database_models import Vector, Sorteo

logger = logging.getLogger(__name__)


def create_embedding(numbers: List[int], method: str = 'frequency_position') -> np.ndarray:
    """Create vector embedding from lottery numbers."""
    try:
        if method == 'frequency_position':
            return _create_frequency_position_embedding(numbers)
        elif method == 'statistical':
            return _create_statistical_embedding(numbers)
        elif method == 'pattern':
            return _create_pattern_embedding(numbers)
        else:
            # Default to frequency_position
            return _create_frequency_position_embedding(numbers)
            
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        # Return zero vector as fallback
        return np.zeros(settings.embedding_dimensions)


def _create_frequency_position_embedding(numbers: List[int]) -> np.ndarray:
    """Create embedding based on number frequencies and positions."""
    try:
        # Initialize embedding vector
        embedding = np.zeros(settings.embedding_dimensions)
        
        # Part 1: One-hot encoding for numbers (100 dimensions)
        for i, num in enumerate(numbers[:3]):  # Only first 3 numbers
            if 0 <= num <= 99:
                embedding[num] = 1.0
        
        # Part 2: Position-weighted encoding (remaining dimensions)
        remaining_dims = settings.embedding_dimensions - 100
        if remaining_dims > 0:
            section_size = remaining_dims // 4
            
            # Statistical features
            if len(numbers) > 0:
                # Basic statistics
                start_idx = 100
                embedding[start_idx] = np.mean(numbers) / 99.0  # Normalized mean
                embedding[start_idx + 1] = np.std(numbers) / 50.0  # Normalized std
                embedding[start_idx + 2] = len(set(numbers)) / len(numbers) if numbers else 0  # Uniqueness
                
                # Parity features
                even_count = sum(1 for n in numbers if n % 2 == 0)
                embedding[start_idx + 3] = even_count / len(numbers)
                
                # Range features
                if len(numbers) >= 3:
                    embedding[start_idx + 4] = (max(numbers) - min(numbers)) / 99.0  # Normalized range
                
                # Decade distribution
                for i in range(10):
                    decade_count = sum(1 for n in numbers if i*10 <= n < (i+1)*10)
                    if start_idx + 5 + i < len(embedding):
                        embedding[start_idx + 5 + i] = decade_count / len(numbers)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Frequency-position embedding creation failed: {e}")
        return np.zeros(settings.embedding_dimensions)


def _create_statistical_embedding(numbers: List[int]) -> np.ndarray:
    """Create embedding based on statistical features."""
    try:
        embedding = np.zeros(settings.embedding_dimensions)
        
        if not numbers:
            return embedding
        
        # Convert to numpy array for easier computation
        nums = np.array(numbers)
        
        # Basic statistics (normalized)
        embedding[0] = np.mean(nums) / 99.0
        embedding[1] = np.std(nums) / 50.0
        embedding[2] = np.median(nums) / 99.0
        embedding[3] = (np.max(nums) - np.min(nums)) / 99.0
        
        # Distribution features
        embedding[4] = len(np.unique(nums)) / len(nums)  # Uniqueness ratio
        embedding[5] = np.sum(nums % 2 == 0) / len(nums)  # Even ratio
        embedding[6] = np.sum(nums >= 50) / len(nums)  # High number ratio
        
        # Sequence features
        if len(nums) > 1:
            diffs = np.diff(sorted(nums))
            embedding[7] = np.mean(diffs) / 99.0
            embedding[8] = np.std(diffs) / 50.0
            
            # Check for arithmetic progression
            if len(set(diffs)) == 1:
                embedding[9] = 1.0
        
        # Digit features
        digit_sum = sum(sum(int(d) for d in str(n).zfill(2)) for n in nums)
        embedding[10] = digit_sum / (len(nums) * 18)  # Max digit sum per number is 18 (99)
        
        # Pattern features
        # Same decade count
        decades = [n // 10 for n in nums]
        max_same_decade = max(decades.count(d) for d in set(decades))
        embedding[11] = max_same_decade / len(nums)
        
        # Ending digit patterns
        endings = [n % 10 for n in nums]
        max_same_ending = max(endings.count(e) for e in set(endings))
        embedding[12] = max_same_ending / len(nums)
        
        # Fill remaining dimensions with harmonic features
        for i in range(13, min(settings.embedding_dimensions, 50)):
            harmonic_idx = i - 13
            if harmonic_idx < len(nums):
                # Use sine/cosine encoding for cyclic patterns
                embedding[i] = np.sin(2 * np.pi * nums[harmonic_idx] / 100.0)
                if i + 1 < settings.embedding_dimensions:
                    embedding[i + 1] = np.cos(2 * np.pi * nums[harmonic_idx] / 100.0)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Statistical embedding creation failed: {e}")
        return np.zeros(settings.embedding_dimensions)


def _create_pattern_embedding(numbers: List[int]) -> np.ndarray:
    """Create embedding focusing on pattern features."""
    try:
        embedding = np.zeros(settings.embedding_dimensions)
        
        if not numbers:
            return embedding
        
        # Pattern-based features
        nums = sorted(numbers)
        
        # Consecutive patterns
        consecutive_pairs = 0
        for i in range(len(nums) - 1):
            if nums[i + 1] == nums[i] + 1:
                consecutive_pairs += 1
        embedding[0] = consecutive_pairs / max(1, len(nums) - 1)
        
        # Arithmetic progression check
        if len(nums) >= 3:
            diffs = [nums[i+1] - nums[i] for i in range(len(nums) - 1)]
            if len(set(diffs)) == 1:
                embedding[1] = 1.0  # Perfect arithmetic progression
            else:
                embedding[1] = 1.0 / len(set(diffs))  # Inverse of unique differences
        
        # Sum patterns
        total_sum = sum(numbers)
        embedding[2] = (total_sum % 10) / 10.0  # Last digit of sum
        embedding[3] = total_sum / 297.0  # Normalized sum (max possible sum is 97+98+99)
        
        # Parity patterns
        parities = [n % 2 for n in numbers]
        embedding[4] = sum(parities) / len(parities)  # Odd ratio
        
        # Digital root
        digital_root = total_sum
        while digital_root >= 10:
            digital_root = sum(int(d) for d in str(digital_root))
        embedding[5] = digital_root / 9.0
        
        # Position-based patterns (for lottery draws)
        if len(numbers) >= 3:
            # First-second difference
            embedding[6] = abs(numbers[1] - numbers[0]) / 99.0
            # Second-third difference  
            embedding[7] = abs(numbers[2] - numbers[1]) / 99.0
            # First-third difference
            embedding[8] = abs(numbers[2] - numbers[0]) / 99.0
        
        # Frequency-based encoding (simplified)
        for i, num in enumerate(numbers[:min(10, len(numbers))]):
            if 9 + i < settings.embedding_dimensions:
                embedding[9 + i] = num / 99.0
        
        return embedding
        
    except Exception as e:
        logger.error(f"Pattern embedding creation failed: {e}")
        return np.zeros(settings.embedding_dimensions)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        # Handle different input types
        if hasattr(vec1, 'tolist'):
            vec1 = np.array(vec1.tolist())
        if hasattr(vec2, 'tolist'):
            vec2 = np.array(vec2.tolist())
        
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        if len(vec1) != len(vec2):
            logger.warning(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    try:
        if hasattr(vec1, 'tolist'):
            vec1 = np.array(vec1.tolist())
        if hasattr(vec2, 'tolist'):
            vec2 = np.array(vec2.tolist())
        
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        if len(vec1) != len(vec2):
            logger.warning(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
            return float('inf')
        
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
        
    except Exception as e:
        logger.error(f"Euclidean distance calculation failed: {e}")
        return float('inf')


def store_embedding(sorteo_id: int, numbers: List[int], method: str = 'frequency_position') -> bool:
    """Store embedding vector in database."""
    try:
        # Create embedding
        embedding = create_embedding(numbers, method)
        
        if np.allclose(embedding, 0):
            logger.warning(f"Zero embedding created for sorteo {sorteo_id}")
            return False
        
        # Store in database
        with get_db_connection() as session:
            # Check if embedding already exists
            existing = session.query(Vector).filter(Vector.sorteo_id == sorteo_id).first()
            
            if existing:
                # Update existing embedding
                existing.embedding = embedding
                logger.info(f"Updated embedding for sorteo {sorteo_id}")
            else:
                # Create new embedding
                vector_entry = Vector(
                    sorteo_id=sorteo_id,
                    embedding=embedding
                )
                session.add(vector_entry)
                logger.info(f"Created embedding for sorteo {sorteo_id}")
            
            session.commit()
            return True
            
    except Exception as e:
        logger.error(f"Embedding storage failed for sorteo {sorteo_id}: {e}")
        return False


def find_similar_vectors(target_embedding: np.ndarray, limit: int = 10, 
                        similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Find similar vectors in database using cosine similarity."""
    try:
        with get_db_connection() as session:
            # Get all vectors (in practice, you might want to limit this)
            vectors = session.query(Vector, Sorteo).join(
                Sorteo, Vector.sorteo_id == Sorteo.id
            ).limit(1000).all()  # Limit for performance
            
            similarities = []
            
            for vector, sorteo in vectors:
                try:
                    similarity = cosine_similarity(target_embedding, vector.embedding)
                    
                    if similarity >= similarity_threshold:
                        similarities.append({
                            'sorteo_id': sorteo.id,
                            'fecha': sorteo.fecha,
                            'numbers': [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar],
                            'similarity': similarity
                        })
                        
                except Exception as e:
                    logger.warning(f"Similarity calculation failed for vector {vector.id}: {e}")
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:limit]
            
    except Exception as e:
        logger.error(f"Similar vector search failed: {e}")
        return []


def batch_create_embeddings(sorteos: List[Sorteo], method: str = 'frequency_position') -> int:
    """Create embeddings for multiple sorteos in batch."""
    try:
        success_count = 0
        
        with get_db_connection() as session:
            for sorteo in sorteos:
                try:
                    numbers = [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar]
                    embedding = create_embedding(numbers, method)
                    
                    if not np.allclose(embedding, 0):
                        # Check if embedding already exists
                        existing = session.query(Vector).filter(Vector.sorteo_id == sorteo.id).first()
                        
                        if not existing:
                            vector_entry = Vector(
                                sorteo_id=sorteo.id,
                                embedding=embedding
                            )
                            session.add(vector_entry)
                            success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to create embedding for sorteo {sorteo.id}: {e}")
                    continue
            
            session.commit()
            logger.info(f"Created {success_count} embeddings in batch")
            return success_count
            
    except Exception as e:
        logger.error(f"Batch embedding creation failed: {e}")
        return 0


def update_all_embeddings(method: str = 'frequency_position') -> int:
    """Update embeddings for all sorteos without vectors."""
    try:
        with get_db_connection() as session:
            # Find sorteos without embeddings
            sorteos_without_vectors = session.query(Sorteo).outerjoin(Vector).filter(
                Vector.sorteo_id.is_(None)
            ).all()
            
            logger.info(f"Found {len(sorteos_without_vectors)} sorteos without embeddings")
            
            return batch_create_embeddings(sorteos_without_vectors, method)
            
    except Exception as e:
        logger.error(f"Embedding update failed: {e}")
        return 0


def calculate_embedding_statistics() -> Dict[str, Any]:
    """Calculate statistics about stored embeddings."""
    try:
        with get_db_connection() as session:
            # Count total embeddings
            total_vectors = session.query(Vector).count()
            total_sorteos = session.query(Sorteo).count()
            
            # Calculate coverage
            coverage = total_vectors / total_sorteos if total_sorteos > 0 else 0
            
            # Get sample of embeddings for analysis
            sample_vectors = session.query(Vector).limit(100).all()
            
            if sample_vectors:
                # Convert to numpy array
                embeddings_array = np.array([v.embedding for v in sample_vectors])
                
                # Calculate statistics
                mean_embedding = np.mean(embeddings_array, axis=0)
                std_embedding = np.std(embeddings_array, axis=0)
                
                stats = {
                    'total_vectors': total_vectors,
                    'total_sorteos': total_sorteos,
                    'coverage_percentage': coverage * 100,
                    'embedding_dimensions': len(mean_embedding),
                    'mean_magnitude': float(np.linalg.norm(mean_embedding)),
                    'std_magnitude': float(np.linalg.norm(std_embedding)),
                    'sample_size': len(sample_vectors)
                }
            else:
                stats = {
                    'total_vectors': total_vectors,
                    'total_sorteos': total_sorteos,
                    'coverage_percentage': 0,
                    'embedding_dimensions': 0,
                    'sample_size': 0
                }
            
            return stats
            
    except Exception as e:
        logger.error(f"Embedding statistics calculation failed: {e}")
        return {'error': str(e)}


def cluster_embeddings(n_clusters: int = 5) -> Dict[str, Any]:
    """Cluster embeddings to find pattern groups."""
    try:
        # Import clustering libraries
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        with get_db_connection() as session:
            # Get recent embeddings
            vectors = session.query(Vector, Sorteo).join(
                Sorteo, Vector.sorteo_id == Sorteo.id
            ).order_by(Sorteo.fecha.desc()).limit(500).all()
            
            if len(vectors) < n_clusters:
                return {'error': 'Not enough vectors for clustering'}
            
            # Extract embeddings
            embeddings = np.array([v[0].embedding for v in vectors])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Organize results by cluster
            clusters = {}
            for i, (vector, sorteo) in enumerate(vectors):
                cluster_id = int(cluster_labels[i])
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = {
                        'center': kmeans.cluster_centers_[cluster_id].tolist(),
                        'members': [],
                        'size': 0
                    }
                
                clusters[cluster_id]['members'].append({
                    'sorteo_id': sorteo.id,
                    'fecha': sorteo.fecha.isoformat(),
                    'numbers': [sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar],
                    'coords_2d': embeddings_2d[i].tolist()
                })
                clusters[cluster_id]['size'] += 1
            
            return {
                'n_clusters': n_clusters,
                'total_vectors': len(vectors),
                'clusters': clusters,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'inertia': float(kmeans.inertia_)
            }
            
    except Exception as e:
        logger.error(f"Embedding clustering failed: {e}")
        return {'error': str(e)}


def embedding_quality_check() -> Dict[str, Any]:
    """Check quality of stored embeddings."""
    try:
        with get_db_connection() as session:
            vectors = session.query(Vector).limit(100).all()
            
            if not vectors:
                return {'error': 'No vectors found'}
            
            quality_metrics = {
                'total_checked': len(vectors),
                'zero_vectors': 0,
                'invalid_dimensions': 0,
                'valid_vectors': 0,
                'dimension_consistency': True,
                'average_magnitude': 0.0
            }
            
            magnitudes = []
            expected_dim = settings.embedding_dimensions
            
            for vector in vectors:
                try:
                    embedding = np.array(vector.embedding)
                    
                    # Check dimensions
                    if len(embedding) != expected_dim:
                        quality_metrics['invalid_dimensions'] += 1
                        quality_metrics['dimension_consistency'] = False
                    
                    # Check for zero vectors
                    if np.allclose(embedding, 0):
                        quality_metrics['zero_vectors'] += 1
                    else:
                        quality_metrics['valid_vectors'] += 1
                        magnitudes.append(np.linalg.norm(embedding))
                
                except Exception as e:
                    logger.warning(f"Vector quality check failed for vector {vector.id}: {e}")
                    quality_metrics['invalid_dimensions'] += 1
            
            if magnitudes:
                quality_metrics['average_magnitude'] = float(np.mean(magnitudes))
                quality_metrics['magnitude_std'] = float(np.std(magnitudes))
            
            return quality_metrics
            
    except Exception as e:
        logger.error(f"Embedding quality check failed: {e}")
        return {'error': str(e)}