"""Cache management utilities using Redis."""

import redis
import json
import pickle
from typing import Any, Optional, List, Dict
import logging
from datetime import datetime, timedelta
from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching operations with Redis."""
    
    def __init__(self):
        self.redis_client = None
        self.is_connected = False
        self._connect()
    
    def _connect(self):
        """Connect to Redis server."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=False,  # Keep as bytes for pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.is_connected = True
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.is_connected = False
            self.redis_client = None
    
    def initialize(self):
        """Initialize cache connection."""
        if not self.is_connected:
            self._connect()
    
    def close(self):
        """Close Redis connection."""
        try:
            if self.redis_client:
                self.redis_client.close()
                self.is_connected = False
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self.is_connected:
            logger.warning("Cache not connected, cannot set value")
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL
            if ttl:
                result = self.redis_client.setex(key, ttl, serialized_value)
            else:
                result = self.redis_client.set(key, serialized_value)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache set failed for key '{key}': {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_connected:
            logger.warning("Cache not connected, cannot get value")
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return pickle.loads(value)
                except:
                    # If both fail, return as string
                    return value.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Cache get failed for key '{key}': {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_connected:
            return False
        
        try:
            result = self.redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete failed for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists check failed for key '{key}': {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.is_connected:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache pattern delete failed for pattern '{pattern}': {e}")
            return 0
    
    def clear_all(self) -> int:
        """Clear all cache entries."""
        if not self.is_connected:
            return 0
        
        try:
            keys = self.redis_client.keys('*')
            if keys:
                count = self.redis_client.delete(*keys)
                logger.info(f"Cleared {count} cache entries")
                return count
            return 0
        except Exception as e:
            logger.error(f"Cache clear all failed: {e}")
            return 0
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key."""
        if not self.is_connected:
            return None
        
        try:
            ttl = self.redis_client.ttl(key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            logger.error(f"TTL check failed for key '{key}': {e}")
            return None
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend TTL for a key."""
        if not self.is_connected:
            return False
        
        try:
            current_ttl = self.redis_client.ttl(key)
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                return bool(self.redis_client.expire(key, new_ttl))
            return False
        except Exception as e:
            logger.error(f"TTL extension failed for key '{key}': {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics."""
        if not self.is_connected:
            return {'connected': False, 'error': 'Not connected to Redis'}
        
        try:
            info = self.redis_client.info()
            memory_info = self.redis_client.info('memory')
            
            # Count keys by pattern
            total_keys = len(self.redis_client.keys('*'))
            prediction_keys = len(self.redis_client.keys('predicciones_*'))
            stats_keys = len(self.redis_client.keys('estadisticas_*'))
            
            return {
                'connected': True,
                'redis_version': info.get('redis_version'),
                'total_keys': total_keys,
                'prediction_cache_keys': prediction_keys,
                'statistics_cache_keys': stats_keys,
                'memory_used': memory_info.get('used_memory_human'),
                'memory_peak': memory_info.get('used_memory_peak_human'),
                'connected_clients': info.get('connected_clients'),
                'uptime_seconds': info.get('uptime_in_seconds')
            }
            
        except Exception as e:
            logger.error(f"Cache info retrieval failed: {e}")
            return {'connected': False, 'error': str(e)}
    
    def set_hash(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash in Redis."""
        if not self.is_connected:
            return False
        
        try:
            # Serialize hash values
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v, default=str)
                else:
                    serialized_mapping[k] = str(v)
            
            result = self.redis_client.hset(key, mapping=serialized_mapping)
            
            if ttl:
                self.redis_client.expire(key, ttl)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache hash set failed for key '{key}': {e}")
            return False
    
    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis."""
        if not self.is_connected:
            return None
        
        try:
            hash_data = self.redis_client.hgetall(key)
            if not hash_data:
                return None
            
            # Deserialize hash values
            result = {}
            for k, v in hash_data.items():
                k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                v_str = v.decode('utf-8') if isinstance(v, bytes) else v
                
                try:
                    result[k_str] = json.loads(v_str)
                except json.JSONDecodeError:
                    result[k_str] = v_str
            
            return result
            
        except Exception as e:
            logger.error(f"Cache hash get failed for key '{key}': {e}")
            return None
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter."""
        if not self.is_connected:
            return None
        
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment failed for key '{key}': {e}")
            return None
    
    def get_list(self, key: str) -> List[Any]:
        """Get list from Redis."""
        if not self.is_connected:
            return []
        
        try:
            list_data = self.redis_client.lrange(key, 0, -1)
            result = []
            
            for item in list_data:
                try:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    result.append(json.loads(item))
                except json.JSONDecodeError:
                    result.append(item)
            
            return result
            
        except Exception as e:
            logger.error(f"Cache list get failed for key '{key}': {e}")
            return []
    
    def push_to_list(self, key: str, value: Any, max_length: Optional[int] = None) -> bool:
        """Push value to list with optional max length."""
        if not self.is_connected:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            # Push to list
            self.redis_client.lpush(key, serialized_value)
            
            # Trim list if max_length specified
            if max_length:
                self.redis_client.ltrim(key, 0, max_length - 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache list push failed for key '{key}': {e}")
            return False


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(self, ttl: int = 3600, key_prefix: str = ""):
        self.ttl = ttl
        self.key_prefix = key_prefix
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{self.key_prefix}{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, self.ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper


# Utility functions
def cache_predictions(tipo_loteria_id: int, predictions: Dict[str, Any], 
                     ttl: Optional[int] = None) -> bool:
    """Cache predictions for a lottery type."""
    key = f"predicciones_hoy_{tipo_loteria_id}"
    ttl = ttl or settings.cache_ttl_predictions
    return cache_manager.set(key, predictions, ttl)


def get_cached_predictions(tipo_loteria_id: int) -> Optional[Dict[str, Any]]:
    """Get cached predictions for a lottery type."""
    key = f"predicciones_hoy_{tipo_loteria_id}"
    return cache_manager.get(key)


def cache_statistics(stats_type: str, data: Dict[str, Any], 
                    ttl: Optional[int] = None) -> bool:
    """Cache statistics data."""
    key = f"estadisticas_{stats_type}"
    ttl = ttl or settings.cache_ttl_statistics
    return cache_manager.set(key, data, ttl)


def get_cached_statistics(stats_type: str) -> Optional[Dict[str, Any]]:
    """Get cached statistics."""
    key = f"estadisticas_{stats_type}"
    return cache_manager.get(key)


def invalidate_prediction_cache(tipo_loteria_id: Optional[int] = None) -> int:
    """Invalidate prediction cache entries."""
    if tipo_loteria_id:
        pattern = f"predicciones_*_{tipo_loteria_id}"
    else:
        pattern = "predicciones_*"
    
    return cache_manager.delete_pattern(pattern)


def warm_cache():
    """Warm up cache with commonly accessed data."""
    try:
        logger.info("Warming up cache...")
        
        # This would typically involve pre-loading frequently accessed data
        # For now, we'll just ensure the connection is working
        cache_manager.set("cache_warmed", datetime.now().isoformat(), 300)
        
        logger.info("Cache warmed successfully")
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")


# Global cache manager instance
cache_manager = CacheManager()