#!/usr/bin/env python3
"""
Performance monitoring script for the optimized lottery prediction system.
Monitors system resources, database performance, and processing rates.

Usage:
    python scripts/monitor_performance.py --continuous
    python scripts/monitor_performance.py --report
    python scripts/monitor_performance.py --alert-thresholds
"""

import argparse
import sys
import os
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.database import get_db_connection, get_database_performance_metrics
from predictions.predictor_engine import predictor_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system and application performance."""
    
    def __init__(self):
        self.alert_thresholds = {
            'memory_percent': 85,
            'cpu_percent': 80,
            'disk_usage_percent': 90,
            'db_connections': 80,  # Percentage of max connections
            'processing_rate_min': 10  # Minimum dates per hour
        }
        self.metrics_history = []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            
            # Network metrics (if needed)
            network_io = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg_1m': load_avg[0],
                    'load_avg_5m': load_avg[1],
                    'load_avg_15m': load_avg[2]
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                },
                'swap': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'percent': swap.percent
                },
                'disk': {
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'percent': (disk_usage.used / disk_usage.total) * 100
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        try:
            return get_database_performance_metrics()
        except Exception as e:
            logger.error(f"Failed to get database metrics: {e}")
            return {'error': str(e)}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        try:
            metrics = {}
            
            # Get model performance summary
            try:
                model_summary = predictor_engine.get_model_performance_summary()
                metrics['models'] = {
                    'total_models': len(model_summary),
                    'active_models': len([m for m in model_summary.values() 
                                        if m.get('metadata', {}).get('is_active', True)])
                }
            except Exception as e:
                logger.warning(f"Failed to get model metrics: {e}")
                metrics['models'] = {'error': str(e)}
            
            # Get prediction counts for today
            try:
                with get_db_connection() as session:
                    from models.database_models import PrediccionQuiniela, PrediccionPale, PrediccionTripleta
                    from sqlalchemy import func
                    
                    today = datetime.now().date()
                    
                    quiniela_count = session.query(func.count(PrediccionQuiniela.id)).filter(
                        PrediccionQuiniela.fecha_prediccion == today
                    ).scalar()
                    
                    pale_count = session.query(func.count(PrediccionPale.id)).filter(
                        PrediccionPale.fecha_prediccion == today
                    ).scalar()
                    
                    tripleta_count = session.query(func.count(PrediccionTripleta.id)).filter(
                        PrediccionTripleta.fecha_prediccion == today
                    ).scalar()
                    
                    metrics['predictions_today'] = {
                        'quiniela': quiniela_count or 0,
                        'pale': pale_count or 0,
                        'tripleta': tripleta_count or 0,
                        'total': (quiniela_count or 0) + (pale_count or 0) + (tripleta_count or 0)
                    }
            
            except Exception as e:
                logger.warning(f"Failed to get prediction counts: {e}")
                metrics['predictions_today'] = {'error': str(e)}
            
            # Get cache statistics
            try:
                cache_stats = predictor_engine.limpiar_cache_modelos(0)  # Get count without cleaning
                metrics['cache'] = {
                    'model_cache_files': cache_stats
                }
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                metrics['cache'] = {'error': str(e)}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get application metrics: {e}")
            return {'error': str(e)}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'system': self.get_system_metrics(),
            'database': self.get_database_metrics(),
            'application': self.get_application_metrics()
        }
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        try:
            # System alerts
            system = metrics.get('system', {})
            
            if system.get('cpu', {}).get('percent', 0) > self.alert_thresholds['cpu_percent']:
                alerts.append(f"HIGH CPU: {system['cpu']['percent']:.1f}%")
            
            if system.get('memory', {}).get('percent', 0) > self.alert_thresholds['memory_percent']:
                alerts.append(f"HIGH MEMORY: {system['memory']['percent']:.1f}%")
            
            if system.get('disk', {}).get('percent', 0) > self.alert_thresholds['disk_usage_percent']:
                alerts.append(f"HIGH DISK USAGE: {system['disk']['percent']:.1f}%")
            
            # Database alerts
            database = metrics.get('database', {})
            pool_stats = database.get('pool_stats', {})
            
            if pool_stats.get('checked_out_connections', 0) > 0:
                pool_size = pool_stats.get('pool_size', 1)
                connection_percent = (pool_stats['checked_out_connections'] / pool_size) * 100
                if connection_percent > self.alert_thresholds['db_connections']:
                    alerts.append(f"HIGH DB CONNECTIONS: {connection_percent:.1f}%")
            
            # Application alerts
            app = metrics.get('application', {})
            predictions_today = app.get('predictions_today', {})
            
            if predictions_today.get('total', 0) == 0 and datetime.now().hour > 8:
                alerts.append("NO PREDICTIONS GENERATED TODAY")
            
        except Exception as e:
            alerts.append(f"ALERT CHECK FAILED: {e}")
        
        return alerts
    
    def print_metrics_report(self, metrics: Dict[str, Any]):
        """Print formatted metrics report."""
        print("\n" + "=" * 80)
        print(f"PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # System metrics
        system = metrics.get('system', {})
        if 'error' not in system:
            print("\nSYSTEM METRICS:")
            print(f"  CPU Usage: {system.get('cpu', {}).get('percent', 0):.1f}%")
            print(f"  Memory Usage: {system.get('memory', {}).get('percent', 0):.1f}% "
                  f"({system.get('memory', {}).get('used_gb', 0):.1f}GB / "
                  f"{system.get('memory', {}).get('total_gb', 0):.1f}GB)")
            print(f"  Disk Usage: {system.get('disk', {}).get('percent', 0):.1f}% "
                  f"({system.get('disk', {}).get('used_gb', 0):.1f}GB / "
                  f"{system.get('disk', {}).get('total_gb', 0):.1f}GB)")
            print(f"  Load Average: {system.get('cpu', {}).get('load_avg_1m', 0):.2f} "
                  f"(1m), {system.get('cpu', {}).get('load_avg_5m', 0):.2f} (5m)")
        
        # Database metrics
        database = metrics.get('database', {})
        if 'error' not in database:
            print("\nDATABASE METRICS:")
            pool_stats = database.get('pool_stats', {})
            print(f"  Connection Pool: {pool_stats.get('checked_out_connections', 0)}"
                  f"/{pool_stats.get('pool_size', 0)} active")
            print(f"  Total Checkouts: {pool_stats.get('total_checkouts', 0)}")
            print(f"  Cache Hits: {pool_stats.get('cache_stats', {}).get('hits', 0)}")
            print(f"  Cache Misses: {pool_stats.get('cache_stats', {}).get('misses', 0)}")
            
            if 'active_connections' in database:
                print(f"  Active Connections: {database.get('active_connections', 0)}")
            
            table_stats = database.get('table_stats', [])
            if table_stats:
                print("  Table Statistics:")
                for table in table_stats[:3]:  # Show top 3 tables
                    print(f"    {table.get('tablename', 'unknown')}: "
                          f"{table.get('live_tuples', 0):,} rows")
        
        # Application metrics
        app = metrics.get('application', {})
        if 'error' not in app:
            print("\nAPPLICATION METRICS:")
            
            models = app.get('models', {})
            if 'error' not in models:
                print(f"  Total Models: {models.get('total_models', 0)}")
                print(f"  Active Models: {models.get('active_models', 0)}")
            
            predictions = app.get('predictions_today', {})
            if 'error' not in predictions:
                print(f"  Predictions Today:")
                print(f"    Quiniela: {predictions.get('quiniela', 0)}")
                print(f"    Pale: {predictions.get('pale', 0)}")
                print(f"    Tripleta: {predictions.get('tripleta', 0)}")
                print(f"    Total: {predictions.get('total', 0)}")
            
            cache = app.get('cache', {})
            if 'error' not in cache:
                print(f"  Model Cache Files: {cache.get('model_cache_files', 0)}")
        
        # Check for alerts
        alerts = self.check_alerts(metrics)
        if alerts:
            print(f"\n🚨 ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"  ⚠️  {alert}")
        else:
            print(f"\n✅ No alerts detected")
        
        print("=" * 80)
    
    def continuous_monitoring(self, interval: int = 60):
        """Run continuous monitoring."""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                start_time = time.time()
                
                # Get metrics
                metrics = self.get_all_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 entries
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Print report
                self.print_metrics_report(metrics)
                
                # Check for critical alerts
                alerts = self.check_alerts(metrics)
                critical_alerts = [a for a in alerts if any(word in a.upper() 
                                  for word in ['HIGH', 'CRITICAL', 'FAILED', 'ERROR'])]
                
                if critical_alerts:
                    logger.warning(f"Critical alerts detected: {', '.join(critical_alerts)}")
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def save_metrics_to_file(self, metrics: Dict[str, Any], filename: str = None):
        """Save metrics to JSON file."""
        if not filename:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance monitoring for lottery prediction system")
    
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--report', action='store_true',
                       help='Generate single performance report')
    parser.add_argument('--save-json', type=str,
                       help='Save metrics to JSON file')
    parser.add_argument('--alert-thresholds', action='store_true',
                       help='Show current alert thresholds')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.alert_thresholds:
        print("\nCurrent Alert Thresholds:")
        print("=" * 40)
        for key, value in monitor.alert_thresholds.items():
            print(f"  {key}: {value}")
        print("=" * 40)
        return
    
    if args.continuous:
        monitor.continuous_monitoring(args.interval)
    elif args.report:
        metrics = monitor.get_all_metrics()
        monitor.print_metrics_report(metrics)
        
        if args.save_json:
            monitor.save_metrics_to_file(metrics, args.save_json)
    else:
        # Default: single report
        metrics = monitor.get_all_metrics()
        monitor.print_metrics_report(metrics)


if __name__ == "__main__":
    main()