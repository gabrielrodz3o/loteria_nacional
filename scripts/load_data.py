#!/usr/bin/env python3
"""
Script para llenar la tabla de sorteos desde la API de Lotería Nacional
Scraper optimizado para la lotería dominicana - Agosto 2025

Usage:
    python scripts/fill_sorteos_data.py
    python scripts/fill_sorteos_data.py --dry-run
    python scripts/fill_sorteos_data.py --verbose
    python scripts/fill_sorteos_data.py --no-check-duplicates
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import logging

# Suppress urllib3 warnings for older OpenSSL versions
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create necessary directories
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

from config.settings import settings
from config.database import get_db_connection, init_database, check_database_connection, DatabaseError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Setup logging
log_file = logs_dir / 'sorteos_scraper.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SorteosScraper:
    """Scraper para obtener y almacenar datos de sorteos desde la API."""
    
    # Configuraciones de loterías disponibles
    LOTERIAS_CONFIG = {
        1: {
            'nombre': 'Gana Más',
            'api_id': 12,
            'descripcion': 'Sorteo Gana Más (2:30 PM diario)'
        },
        2: {
            'nombre': 'Lotería Nacional', 
            'api_id': 4,
            'descripcion': 'Lotería Nacional (9:00 PM L-S, 6:00 PM D)'
        }
    }
    
    def __init__(self, tipo_loteria_id: int = None):
        self.api_base_url = "https://api3.bolillerobingoonlinegratis.com/api/sorteos/buscar/combinaciones"
        
        # Si no se especifica tipo, preguntar al usuario
        if tipo_loteria_id is None:
            tipo_loteria_id = self._seleccionar_loteria()
        
        # Validar que el tipo existe
        if tipo_loteria_id not in self.LOTERIAS_CONFIG:
            raise ValueError(f"Tipo de lotería {tipo_loteria_id} no válido. Opciones: {list(self.LOTERIAS_CONFIG.keys())}")
        
        self.tipo_loteria_id = tipo_loteria_id
        self.config_loteria = self.LOTERIAS_CONFIG[tipo_loteria_id]
        self.loteria_externa_id = self.config_loteria['api_id']
        
        self.session = self._create_session()
        self.stats = {
            'total_procesados': 0,
            'insertados': 0,
            'duplicados': 0,
            'errores': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"[INIT] Configurado para: {self.config_loteria['nombre']} (BD ID: {tipo_loteria_id}, API ID: {self.loteria_externa_id})")
    
    def _seleccionar_loteria(self) -> int:
        """Permitir al usuario seleccionar el tipo de lotería."""
        print("\n" + "="*60)
        print("SELECCIÓN DE TIPO DE LOTERÍA")
        print("="*60)
        
        for tipo_id, config in self.LOTERIAS_CONFIG.items():
            print(f"{tipo_id}. {config['nombre']}")
            print(f"   {config['descripcion']}")
            print(f"   API ID: {config['api_id']}")
            print()
        
        while True:
            try:
                seleccion = input("Selecciona el tipo de lotería (1 o 2): ").strip()
                tipo_id = int(seleccion)
                
                if tipo_id in self.LOTERIAS_CONFIG:
                    config_seleccionada = self.LOTERIAS_CONFIG[tipo_id]
                    print(f"\n✅ Seleccionado: {config_seleccionada['nombre']}")
                    return tipo_id
                else:
                    print(f"❌ Opción inválida. Por favor selecciona 1 o 2.")
                    
            except ValueError:
                print("❌ Por favor ingresa un número válido (1 o 2).")
            except KeyboardInterrupt:
                print("\n\n❌ Operación cancelada.")
                exit(1)
    
    def _create_session(self) -> requests.Session:
        """Crear sesión HTTP con retry y timeout configurados."""
        session = requests.Session()
        
        # Configurar retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers simplificados para evitar problemas de codificación
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'es-ES,es;q=0.9',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def fetch_sorteos_data(self) -> List[Dict[str, Any]]:
        """Obtener datos de sorteos desde la API."""
        try:
            logger.info("[API] Obteniendo datos de sorteos desde la API...")
            
            params = {
                'loteria_id': self.loteria_externa_id,
                'voltear': 'false',
                'posicion': 0,
                'numeros': '',
                'jugada': 1
            }
            
            logger.debug(f"[API] URL: {self.api_base_url}")
            logger.debug(f"[API] Params: {params}")
            
            response = self.session.get(
                self.api_base_url,
                params=params,
                timeout=30
            )
            
            logger.info(f"[API] Status Code: {response.status_code}")
            logger.info(f"[API] Content Length: {len(response.content)}")
            logger.debug(f"[API] Response Encoding: {response.encoding}")
            logger.debug(f"[API] Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            logger.debug(f"[API] Content-Encoding: {response.headers.get('Content-Encoding', 'None')}")
            
            response.raise_for_status()
            
            # Asegurarse de que la respuesta se decodifique correctamente
            response.encoding = 'utf-8'
            raw_response = response.text
            
            logger.debug(f"[API] Raw response (first 200 chars): {repr(raw_response[:200])}")
            
            if not raw_response.strip():
                logger.error("[API] Respuesta vacía desde la API")
                return []
            
            # Verificar si la respuesta parece ser texto corrupto/binario
            if any(ord(char) > 127 for char in raw_response[:100]):
                logger.error("[API] La respuesta parece estar corrupta o mal codificada")
                logger.debug(f"[API] Response headers: {dict(response.headers)}")
                
                # Intentar diferentes decodificaciones
                try:
                    # Intentar decodificar como bytes y luego UTF-8
                    if response.content:
                        decoded_content = response.content.decode('utf-8', errors='ignore')
                        logger.debug(f"[API] Decoded content (first 200 chars): {repr(decoded_content[:200])}")
                        raw_response = decoded_content
                except Exception as e:
                    logger.error(f"[API] Error intentando decodificar contenido: {e}")
                    return []
            
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                logger.error(f"[API] Error decodificando JSON: {e}")
                logger.error(f"[API] Raw response que falló: {repr(raw_response[:500])}")
                
                # Intentar limpiar la respuesta
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith('{"') and cleaned_response.endswith('}'):
                    try:
                        data = json.loads(cleaned_response)
                        logger.info("[API] JSON parseado exitosamente después de limpieza")
                    except json.JSONDecodeError as e2:
                        logger.error(f"[API] JSON aún falla después de limpieza: {e2}")
                        return []
                else:
                    logger.error("[API] La respuesta no parece ser JSON válido")
                    return []
            
            if not isinstance(data, dict):
                logger.error(f"[API] Respuesta no es un diccionario: {type(data)}")
                return []
            
            if 'sorteos' not in data:
                logger.error("[API] Respuesta no contiene campo 'sorteos'")
                logger.error(f"[API] Estructura de respuesta: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                return []
            
            sorteos = data['sorteos']
            
            if not isinstance(sorteos, list):
                logger.error(f"[API] Campo 'sorteos' no es una lista: {type(sorteos)}")
                return []
            
            logger.info(f"[API] Obtenidos {len(sorteos)} sorteos desde la API")
            
            # Log sample sorteo for debugging
            if sorteos and len(sorteos) > 0:
                logger.debug(f"[API] Sample sorteo: {sorteos[0]}")
            
            return sorteos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[API] Error en request: {e}")
            return []
        except Exception as e:
            logger.error(f"[API] Error inesperado: {e}")
            return []
    
    def parse_sorteo_data(self, sorteo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parsear datos de un sorteo desde la API."""
        try:
            # Validar campos requeridos
            required_fields = ['fecha_sorteo', 'premios']
            for field in required_fields:
                if field not in sorteo_data or sorteo_data[field] is None:
                    logger.warning(f"[PARSE] Campo requerido '{field}' no encontrado o es nulo en sorteo: {sorteo_data}")
                    return None
            
            # Parsear fecha
            fecha_str = sorteo_data['fecha_sorteo']
            try:
                fecha = datetime.strptime(fecha_str, '%Y-%m-%d').date()
            except ValueError as e:
                logger.warning(f"[PARSE] Formato de fecha inválido '{fecha_str}': {e}")
                return None
            
            # Parsear premios (formato: "83-33-22")
            premios_str = str(sorteo_data['premios']).strip()
            
            # Limpiar espacios extras en los premios
            premios_str = premios_str.replace(' ', '')
            
            try:
                numeros = premios_str.split('-')
                if len(numeros) != 3:
                    logger.warning(f"[PARSE] Formato de premios inválido '{premios_str}': debe tener 3 números separados por guión")
                    return None
                
                # Convertir a enteros y validar
                try:
                    primer_lugar = int(numeros[0])
                    segundo_lugar = int(numeros[1]) 
                    tercer_lugar = int(numeros[2])
                except ValueError as e:
                    logger.warning(f"[PARSE] Error convirtiendo números '{premios_str}': {e}")
                    return None
                
                # Validar rango de números (0-99)
                for num, pos in [(primer_lugar, 'primer'), (segundo_lugar, 'segundo'), (tercer_lugar, 'tercer')]:
                    if not (0 <= num <= 99):
                        logger.warning(f"[PARSE] Número fuera de rango en {pos} lugar: {num}")
                        return None
                
            except (ValueError, IndexError) as e:
                logger.warning(f"[PARSE] Error parseando premios '{premios_str}': {e}")
                return None
            
            # Crear objeto sorteo
            sorteo = {
                'fecha': fecha,
                'tipo_loteria_id': self.tipo_loteria_id,
                'primer_lugar': primer_lugar,
                'segundo_lugar': segundo_lugar,
                'tercer_lugar': tercer_lugar,
                'fuente_scraping': 'api_bolillero_bingo'
            }
            
            logger.debug(f"[PARSE] Sorteo parseado: {fecha} -> {primer_lugar}-{segundo_lugar}-{tercer_lugar}")
            return sorteo
            
        except Exception as e:
            logger.error(f"[PARSE] Error inesperado parseando sorteo: {e}")
            logger.error(f"[PARSE] Datos del sorteo: {sorteo_data}")
            return None
    
    def insert_sorteo_to_db(self, sorteo: Dict[str, Any]) -> bool:
        """Insertar un sorteo en la base de datos."""
        try:
            with get_db_connection() as session:
                # Query de inserción
                insert_query = text("""
                    INSERT INTO sorteos (fecha, tipo_loteria_id, primer_lugar, segundo_lugar, tercer_lugar, fuente_scraping)
                    VALUES (:fecha, :tipo_loteria_id, :primer_lugar, :segundo_lugar, :tercer_lugar, :fuente_scraping)
                """)
                
                session.execute(insert_query, sorteo)
                
                logger.debug(f"[DB] Insertado sorteo: {sorteo['fecha']} -> "
                           f"{sorteo['primer_lugar']}-{sorteo['segundo_lugar']}-{sorteo['tercer_lugar']}")
                
                return True
                
        except IntegrityError as e:
            if "unique_sorteo_fecha_tipo" in str(e):
                logger.debug(f"[DB] Sorteo duplicado para fecha {sorteo['fecha']} (tipo {sorteo['tipo_loteria_id']})")
                self.stats['duplicados'] += 1
                return False
            else:
                logger.error(f"[DB] Error de integridad insertando sorteo {sorteo['fecha']}: {e}")
                self.stats['errores'] += 1
                return False
        except Exception as e:
            logger.error(f"[DB] Error insertando sorteo {sorteo['fecha']}: {e}")
            self.stats['errores'] += 1
            return False
    
    def insert_sorteos_batch(self, sorteos: List[Dict[str, Any]]) -> int:
        """Insertar múltiples sorteos usando batch insert."""
        if not sorteos:
            return 0
        
        try:
            with get_db_connection() as session:
                # Query de inserción batch con ON CONFLICT
                insert_query = text("""
                    INSERT INTO sorteos (fecha, tipo_loteria_id, primer_lugar, segundo_lugar, tercer_lugar, fuente_scraping)
                    VALUES (:fecha, :tipo_loteria_id, :primer_lugar, :segundo_lugar, :tercer_lugar, :fuente_scraping)
                    ON CONFLICT (fecha, tipo_loteria_id) DO NOTHING
                """)
                
                # Ejecutar batch insert
                result = session.execute(insert_query, sorteos)
                inserted_count = result.rowcount
                
                logger.info(f"[DB] Insertados {inserted_count} sorteos nuevos de {len(sorteos)} procesados")
                
                return inserted_count
                
        except Exception as e:
            logger.error(f"[DB] Error en batch insert: {e}")
            return 0
    
    def get_existing_dates(self) -> set:
        """Obtener fechas que ya existen en la BD para evitar duplicados."""
        try:
            with get_db_connection() as session:
                query = text("""
                    SELECT DISTINCT fecha 
                    FROM sorteos 
                    WHERE tipo_loteria_id = :tipo_loteria_id
                """)
                
                result = session.execute(query, {'tipo_loteria_id': self.tipo_loteria_id})
                existing_dates = {row[0] for row in result.fetchall()}
                
                logger.info(f"[DB] Encontradas {len(existing_dates)} fechas existentes en BD")
                return existing_dates
                
        except Exception as e:
            logger.error(f"[DB] Error obteniendo fechas existentes: {e}")
            return set()
    
    def process_sorteos(self, check_duplicates: bool = True) -> Dict[str, Any]:
        """Proceso principal para obtener y almacenar sorteos."""
        self.stats['start_time'] = time.time()
        logger.info("[SCRAPER] Iniciando proceso de scraping de sorteos")
        
        # Obtener datos desde API
        api_sorteos = self.fetch_sorteos_data()
        if not api_sorteos:
            logger.error("[SCRAPER] No se pudieron obtener datos desde la API")
            return self.stats
        
        # Obtener fechas existentes si se solicita verificación
        existing_dates = set()
        if check_duplicates:
            existing_dates = self.get_existing_dates()
        
        # Parsear y filtrar sorteos
        sorteos_validos = []
        for sorteo_data in api_sorteos:
            self.stats['total_procesados'] += 1
            
            # Parsear sorteo
            sorteo = self.parse_sorteo_data(sorteo_data)
            if not sorteo:
                continue
            
            # Verificar duplicados si está habilitado
            if check_duplicates and sorteo['fecha'] in existing_dates:
                self.stats['duplicados'] += 1
                logger.debug(f"[FILTER] Saltando fecha duplicada: {sorteo['fecha']}")
                continue
            
            sorteos_validos.append(sorteo)
        
        logger.info(f"[SCRAPER] Sorteos válidos para insertar: {len(sorteos_validos)}")
        
        # Insertar sorteos en lotes
        if sorteos_validos:
            inserted_count = self.insert_sorteos_batch(sorteos_validos)
            self.stats['insertados'] = inserted_count
        
        self.stats['end_time'] = time.time()
        return self.stats
    
    def get_stats_summary(self) -> str:
        """Obtener resumen de estadísticas."""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            self.stats['duration_seconds'] = duration
        
        summary = f"""
=== RESUMEN DE SCRAPING DE SORTEOS ===
Total procesados: {self.stats['total_procesados']}
Insertados: {self.stats['insertados']}
Duplicados: {self.stats['duplicados']}
Errores: {self.stats['errores']}
Duración: {self.stats.get('duration_seconds', 0):.2f} segundos
        """.strip()
        
        return summary
    
    def validate_database_connection(self) -> bool:
        """Validar conexión a la base de datos y estructura."""
        try:
            with get_db_connection() as session:
                # Verificar que existe la tabla sorteos
                query = text("SELECT COUNT(*) FROM sorteos LIMIT 1")
                result = session.execute(query)
                
                # Verificar que existe el tipo de lotería
                query = text("SELECT id, nombre FROM tipos_loteria WHERE id = :id")
                result = session.execute(query, {'id': self.tipo_loteria_id})
                loteria = result.fetchone()
                
                if not loteria:
                    logger.error(f"[DB] Tipo de lotería con ID {self.tipo_loteria_id} no existe")
                    return False
                
                logger.info(f"[DB] Validación exitosa - Tipo lotería: {loteria[1]} (ID: {loteria[0]})")
                return True
                
        except Exception as e:
            logger.error(f"[DB] Error validando base de datos: {e}")
            return False


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Script para llenar tabla sorteos desde API de Lotería Nacional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Scraping interactivo (te preguntará qué tipo)
    python scripts/fill_sorteos_data.py
    
    # Scraping específico de Gana Más (tipo 1)
    python scripts/fill_sorteos_data.py --tipo-loteria 1
    
    # Scraping específico de Lotería Nacional (tipo 2) 
    python scripts/fill_sorteos_data.py --tipo-loteria 2
    
    # Modo dry-run para probar
    python scripts/fill_sorteos_data.py --tipo-loteria 1 --dry-run
    
    # Sin verificar duplicados (más rápido)
    python scripts/fill_sorteos_data.py --tipo-loteria 2 --no-check-duplicates
    
    # Modo verbose para debugging
    python scripts/fill_sorteos_data.py --tipo-loteria 1 --verbose
        """
    )
    
    parser.add_argument('--tipo-loteria', type=int, choices=[1, 2],
                       help='Tipo de lotería: 1=Gana Más, 2=Lotería Nacional')
    parser.add_argument('--dry-run', action='store_true',
                       help='Mostrar qué se haría sin insertar datos')
    parser.add_argument('--no-check-duplicates', action='store_true',
                       help='No verificar duplicados (más rápido pero puede crear duplicados)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Habilitar logging verbose')
    parser.add_argument('--force', action='store_true',
                       help='Forzar ejecución sin confirmaciones')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("[DEBUG] Modo verbose habilitado")
    
    # Verificar conexión a BD
    logger.info("[INIT] Verificando conexión a base de datos...")
    if not check_database_connection():
        logger.error("[INIT] Error de conexión a base de datos")
        sys.exit(1)
    
    # Crear scraper ANTES del banner
    try:
        scraper = SorteosScraper(tipo_loteria_id=args.tipo_loteria)
    except Exception as e:
        logger.error(f"[INIT] Error creando scraper: {e}")
        sys.exit(1)
    
    # Banner dinámico
    loteria_info = scraper.config_loteria
    logger.info("=" * 70)
    logger.info(f"SCRAPER DE SORTEOS - {loteria_info['nombre'].upper()}")
    logger.info("=" * 70)
    logger.info(f"API Source: bolillerobingoonlinegratis.com")
    logger.info(f"Lotería: {loteria_info['nombre']} ({loteria_info['descripcion']})")
    logger.info(f"API ID: {loteria_info['api_id']} → BD tipo_loteria_id: {scraper.tipo_loteria_id}")
    logger.info(f"Check duplicates: {not args.no_check_duplicates}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Validar configuración de BD
    logger.info("[INIT] Validando configuración de base de datos...")
    if not scraper.validate_database_connection():
        logger.error("[INIT] Error en validación de base de datos")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("[DRY-RUN] === MODO SIMULACIÓN ===")
        
        # Obtener datos para mostrar qué se haría
        api_sorteos = scraper.fetch_sorteos_data()
        if not api_sorteos:
            logger.error("[DRY-RUN] No se pudieron obtener datos desde la API")
            return
        
        logger.info(f"[DRY-RUN] Se obtuvieron {len(api_sorteos)} sorteos desde la API")
        
        # Mostrar algunos ejemplos
        logger.info("[DRY-RUN] Ejemplos de sorteos que se procesarían:")
        for i, sorteo_data in enumerate(api_sorteos[:5]):
            sorteo = scraper.parse_sorteo_data(sorteo_data)
            if sorteo:
                logger.info(f"[DRY-RUN]   {sorteo['fecha']}: {sorteo['primer_lugar']}-{sorteo['segundo_lugar']}-{sorteo['tercer_lugar']}")
        
        if len(api_sorteos) > 5:
            logger.info(f"[DRY-RUN]   ... y {len(api_sorteos) - 5} sorteos más")
        
        logger.info("[DRY-RUN] Simulación completada. Use sin --dry-run para insertar datos.")
        return
    
    # Confirmación para ejecución real
    if not args.force:
        response = input("\n¿Proceder con el scraping e inserción de datos? [y/N]: ")
        if response.lower() not in ['y', 'yes', 'sí', 's']:
            logger.info("[CANCELLED] Operación cancelada por el usuario")
            return
    
    # Ejecutar scraping
    logger.info("[SCRAPER] === INICIANDO SCRAPING ===")
    
    try:
        stats = scraper.process_sorteos(
            check_duplicates=not args.no_check_duplicates
        )
        
        # Mostrar resultados
        logger.info(scraper.get_stats_summary())
        
        # Determinar código de salida
        if stats['errores'] > 0:
            logger.warning(f"[WARNING] Se encontraron {stats['errores']} errores durante el procesamiento")
            sys.exit(1)
        elif stats['insertados'] == 0:
            logger.info("[INFO] No se insertaron nuevos sorteos (posiblemente todos ya existían)")
        else:
            logger.info(f"[SUCCESS] Scraping completado exitosamente - {stats['insertados']} sorteos insertados")
    
    except KeyboardInterrupt:
        logger.info("[INTERRUPTED] Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Error fatal durante el scraping: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()