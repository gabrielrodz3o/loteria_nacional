#!/usr/bin/env python3
"""Script para predicciones con datos preservados."""

import sys
import os
from pathlib import Path
from datetime import date, datetime
import logging

sys.path.insert(0, '/app')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generar predicciones usando datos históricos existentes."""
    logger.info("🚀 Iniciando predicciones con datos preservados...")
    
    try:
        from config.database import get_db_connection, check_database_connection
        from models.database_models import TipoLoteria, Sorteo
        from sqlalchemy import func
        
        if not check_database_connection():
            logger.error("❌ No hay conexión a base de datos")
            return False
        
        with get_db_connection() as session:
            # Verificar datos
            total_registros = session.query(Sorteo).count()
            loterias = session.query(TipoLoteria).filter(TipoLoteria.activo == True).all()
            
            logger.info(f"📊 Datos disponibles: {total_registros:,} registros")
            logger.info(f"🎲 Loterías activas: {len(loterias)}")
            
            if total_registros == 0:
                logger.warning("⚠️ No hay datos históricos para generar predicciones")
                return False
            
            # Mostrar rango de datos
            fecha_min = session.query(func.min(Sorteo.fecha)).scalar()
            fecha_max = session.query(func.max(Sorteo.fecha)).scalar()
            logger.info(f"📅 Rango de datos: {fecha_min} a {fecha_max}")
            
            # Intentar usar el predictor engine si existe
            try:
                from predictions.predictor_engine import predictor_engine
                
                fecha_hoy = date.today()
                predicciones_generadas = 0
                
                for loteria in loterias:
                    logger.info(f"🎯 Procesando {loteria.nombre}...")
                    
                    try:
                        # Entrenamiento
                        resultado_entrenamiento = predictor_engine.entrenar_modelos(
                            fecha=fecha_hoy, 
                            tipo_loteria_id=loteria.id
                        )
                        
                        # Predicciones
                        predicciones = predictor_engine.generar_predicciones_diarias(
                            fecha_hoy.strftime('%Y-%m-%d'), 
                            loteria.id
                        )
                        
                        if predicciones:
                            predictor_engine.insertar_predicciones_en_bd(
                                predicciones, 
                                fecha_hoy, 
                                loteria.id
                            )
                            predicciones_generadas += 1
                            logger.info(f"✅ {loteria.nombre} - Predicciones guardadas")
                        
                    except Exception as e:
                        logger.error(f"Error con {loteria.nombre}: {e}")
                        continue
                
                if predicciones_generadas > 0:
                    logger.info(f"🎉 {predicciones_generadas} loterías procesadas exitosamente")
                    return True
                else:
                    logger.error("❌ No se pudieron generar predicciones")
                    return False
                    
            except ImportError:
                logger.warning("⚠️ Sistema de predicciones avanzado no disponible")
                logger.info("ℹ️  Usando sistema básico...")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error general: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌐 Ver predicciones en:")
        print("  Lotería Nacional: http://localhost:8000/predicciones/hoy/1")
        print("  Gana Más: http://localhost:8000/predicciones/hoy/2")
    sys.exit(0 if success else 1)
