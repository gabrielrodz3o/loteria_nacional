"""FastAPI routes for the lottery prediction system."""

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from datetime import date, datetime
import logging
from sqlalchemy import text

from api.schemas import (
    PredictionResponse, SorteoInput, SorteoResponse, 
    StatisticsResponse, ModelPerformanceResponse,
    ErrorResponse
)
from predictions.predictor_engine import predictor_engine
from config.database import get_db_connection
from models.database_models import Sorteo, TipoLoteria, TipoJuego, PrediccionQuiniela, PrediccionPale, PrediccionTripleta
from utils.cache import cache_manager
from config.settings import settings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Lotería Nacional Dominicana - Sistema de Predicciones",
    description="API para predicciones de lotería usando machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        detail="Internal server error",
        status_code=500
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Lotería Nacional Dominicana - API de Predicciones",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        with get_db_connection() as session:
            session.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/predicciones/hoy/{tipo_loteria}", 
         response_model=PredictionResponse,
         tags=["Predicciones"])
@limiter.limit("30/minute")
async def get_predicciones_hoy(
    request: Request,
    tipo_loteria: int = Path(..., description="ID del tipo de lotería", ge=1)
):
    """Obtener predicciones del día actual."""
    try:
        # Check cache first
        cache_key = f"predicciones_hoy_{tipo_loteria}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached predictions for lottery type {tipo_loteria}")
            return cached_result
        
        # Get predictions from engine
        predicciones = predictor_engine.obtener_predicciones_hoy(tipo_loteria)
        
        if not predicciones:
            raise HTTPException(
                status_code=404, 
                detail=f"No hay predicciones disponibles para el tipo de lotería {tipo_loteria}"
            )
        
        # Validate lottery type exists
        with get_db_connection() as session:
            lottery_type = session.query(TipoLoteria).filter(
                TipoLoteria.id == tipo_loteria,
                TipoLoteria.activo == True
            ).first()
            
            if not lottery_type:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tipo de lotería {tipo_loteria} no encontrado"
                )
        
        result = PredictionResponse(**predicciones)
        
        # Cache the result
        cache_manager.set(cache_key, result, ttl=settings.cache_ttl_predictions)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting today's predictions: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/predicciones/{fecha}/{tipo_loteria}",
         response_model=PredictionResponse,
         tags=["Predicciones"])
@limiter.limit("20/minute")
async def get_predicciones_fecha(
    request: Request,
    fecha: date = Path(..., description="Fecha en formato YYYY-MM-DD"),
    tipo_loteria: int = Path(..., description="ID del tipo de lotería", ge=1)
):
    """Obtener predicciones para una fecha específica."""
    try:
        # Check cache
        cache_key = f"predicciones_{fecha}_{tipo_loteria}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Validate date is not too far in the future
        if fecha > date.today().replace(day=date.today().day + 7):
            raise HTTPException(
                status_code=400,
                detail="No se pueden obtener predicciones para fechas muy futuras"
            )
        
        # Get predictions from database
        with get_db_connection() as session:
            # Check if lottery type exists
            lottery_type = session.query(TipoLoteria).filter(
                TipoLoteria.id == tipo_loteria
            ).first()
            
            if not lottery_type:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tipo de lotería {tipo_loteria} no encontrado"
                )
            
            # Get predictions from database
            predicciones = {}
            
            # Query each prediction type
    
            
            # Quiniela
            quinielas = session.query(PrediccionQuiniela).filter(
                PrediccionQuiniela.fecha_prediccion == fecha,
                PrediccionQuiniela.tipo_loteria_id == tipo_loteria
            ).order_by(PrediccionQuiniela.posicion).all()
            
            predicciones['quiniela'] = [
                {
                    'posicion': q.posicion,
                    'numero': q.numero_predicho,
                    'probabilidad': q.probabilidad,
                    'metodo_generacion': q.metodo_generacion,
                    'score_confianza': q.score_confianza
                } for q in quinielas
            ]
            
            # Pale
            pales = session.query(PrediccionPale).filter(
                PrediccionPale.fecha_prediccion == fecha,
                PrediccionPale.tipo_loteria_id == tipo_loteria
            ).order_by(PrediccionPale.posicion).all()
            
            predicciones['pale'] = [
                {
                    'posicion': p.posicion,
                    'numeros': [p.numero_1, p.numero_2],
                    'probabilidad': p.probabilidad,
                    'metodo_generacion': p.metodo_generacion,
                    'score_confianza': p.score_confianza
                } for p in pales
            ]
            
            # Tripleta
            tripletas = session.query(PrediccionTripleta).filter(
                PrediccionTripleta.fecha_prediccion == fecha,
                PrediccionTripleta.tipo_loteria_id == tipo_loteria
            ).order_by(PrediccionTripleta.posicion).all()
            
            predicciones['tripleta'] = [
                {
                    'posicion': t.posicion,
                    'numeros': [t.numero_1, t.numero_2, t.numero_3],
                    'probabilidad': t.probabilidad,
                    'metodo_generacion': t.metodo_generacion,
                    'score_confianza': t.score_confianza
                } for t in tripletas
            ]
            
            if not any(predicciones.values()):
                raise HTTPException(
                    status_code=404,
                    detail=f"No hay predicciones para la fecha {fecha}"
                )
            
            predicciones.update({
                'fecha': fecha.strftime('%Y-%m-%d'),
                'tipo_loteria_id': tipo_loteria
            })
            
            result = PredictionResponse(**predicciones)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl=settings.cache_ttl_predictions)
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for date {fecha}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/sorteos/resultado",
          response_model=SorteoResponse,
          tags=["Sorteos"])
@limiter.limit("10/minute")
async def registrar_resultado_sorteo(
    request: Request,
    sorteo_data: SorteoInput
):
    """Registrar resultado de un sorteo."""
    try:
        with get_db_connection() as session:
            # Check if sorteo already exists
            existing = session.query(Sorteo).filter(
                Sorteo.fecha == sorteo_data.fecha,
                Sorteo.tipo_loteria_id == sorteo_data.tipo_loteria_id
            ).first()
            
            if existing:
                raise HTTPException(
                    status_code=409,
                    detail=f"Ya existe un sorteo para la fecha {sorteo_data.fecha}"
                )
            
            # Validate lottery type
            lottery_type = session.query(TipoLoteria).filter(
                TipoLoteria.id == sorteo_data.tipo_loteria_id
            ).first()
            
            if not lottery_type:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tipo de lotería {sorteo_data.tipo_loteria_id} no encontrado"
                )
            
            # Create new sorteo
            nuevo_sorteo = Sorteo(
                fecha=sorteo_data.fecha,
                tipo_loteria_id=sorteo_data.tipo_loteria_id,
                primer_lugar=sorteo_data.primer_lugar,
                segundo_lugar=sorteo_data.segundo_lugar,
                tercer_lugar=sorteo_data.tercer_lugar,
                fuente_scraping=sorteo_data.fuente_scraping
            )
            
            session.add(nuevo_sorteo)
            session.commit()
            session.refresh(nuevo_sorteo)
            
            # Invalidate prediction caches
            cache_manager.delete_pattern(f"predicciones_*_{sorteo_data.tipo_loteria_id}")
            
            logger.info(f"Sorteo registered: {nuevo_sorteo.id}")
            
            return SorteoResponse(
                id=nuevo_sorteo.id,
                fecha=nuevo_sorteo.fecha,
                tipo_loteria_id=nuevo_sorteo.tipo_loteria_id,
                primer_lugar=nuevo_sorteo.primer_lugar,
                segundo_lugar=nuevo_sorteo.segundo_lugar,
                tercer_lugar=nuevo_sorteo.tercer_lugar,
                fuente_scraping=nuevo_sorteo.fuente_scraping,
                creado_en=nuevo_sorteo.creado_en
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering sorteo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/estadisticas/metodos",
         response_model=ModelPerformanceResponse,
         tags=["Estadísticas"])
@limiter.limit("15/minute")
async def get_estadisticas_metodos(request: Request):
    """Obtener estadísticas de rendimiento de los métodos de predicción."""
    try:
        # Check cache
        cache_key = "estadisticas_metodos"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Get performance summary from predictor engine
        performance_summary = predictor_engine.get_model_performance_summary()
        
        if not performance_summary:
            # Get basic statistics from database
            with get_db_connection() as session:
                # Use raw SQL to get method statistics
                result = session.execute("""
                    SELECT 
                        rp.tipo_juego_id,
                        tj.nombre as tipo_juego,
                        CASE 
                            WHEN rp.tipo_juego_id = 1 THEN pq.metodo_generacion
                            WHEN rp.tipo_juego_id = 2 THEN pp.metodo_generacion
                            WHEN rp.tipo_juego_id = 3 THEN pt.metodo_generacion
                        END as metodo,
                        COUNT(*) as total_predicciones,
                        SUM(CASE WHEN rp.acierto THEN 1 ELSE 0 END) as aciertos,
                        ROUND(AVG(CASE WHEN rp.acierto THEN 1.0 ELSE 0.0 END) * 100, 2) as porcentaje_acierto
                    FROM resultados_predicciones rp
                    JOIN tipos_juego tj ON rp.tipo_juego_id = tj.id
                    LEFT JOIN predicciones_quiniela pq ON rp.tipo_juego_id = 1 AND rp.prediccion_id = pq.id
                    LEFT JOIN predicciones_pale pp ON rp.tipo_juego_id = 2 AND rp.prediccion_id = pp.id
                    LEFT JOIN predicciones_tripleta pt ON rp.tipo_juego_id = 3 AND rp.prediccion_id = pt.id
                    GROUP BY rp.tipo_juego_id, tj.nombre, metodo
                    ORDER BY rp.tipo_juego_id, porcentaje_acierto DESC
                """).fetchall()
                
                # Format results
                performance_summary = {}
                for row in result:
                    method_name = row[2] or 'unknown'
                    performance_summary[method_name] = {
                        'performance': {
                            'avg_accuracy': row[5] / 100.0,
                            'recent_accuracy': row[5] / 100.0,
                            'evaluations': row[3]
                        },
                        'metadata': {
                            'description': f'Method for {row[1]}',
                            'version': '1.0',
                            'is_active': True
                        }
                    }
        
        result = ModelPerformanceResponse(
            timestamp=datetime.now(),
            models=performance_summary
        )
        
        # Cache result
        cache_manager.set(cache_key, result, ttl=settings.cache_ttl_statistics)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting method statistics: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/estadisticas/historico",
         response_model=StatisticsResponse,
         tags=["Estadísticas"])
@limiter.limit("10/minute")
async def get_estadisticas_historico(
    request,
    tipo_loteria_id: Optional[int] = Query(None, description="ID del tipo de lotería"),
    dias: int = Query(30, description="Número de días hacia atrás", ge=1, le=365)
):
    """Obtener estadísticas históricas de predicciones."""
    try:
        cache_key = f"estadisticas_historico_{tipo_loteria_id}_{dias}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            return cached_result
        
        with get_db_connection() as session:
            # Build query for historical data
            query = session.query(Sorteo)
            
            if tipo_loteria_id:
                query = query.filter(Sorteo.tipo_loteria_id == tipo_loteria_id)
            
            # Get recent sorteos
            cutoff_date = date.today().replace(day=date.today().day - dias)
            sorteos = query.filter(Sorteo.fecha >= cutoff_date).order_by(Sorteo.fecha.desc()).all()
            
            # Calculate statistics
            total_sorteos = len(sorteos)
            
            if total_sorteos == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No hay datos históricos para el período solicitado"
                )
            
            # Number frequency analysis
            all_numbers = []
            for sorteo in sorteos:
                all_numbers.extend([sorteo.primer_lugar, sorteo.segundo_lugar, sorteo.tercer_lugar])
            
            from collections import Counter
            number_counts = Counter(all_numbers)
            most_frequent = number_counts.most_common(10)
            least_frequent = number_counts.most_common()[-10:]
            
            # Basic statistics
            avg_first = sum(s.primer_lugar for s in sorteos) / total_sorteos
            avg_second = sum(s.segundo_lugar for s in sorteos) / total_sorteos
            avg_third = sum(s.tercer_lugar for s in sorteos) / total_sorteos
            
            result = StatisticsResponse(
                timestamp=datetime.now(),
                periodo_dias=dias,
                tipo_loteria_id=tipo_loteria_id,
                total_sorteos=total_sorteos,
                numeros_mas_frecuentes=[{'numero': num, 'frecuencia': freq} for num, freq in most_frequent],
                numeros_menos_frecuentes=[{'numero': num, 'frecuencia': freq} for num, freq in least_frequent],
                promedios={
                    'primer_lugar': round(avg_first, 2),
                    'segundo_lugar': round(avg_second, 2),
                    'tercer_lugar': round(avg_third, 2)
                }
            )
            
            # Cache result
            cache_manager.set(cache_key, result, ttl=settings.cache_ttl_statistics)
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical statistics: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/tipos-loteria", tags=["Configuración"])
async def get_tipos_loteria():
    """Obtener tipos de lotería disponibles."""
    try:
        with get_db_connection() as session:
            tipos = session.query(TipoLoteria).filter(TipoLoteria.activo == True).all()
            
            return [
                {
                    'id': tipo.id,
                    'nombre': tipo.nombre,
                    'descripcion': tipo.descripcion,
                    'hora_sorteo': tipo.hora_sorteo.strftime('%H:%M:%S') if tipo.hora_sorteo else None
                }
                for tipo in tipos
            ]
            
    except Exception as e:
        logger.error(f"Error getting lottery types: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/tipos-juego", tags=["Configuración"])
async def get_tipos_juego():
    """Obtener tipos de juego disponibles."""
    try:
        with get_db_connection() as session:
            tipos = session.query(TipoJuego).filter(TipoJuego.activo == True).all()
            
            return [
                {
                    'id': tipo.id,
                    'nombre': tipo.nombre,
                    'descripcion': tipo.descripcion,
                    'formato_numeros': tipo.formato_numeros
                }
                for tipo in tipos
            ]
            
    except Exception as e:
        logger.error(f"Error getting game types: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/generar-predicciones-faltantes", tags=["Administración"])
async def generar_predicciones_faltantes(
    request: Request,
    tipo_loteria_id: Optional[int] = Query(None, description="ID específico de tipo de lotería"),
    limite_dias: int = Query(30, description="Límite de días a procesar", ge=1, le=15365)
):
    """Generar predicciones para sorteos que no tienen predicciones, procesando desde el más antiguo."""
    try:
        logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Iniciando generación de predicciones faltantes - tipo_loteria_id: {tipo_loteria_id}, limite_dias: {limite_dias}")
        
        with get_db_connection() as session:
            # Construir query base para sorteos sin predicciones
            query = """
            SELECT DISTINCT s.fecha, s.tipo_loteria_id, tl.nombre as tipo_loteria_nombre
            FROM sorteos s
            JOIN tipos_loteria tl ON s.tipo_loteria_id = tl.id
            WHERE s.fecha NOT IN (
                SELECT DISTINCT fecha_prediccion 
                FROM predicciones_quiniela pq 
                WHERE pq.tipo_loteria_id = s.tipo_loteria_id
            )
            AND s.fecha NOT IN (
                SELECT DISTINCT fecha_prediccion 
                FROM predicciones_pale pp 
                WHERE pp.tipo_loteria_id = s.tipo_loteria_id
            )
            AND s.fecha NOT IN (
                SELECT DISTINCT fecha_prediccion 
                FROM predicciones_tripleta pt 
                WHERE pt.tipo_loteria_id = s.tipo_loteria_id
            )
            """
            
            # Agregar filtro por tipo de lotería si se especifica
            if tipo_loteria_id:
                query += f" AND s.tipo_loteria_id = {tipo_loteria_id}"
            
            # Ordenar por fecha ascendente (más antiguo primero) y limitar
            query += f" ORDER BY s.fecha ASC LIMIT {limite_dias}"
            
            # Ejecutar query
            result = session.execute(text(query)).fetchall()
            
            if not result:
                return {
                    "message": "No se encontraron sorteos sin predicciones",
                    "sorteos_procesados": 0,
                    "errores": []
                }
            
            logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Encontrados {len(result)} sorteos sin predicciones")
            
            sorteos_procesados = 0
            errores = []
            
            # Procesar cada sorteo sin predicciones
            for row in result:
                fecha_sorteo = row[0]
                tipo_loteria_sorteo = row[1]
                nombre_loteria = row[2]
                
                try:
                    logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Procesando sorteo: {fecha_sorteo} - {nombre_loteria}")
                    
                    # Verificar si ya existen predicciones (doble verificación)
                    existing_quiniela = session.query(PrediccionQuiniela).filter(
                        PrediccionQuiniela.fecha_prediccion == fecha_sorteo,
                        PrediccionQuiniela.tipo_loteria_id == tipo_loteria_sorteo
                    ).first()
                    
                    if existing_quiniela:
                        logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Saltando {fecha_sorteo} - ya tiene predicciones")
                        continue
                    
                    # Entrenar modelos para esta fecha
                    logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Entrenando modelos para {fecha_sorteo}")
                    training_results = predictor_engine.entrenar_modelos(
                        fecha=fecha_sorteo,
                        tipo_loteria_id=tipo_loteria_sorteo
                    )
                    
                    # Generar predicciones para esta fecha
                    logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Generando predicciones para {fecha_sorteo}")
                    predicciones = predictor_engine.generar_predicciones_diarias(
                        fecha=fecha_sorteo.strftime('%Y-%m-%d'),
                        tipo_loteria_id=tipo_loteria_sorteo
                    )
                    
                    # Insertar predicciones en base de datos
                    if predicciones and any(predicciones.values()):
                        logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Insertando predicciones para {fecha_sorteo}")
                        predictor_engine.insertar_predicciones_en_bd(
                            predicciones,
                            fecha_sorteo,
                            tipo_loteria_sorteo
                        )
                        sorteos_procesados += 1
                        logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Completado exitosamente: {fecha_sorteo}")
                    else:
                        error_msg = f"No se pudieron generar predicciones para {fecha_sorteo}"
                        logger.warning(f"[GENERAR-PREDICCIONES-FALTANTES] {error_msg}")
                        errores.append(error_msg)
                        
                except Exception as e:
                    error_msg = f"Error procesando sorteo {fecha_sorteo}: {str(e)}"
                    logger.error(f"[GENERAR-PREDICCIONES-FALTANTES] {error_msg}")
                    errores.append(error_msg)
                    continue
            
            # Invalidar caches relacionados
            if sorteos_procesados > 0:
                cache_manager.delete_pattern("predicciones_*")
                cache_manager.delete_pattern("estadisticas_*")
            
            logger.info(f"[GENERAR-PREDICCIONES-FALTANTES] Proceso completado. Sorteos procesados: {sorteos_procesados}, Errores: {len(errores)}")
            
            return {
                "message": f"Proceso completado. Se generaron predicciones para {sorteos_procesados} sorteos.",
                "sorteos_procesados": sorteos_procesados,
                "errores": errores[:10] if errores else [],  # Limitar errores mostrados
                "total_errores": len(errores)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GENERAR-PREDICCIONES-FALTANTES] Error general: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/entrenar-modelos", tags=["Administración"])
async def entrenar_modelos(
    request: Request,
    tipo_loteria_id: Optional[int] = Query(None, description="ID específico de tipo de lotería"),
    forzar: bool = Query(False, description="Forzar reentrenamiento")
):
    """Entrenar modelos de predicción."""
    try:
        logger.info(f"[ENTRENAR-MODELOS] Iniciando entrenamiento de modelos - tipo_loteria_id: {tipo_loteria_id}, forzar: {forzar}")
        
        # Check if training is needed
        if not forzar:
            logger.info("[ENTRENAR-MODELOS] Verificando si ya existen modelos entrenados hoy")
            # Simple check: see if we have recent cached models
            import os
            
            cache_dir = predictor_engine.model_cache_dir
            today_str = date.today().strftime('%Y%m%d')
            logger.info(f"[ENTRENAR-MODELOS] Buscando modelos en directorio: {cache_dir} con fecha: {today_str}")
            
            recent_models = [f for f in os.listdir(cache_dir) if today_str in f]
            logger.info(f"[ENTRENAR-MODELOS] Modelos encontrados: {len(recent_models)} - {recent_models[:5]}")
            
            if recent_models:
                logger.info("[ENTRENAR-MODELOS] Modelos ya entrenados hoy, retornando sin entrenar")
                return {
                    "message": "Modelos ya entrenados hoy. Use forzar=true para reentrenar",
                    "modelos_existentes": len(recent_models)
                }
        
        # Start training
        logger.info("[ENTRENAR-MODELOS] Iniciando proceso de entrenamiento en predictor_engine")
        training_results = predictor_engine.entrenar_modelos(
            fecha=date.today(),
            tipo_loteria_id=tipo_loteria_id
        )
        logger.info(f"[ENTRENAR-MODELOS] Entrenamiento completado. Resultados: {training_results}")
        
        # Invalidate prediction caches
        logger.info("[ENTRENAR-MODELOS] Invalidando caches de predicciones")
        cache_manager.delete_pattern("predicciones_*")
        cache_manager.delete_pattern("estadisticas_*")
        logger.info("[ENTRENAR-MODELOS] Caches invalidados")
        
        response = {
            "message": "Entrenamiento completado",
            "resultados": training_results,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"[ENTRENAR-MODELOS] Retornando respuesta exitosa: {response}")
        return response
        
    except Exception as e:
        logger.error(f"[ENTRENAR-MODELOS] Error en entrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")


@app.post("/generar-predicciones", tags=["Administración"])

async def generar_predicciones(
    request: Request,
    fecha: Optional[date] = Query(None, description="Fecha para predicciones (default: hoy)"),
    tipo_loteria_id: int = Query(..., description="ID del tipo de lotería")
):
    """Generar predicciones para una fecha específica."""
    try:
        fecha_target = fecha or date.today()
        fecha_str = fecha_target.strftime('%Y-%m-%d')
        
        logger.info(f"Generating predictions via API for {fecha_str}")
        
        # Generate predictions
        predicciones = predictor_engine.generar_predicciones_diarias(
            fecha_str, tipo_loteria_id
        )
        
        # Insert into database
        predictor_engine.insertar_predicciones_en_bd(
            predicciones, fecha_target, tipo_loteria_id
        )
        
        # Invalidate relevant caches
        cache_manager.delete_pattern(f"predicciones_*_{tipo_loteria_id}")
        
        return {
            "message": "Predicciones generadas exitosamente",
            "fecha": fecha_str,
            "tipo_loteria_id": tipo_loteria_id,
            "predicciones": predicciones
        }
        
    except Exception as e:
        logger.error(f"Prediction generation via API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando predicciones: {str(e)}")


@app.delete("/cache/limpiar", tags=["Administración"])
@limiter.limit("10/hour")
async def limpiar_cache(request: Request):
    """Limpiar caché del sistema."""
    try:
        # Clear all caches
        cleared_keys = cache_manager.clear_all()
        
        return {
            "message": "Caché limpiado exitosamente",
            "keys_eliminadas": cleared_keys,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Error limpiando caché")


@app.get("/sistema/estado", tags=["Administración"])
async def get_sistema_estado():
    """Obtener estado general del sistema."""
    try:
        with get_db_connection() as session:
            # Count records
            total_sorteos = session.query(Sorteo).count()
            
        
            total_predicciones = (
                session.query(PrediccionQuiniela).count() +
                session.query(PrediccionPale).count() +
                session.query(PrediccionTripleta).count()
            )
            
            # Get latest sorteo
            latest_sorteo = session.query(Sorteo).order_by(Sorteo.fecha.desc()).first()
            
            # Check model cache
            import os
            cache_files = len([f for f in os.listdir(predictor_engine.model_cache_dir) if f.endswith('.pkl')])
            
            return {
                "estado": "operativo",
                "timestamp": datetime.now().isoformat(),
                "base_datos": {
                    "total_sorteos": total_sorteos,
                    "total_predicciones": total_predicciones,
                    "ultimo_sorteo": latest_sorteo.fecha.isoformat() if latest_sorteo else None
                },
                "modelos": {
                    "modelos_disponibles": len(predictor_engine.models),
                    "archivos_cache": cache_files
                },
                "cache": {
                    "estado": "activo" if cache_manager else "inactivo"
                }
            }
            
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estado del sistema")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return ErrorResponse(
        detail="Endpoint no encontrado",
        status_code=404
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return ErrorResponse(
        detail="Error de validación en los datos enviados",
        status_code=422
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return ErrorResponse(
        detail="Error interno del servidor",
        status_code=500
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting Lottery Prediction API")
    
    try:
        # Check database connection
        from config.database import check_database_connection
        if not check_database_connection():
            logger.error("Database connection failed during startup")
            raise Exception("Database connection failed")
        
        # Initialize cache
        cache_manager.initialize()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Lottery Prediction API")
    
    try:
        # Cleanup cache connections
        cache_manager.close()
        
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Include additional route modules here if needed
# app.include_router(admin_router, prefix="/admin", tags=["Admin"])
# app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.routes:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )