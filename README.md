# Lotería Nacional Dominicana - Sistema de Predicciones

Sistema completo de predicción de loterías dominicanas usando Machine Learning, con soporte para Lotería Nacional y Gana Más.

## 🎯 Características Principales

- **Predicciones Diarias**: 3 predicciones por tipo de juego (Quiniela, Palé, Tripleta)
- **Múltiples Modelos ML**: Redes neuronales, XGBoost, LightGBM, Monte Carlo, etc.
- **API REST**: FastAPI con documentación automática
- **Scraping Automático**: Obtención de resultados oficiales
- **Base de Datos**: PostgreSQL con pgvector para búsquedas semánticas
- **Cache Redis**: Optimización de rendimiento
- **Scheduler**: Tareas automatizadas diarias
- **Docker**: Despliegue completo con docker-compose

## 📋 Tipos de Juego

| Juego | Descripción | Números | Ejemplo |
|-------|-------------|---------|---------|
| **Quiniela** | 1 número individual | 00-99 | `23` |
| **Palé** | 2 números diferentes | 00-99 | `23-45` |
| **Tripleta** | 3 números diferentes | 00-99 | `23-45-67` |

## 🏗️ Arquitectura del Sistema

```
loteria_predictor/
├── config/                 # Configuración y base de datos
├── models/                 # Modelos SQLAlchemy y ML
├── predictions/            # Algoritmos de predicción
├── api/                    # FastAPI routes y schemas
├── scraping/              # Web scraping de resultados
├── analysis/              # Análisis y estadísticas
├── utils/                 # Utilidades (cache, embeddings, etc.)
├── tests/                 # Tests unitarios e integración
├── alembic/               # Migraciones de BD
├── docker-compose.yml     # Orquestación de servicios
├── Dockerfile             # Imagen de la aplicación
└── main.py               # Punto de entrada
```

## 🚀 Instalación y Configuración

### Prerequisitos

- Docker & Docker Compose
- Python 3.11+ (para desarrollo local)
- PostgreSQL 15+ con pgvector (incluido en Docker)

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd loteria_predictor
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

### 3. Levantar con Docker

```bash
# Servicios básicos (app + DB + Redis)
docker-compose up -d

# Con administración (incluye pgAdmin)
docker-compose --profile admin up -d

# Con monitoreo (incluye Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Producción completa
docker-compose --profile production --profile monitoring up -d
```

### 4. Verificar Instalación

```bash
# Verificar que los servicios estén corriendo
docker-compose ps

# Verificar salud de la API
curl http://localhost:8000/health

# Ver logs
docker-compose logs -f loteria_app
```

## 🔧 Configuración

### Variables de Entorno Principales

```bash
# Base de Datos
DATABASE_URL=postgresql://loteria_user:LoteriaPass2024!@localhost:5433/loteria_db

# Redis Cache
REDIS_URL=redis://localhost:6380/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Machine Learning
MODEL_CACHE_DAYS=7
HISTORICAL_WINDOW_DAYS=365
MIN_TRAINING_SAMPLES=50

# Scraping
SELENIUM_ENABLED=true
SELENIUM_HEADLESS=true

# Scheduler
ENABLE_SCHEDULER=true
SCRAPING_SCHEDULE=0 22 * * *    # 10 PM diario
PREDICTION_SCHEDULE=0 8 * * *   # 8 AM diario
```

## 📊 API Endpoints

### Predicciones

```bash
# Predicciones de hoy
GET /predicciones/hoy/{tipo_loteria}

# Predicciones por fecha
GET /predicciones/{fecha}/{tipo_loteria}

# Ejemplo
curl "http://localhost:8000/predicciones/hoy/1"
```

### Resultados de Sorteos

```bash
# Registrar resultado
POST /sorteos/resultado

# Ejemplo
curl -X POST "http://localhost:8000/sorteos/resultado" \
  -H "Content-Type: application/json" \
  -d '{
    "fecha": "2024-01-15",
    "tipo_loteria_id": 1,
    "primer_lugar": 23,
    "segundo_lugar": 45,
    "tercer_lugar": 67
  }'
```

### Estadísticas

```bash
# Rendimiento de métodos
GET /estadisticas/metodos

# Estadísticas históricas
GET /estadisticas/historico?dias=30&tipo_loteria_id=1
```

### Administración

```bash
# Entrenar modelos
POST /entrenar-modelos?tipo_loteria_id=1

# Generar predicciones
POST /generar-predicciones?fecha=2024-01-15&tipo_loteria_id=1

# Limpiar cache
DELETE /cache/limpiar
```

## 🤖 Modelos de Machine Learning

### Modelos Implementados

1. **Redes Neuronales**
   - LSTM para secuencias temporales
   - MLP para patrones complejos

2. **Gradient Boosting**
   - XGBoost
   - LightGBM
   - CatBoost

3. **Estadísticos**
   - Análisis de frecuencias
   - Promedios móviles
   - Bayesiano con Dirichlet-Multinomial

4. **Monte Carlo**
   - Simulación clásica
   - Quasi-Monte Carlo (Sobol, Halton)
   - MCMC con cadenas de Markov

5. **Ensemble**
   - Stacking
   - Voting
   - Blending

### Selección de Modelos

El sistema evalúa automáticamente el rendimiento de cada modelo y selecciona las 3 mejores predicciones basándose en:

- **Expected Log Score**: Probabilidad calibrada
- **Score de Confianza**: Certeza del modelo
- **Diversidad**: Evita predicciones repetidas

## 📈 Ejemplo de Uso Programático

```python
from predictions.predictor_engine import PredictorEngine
from datetime import date

# Inicializar motor
engine = PredictorEngine()

# Entrenar modelos
training_results = engine.entrenar_modelos(
    fecha=date.today(),
    tipo_loteria_id=1
)

# Generar predicciones
predicciones = engine.generar_predicciones_diarias(
    fecha='2024-01-15',
    tipo_loteria_id=1
)

# Insertar en base de datos
engine.insertar_predicciones_en_bd(
    predicciones,
    date(2024, 1, 15),
    tipo_loteria_id=1
)

print(predicciones)
# Output:
# {
#   "quiniela": [
#     {"posicion": 1, "numero": 23, "probabilidad": 0.85, ...}
#   ],
#   "pale": [...],
#   "tripleta": [...]
# }
```

## 🕐 Scheduler Automático

El sistema incluye un scheduler que ejecuta automáticamente:

- **22:00 diario**: Scraping de resultados oficiales
- **08:00 diario**: Entrenamiento de modelos y generación de predicciones
- **02:00 domingo**: Limpieza de datos antiguos

## 🗄️ Estructura de Base de Datos

### Tablas Principales

- `tipos_loteria`: Gana Más, Lotería Nacional
- `tipos_juego`: quiniela, pale, tripleta
- `sorteos`: resultados históricos
- `predicciones_*`: predicciones por tipo de juego
- `vectores`: embeddings para búsquedas semánticas (pgvector)
- `resultados_predicciones`: evaluación de aciertos

### Funciones SQL Útiles

```sql
-- Obtener predicciones del día
SELECT * FROM obtener_predicciones_dia('2024-01-15', 1);

-- Limpiar predicciones antiguas
SELECT limpiar_predicciones_antiguas(90);
```

## 🧪 Testing

```bash
# Tests unitarios
pytest tests/test_predictions.py

# Tests de API
pytest tests/test_api.py

# Tests de integración
pytest tests/test_data_pipeline.py

# Cobertura
pytest --cov=. --cov-report=html
```

## 📊 Monitoreo

### Métricas Disponibles

- Accuracy por modelo y tipo de juego
- Latencia de predicciones
- Uso de cache
- Errores de scraping
- Rendimiento de base de datos

### Dashboards

Con el perfil `monitoring`:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/LoteriaGrafana2024!)

## 🔒 Seguridad

- Rate limiting en endpoints públicos
- Validación estricta con Pydantic
- Sanitización de inputs de scraping
- Logs estructurados para auditoría
- CORS configurado apropiadamente

## 📝 Logs

Los logs están estructurados en formato JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "level": "INFO",
  "logger": "predictions.neural_network",
  "message": "Model training completed",
  "accuracy": 0.85,
  "samples": 1500
}
```

## 🐛 Troubleshooting

### Problemas Comunes

1. **Error de conexión a BD**
   ```bash
   docker-compose logs postgres_loteria
   # Verificar que el contenedor esté saludable
   ```

2. **Modelos no entrenan**
   ```bash
   # Verificar datos históricos
   docker-compose exec loteria_app python -c "
   from config.database import get_db_connection
   from models.database_models import Sorteo
   with get_db_connection() as session:
       count = session.query(Sorteo).count()
       print(f'Sorteos en BD: {count}')
   "
   ```

3. **Scraping falla**
   ```bash
   # Test de scraping
   curl -X POST "http://localhost:8000/admin/test-scraping"
   ```

### Comandos Útiles

```bash
# Reiniciar servicios
docker-compose restart

# Ver logs en tiempo real
docker-compose logs -f loteria_app

# Acceder al contenedor
docker-compose exec loteria_app bash

# Backup de BD
docker-compose exec postgres_loteria pg_dump -U loteria_user loteria_db > backup.sql

# Limpiar todo y reiniciar
docker-compose down -v
docker-compose up -d
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Lotería Nacional Dominicana por los datos públicos
- Comunidad de ML/AI por las librerías utilizadas
- PostgreSQL y pgvector por las capacidades de búsqueda vectorial

## 📞 Soporte

Para soporte técnico o consultas:
- Crear un issue en GitHub
- Revisar la documentación de la API en `/docs`
- Consultar logs del sistema

---

**⚠️ Disclaimer**: Este sistema es para fines educativos y de investigación. Las predicciones no garantizan resultados en juegos de azar reales.