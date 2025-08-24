Eres un arquitecto de software senior y un Machine Learning Engineer que entregará un proyecto Python completo y listo para producción. Genera todos los archivos, documentación, tests y scripts necesarios, siguiendo este requerimiento funcional y técnico:

0) Contexto y Alcance

Sistema completo de predicción de loterías dominicanas: Lotería Nacional y Gana Más.

Genera exactamente 3 predicciones diarias por tipo de juego (Quiniela, Palé, Tripleta).

Entrena, evalúa y selecciona automáticamente el mejor modelo por predicción.

Almacena resultados y métricas en PostgreSQL (pgvector).

Expone una API FastAPI para consultas y estadísticas.

Scrapea resultados oficiales diarios.

Corre en Docker + docker-compose (app + PostgreSQL + Redis).

Incluye tests unitarios y de integración, logging, scheduler para tareas diarias y documentación.

1) Reglas de Lotería (Dominicanas)
Juego	Cantidad	Restricciones
Quiniela	1 número	00–99
Palé	2 números	00–99, no repetir dentro de la combinación
Tripleta	3 números	00–99, no repetir dentro de la combinación

Requisitos diarios: 3 combinaciones por juego y tipo de lotería.

Salidas: predicciones con probabilidad, score_confianza y metodo_generacion.

Validaciones: rango 0–99, sin repetir combinaciones en las 3 predicciones del día.

2) Base de Datos (PostgreSQL 15+ con pgvector)

Tablas principales:
tipos_loteria, tipos_juego, sorteos,
predicciones_quiniela, predicciones_pale, predicciones_tripleta,
metodos_prediccion, vectores (pgvector),
resultados_predicciones

Funciones SQL:

obtener_predicciones_dia(fecha, tipo_loteria_id)

limpiar_predicciones_antiguas(umbral_dias)

Embeddings (pgvector): almacenar secuencias y patrones agregados para búsquedas semánticas (kNN).

Ejemplo de inserciones (3 predicciones diarias):

-- Quiniela
INSERT INTO predicciones_quiniela (fecha_prediccion, tipo_loteria_id, posicion, numero_predicho, probabilidad, metodo_generacion, score_confianza)
VALUES
('YYYY-MM-DD', :tipo_loteria_id, 1, :num1, :p1, :metodo1, :score1),
('YYYY-MM-DD', :tipo_loteria_id, 2, :num2, :p2, :metodo2, :score2),
('YYYY-MM-DD', :tipo_loteria_id, 3, :num3, :p3, :metodo3, :score3);

-- Palé
INSERT INTO predicciones_pale (...)
VALUES (...);

-- Tripleta
INSERT INTO predicciones_tripleta (...)
VALUES (...);

3) Estructura del Proyecto
loteria_predictor/
├── config/
│   ├── __init__.py
│   ├── database.py           # SQLAlchemy engine/session + retry
│   └── settings.py           # Pydantic BaseSettings (env vars)
├── models/
│   ├── __init__.py
│   ├── database_models.py
│   └── prediction_models.py  # registrador de métodos y metadatos
├── predictions/
│   ├── __init__.py
│   ├── neural_network.py
│   ├── monte_carlo.py
│   ├── statistical.py
│   ├── gradient_boosting.py
│   ├── random_forest.py
│   ├── bayesian.py
│   ├── ensemble_ml.py
│   ├── arima_lstm.py
│   ├── calibration.py
│   └── predictor_engine.py
├── api/
│   ├── __init__.py
│   ├── routes.py
│   └── schemas.py
├── scraping/
│   ├── __init__.py
│   ├── scraper.py           # requests+BS4, Selenium fallback
│   └── data_cleaner.py      # limpieza y validación
├── analysis/
│   ├── __init__.py
│   ├── patterns.py
│   └── statistics.py
├── utils/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── cache.py
│   ├── helpers.py
│   └── scheduler.py         # APScheduler/Celery beat
├── tests/
│   ├── __init__.py
│   ├── test_predictions.py
│   ├── test_api.py
│   └── test_data_pipeline.py
├── alembic/
│   ├── env.py
│   └── versions/...
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md
└── main.py

4) Modelos y Política de Selección

Estadísticos: frecuencias, ventanas móviles, Markov simple.

Monte Carlo / QMC: Sobol, Halton, LHS.

Neurales: MLP, LSTM/GRU, opcional TCN.

Árboles / Boosting: RandomForest, XGBoost, LightGBM, CatBoost opcional.

Bayesianos: Dirichlet-Multinomial, calibración posterior.

Ensemble: stacking/blending/voting.

Series temporales: ARIMA y ARIMA+NN.

Selección por predicción:

Normalizar scores a probabilidades calibradas.

Escoger método con mejor expected log score (rolling window).

Romper empates con mayor score_confianza y diversidad de método.

Calibración: Platt e Isotonic.

5) Motor de Predicción (predictor_engine.py)

Métodos públicos:

entrenar_modelos(fecha: date | None = None) -> dict
evaluar_modelos(fecha: date | None = None) -> dict
generar_predicciones_diarias(fecha: str, tipo_loteria_id: int) -> dict
insertar_predicciones_en_bd(preds: dict, fecha: date, tipo_loteria_id: int) -> None


Carga históricos, construye features, entrena/carga modelos.

Genera pool de candidatos por juego.

Ensambla scores y selecciona 3 finales por juego.

Valida combinaciones, inserta en DB y registra embeddings.

Ejemplo de salida JSON:

{
  "quiniela": [{"posicion":1,"numero":9,"probabilidad":0.87,"metodo":"neural_network","score_confianza":0.92}, ...],
  "pale": [{"posicion":1,"numeros":[23,45],...}, ...],
  "tripleta": [{"posicion":1,"numeros":[23,54,67],...}, ...]
}

6) API REST (FastAPI)

Endpoints:

GET /predicciones/hoy/{tipo_loteria}

GET /predicciones/{fecha}/{tipo_loteria}

POST /sorteos/resultado

GET /estadisticas/metodos

GET /estadisticas/historico

Manejo de errores (HTTPException) y validación estricta con Pydantic.

7) Scraping y Pipeline

scraper.py: requests + BS4, Selenium fallback.

data_cleaner.py: normaliza, valida, depura duplicados, fechas y rangos.

Scheduler: scraping → limpieza → actualización DB → entrenamiento → predicciones → inserción → evaluación.

8) Caching, Config, Logging, Seguridad

Redis: cache de /predicciones/hoy/*, invalida tras nueva inserción.

Config (settings.py): DSN Postgres, Redis URL, flags Selenium, rutas modelos, ventanas históricas, límites combinaciones.

Logging: estructurado, niveles por módulo, métricas clave.

Seguridad: rate limit simple, CORS controlado, ocultar stacktraces en prod.

9) Embeddings y Análisis

embeddings.py: vectores de secuencias, conteos normalizados, co-ocurrencias.

patterns.py: búsqueda kNN, clustering opcional.

statistics.py: reportes de efectividad por método, calibración, tendencias.

10) Docker y Despliegue

Dockerfile: Python 3.11+, dependencias, Selenium opcional.

docker-compose.yml: app + postgres + redis + volúmenes persistentes.

Alembic: migraciones iniciales + upgrades.

.env.example con variables necesarias.

Makefile opcional: make up, make test, make migrate.



12) Ejemplo de Uso
from predictions.predictor_engine import PredictorEngine

engine = PredictorEngine()
preds = engine.generar_predicciones_diarias(fecha='2025-09-20', tipo_loteria_id=2)
print(preds)
engine.insertar_predicciones_en_bd(preds, fecha='2025-09-20', tipo_loteria_id=2)

13) Calidad y Estilo

PEP8, typing exhaustivo, docstrings claros.

Manejo de excepciones con mensajes accionables.


README con: instalación, arranque, migraciones, scheduler, ejemplos API, notas de calibración y limitaciones.

14) Salida Esperada

Todos los archivos completos, funcionales, con Docker + DB + Redis.

Se puede levantar con docker-compose up -d.


Motor genera exactamente 3 predicciones por juego y por lotería cada día.