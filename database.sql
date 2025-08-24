-- ========================================
-- ESQUEMA DE BASE DE DATOS PARA SISTEMA DE PREDICCIONES
-- LOTERÍA NACIONAL DOMINICANA
-- PostgreSQL 15+ con extensión pgvector
-- ========================================

-- Activar extensión pgvector (comentado temporalmente)
-- CREATE EXTENSION IF NOT EXISTS vector;
-- Descomenta la línea anterior una vez que tengas pgvector instalado

-- ========================================
-- TABLA: tipos_loteria
-- Define los tipos de sorteos disponibles
-- ========================================
CREATE TABLE tipos_loteria (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL UNIQUE,
    descripcion TEXT,
    hora_sorteo TIME NOT NULL,
    activo BOOLEAN DEFAULT true,
    creado_en TIMESTAMP DEFAULT NOW()
);

-- Insertar tipos de lotería simplificados
INSERT INTO tipos_loteria (nombre, descripcion, hora_sorteo) VALUES
('Gana Más', 'Sorteo Gana Más 2:30 PM (Lunes a Domingo)', '14:30:00'),
('Lotería Nacional', 'Sorteo Lotería Nacional 9:00 PM (Lunes a Sábado) y 6:00 PM (Domingo)', '21:00:00');

-- ========================================
-- TABLA: sorteos
-- Almacena los resultados históricos de los sorteos
-- ========================================
CREATE TABLE sorteos (
    id SERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    primer_lugar INTEGER NOT NULL CHECK (primer_lugar >= 0 AND primer_lugar <= 99),
    segundo_lugar INTEGER NOT NULL CHECK (segundo_lugar >= 0 AND segundo_lugar <= 99),
    tercer_lugar INTEGER NOT NULL CHECK (tercer_lugar >= 0 AND tercer_lugar <= 99),
    fuente_scraping VARCHAR(255),
    creado_en TIMESTAMP DEFAULT NOW(),
    
    -- Constraint único por fecha y tipo de lotería
    CONSTRAINT unique_sorteo_fecha_tipo UNIQUE (fecha, tipo_loteria_id)
);

-- ========================================
-- TABLA: predicciones
-- Almacena las predicciones generadas por diferentes métodos
-- ========================================
CREATE TABLE predicciones (
    id SERIAL PRIMARY KEY,
    fecha_prediccion DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    metodo VARCHAR(100) NOT NULL,
    numeros_recomendados JSONB NOT NULL,
    score FLOAT CHECK (score >= 0 AND score <= 1),
    creado_en TIMESTAMP DEFAULT NOW(),
    
    -- Constraint único por fecha, tipo y método
    CONSTRAINT unique_prediccion_fecha_tipo_metodo UNIQUE (fecha_prediccion, tipo_loteria_id, metodo)
);

-- ========================================
-- TABLA: vectores
-- Almacena los embeddings vectoriales para búsquedas de similitud
-- ========================================
CREATE TABLE vectores (
    id SERIAL PRIMARY KEY,
    sorteo_id INTEGER NOT NULL REFERENCES sorteos(id) ON DELETE CASCADE,
    -- Temporalmente usar array de FLOAT hasta instalar pgvector
    embedding FLOAT[] NOT NULL,
    -- Una vez instalado pgvector, cambiar a: embedding VECTOR(128) NOT NULL,
    creado_en TIMESTAMP DEFAULT NOW()
);

-- ========================================
-- ÍNDICES PARA OPTIMIZACIÓN
-- ========================================

-- Índices para tabla sorteos
CREATE INDEX idx_sorteos_fecha ON sorteos(fecha DESC);
CREATE INDEX idx_sorteos_tipo_loteria ON sorteos(tipo_loteria_id);
CREATE INDEX idx_sorteos_fecha_tipo ON sorteos(fecha DESC, tipo_loteria_id);
CREATE INDEX idx_sorteos_creado_en ON sorteos(creado_en DESC);
CREATE INDEX idx_sorteos_primer_lugar ON sorteos(primer_lugar);
CREATE INDEX idx_sorteos_segundo_lugar ON sorteos(segundo_lugar);
CREATE INDEX idx_sorteos_tercer_lugar ON sorteos(tercer_lugar);

-- Índice compuesto para búsquedas de combinaciones por tipo
CREATE INDEX idx_sorteos_combinacion_tipo ON sorteos(tipo_loteria_id, primer_lugar, segundo_lugar, tercer_lugar);

-- Índices para tabla predicciones
CREATE INDEX idx_predicciones_fecha ON predicciones(fecha_prediccion DESC);
CREATE INDEX idx_predicciones_tipo_loteria ON predicciones(tipo_loteria_id);
CREATE INDEX idx_predicciones_fecha_tipo ON predicciones(fecha_prediccion DESC, tipo_loteria_id);
CREATE INDEX idx_predicciones_metodo ON predicciones(metodo);
CREATE INDEX idx_predicciones_score ON predicciones(score DESC);
CREATE INDEX idx_predicciones_creado_en ON predicciones(creado_en DESC);

-- Índice compuesto para consultas por método, tipo y fecha
CREATE INDEX idx_predicciones_metodo_tipo_fecha ON predicciones(metodo, tipo_loteria_id, fecha_prediccion DESC);

-- Índices para tabla vectores
CREATE INDEX idx_vectores_sorteo_id ON vectores(sorteo_id);
CREATE INDEX idx_vectores_creado_en ON vectores(creado_en DESC);

-- Índices vectoriales (comentados hasta instalar pgvector)
-- Una vez instalado pgvector, descomenta estas líneas:
-- CREATE INDEX idx_vectores_embedding_hnsw ON vectores 
-- USING hnsw (embedding vector_cosine_ops) 
-- WITH (m = 16, ef_construction = 64);

-- Índice temporal para arrays
CREATE INDEX idx_vectores_embedding_gin ON vectores USING gin(embedding);

-- IVFFlat: Alternativa para datasets muy grandes (comentado por defecto)
-- CREATE INDEX idx_vectores_embedding_ivfflat ON vectores 
-- USING ivfflat (embedding vector_cosine_ops) 
-- WITH (lists = 100);

-- ========================================
-- VISTAS ÚTILES PARA ANÁLISIS
-- ========================================

-- Vista para análisis de tendencias recientes por tipo
CREATE OR REPLACE VIEW sorteos_recientes AS
SELECT 
    s.*,
    tl.nombre as tipo_loteria_nombre,
    tl.hora_sorteo,
    EXTRACT(DOW FROM s.fecha) as dia_semana,
    EXTRACT(MONTH FROM s.fecha) as mes,
    EXTRACT(YEAR FROM s.fecha) as año
FROM sorteos s
JOIN tipos_loteria tl ON s.tipo_loteria_id = tl.id
ORDER BY s.fecha DESC, tl.hora_sorteo DESC;

-- Vista para estadísticas de números más frecuentes por tipo
CREATE OR REPLACE VIEW estadisticas_numeros AS
WITH numeros_expandidos AS (
    SELECT 
        s.primer_lugar as numero, 
        s.fecha, 
        s.tipo_loteria_id,
        tl.nombre as tipo_loteria_nombre
    FROM sorteos s
    JOIN tipos_loteria tl ON s.tipo_loteria_id = tl.id
    UNION ALL
    SELECT 
        s.segundo_lugar as numero, 
        s.fecha, 
        s.tipo_loteria_id,
        tl.nombre as tipo_loteria_nombre
    FROM sorteos s
    JOIN tipos_loteria tl ON s.tipo_loteria_id = tl.id
    UNION ALL
    SELECT 
        s.tercer_lugar as numero, 
        s.fecha, 
        s.tipo_loteria_id,
        tl.nombre as tipo_loteria_nombre
    FROM sorteos s
    JOIN tipos_loteria tl ON s.tipo_loteria_id = tl.id
)
SELECT 
    numero,
    tipo_loteria_id,
    tipo_loteria_nombre,
    COUNT(*) as frecuencia,
    ROUND(COUNT(*) * 100.0 / (
        SELECT COUNT(*) * 3 
        FROM sorteos s2 
        WHERE s2.tipo_loteria_id = numeros_expandidos.tipo_loteria_id
    ), 2) as porcentaje,
    MAX(fecha) as ultima_aparicion,
    MIN(fecha) as primera_aparicion
FROM numeros_expandidos
GROUP BY numero, tipo_loteria_id, tipo_loteria_nombre
ORDER BY tipo_loteria_id, frecuencia DESC;

-- Vista para análisis de predicciones por método y tipo
CREATE OR REPLACE VIEW resumen_predicciones AS
SELECT 
    p.metodo,
    p.tipo_loteria_id,
    tl.nombre as tipo_loteria_nombre,
    COUNT(*) as total_predicciones,
    AVG(p.score) as score_promedio,
    MAX(p.score) as mejor_score,
    MIN(p.score) as peor_score,
    MAX(p.creado_en) as ultima_prediccion
FROM predicciones p
JOIN tipos_loteria tl ON p.tipo_loteria_id = tl.id
GROUP BY p.metodo, p.tipo_loteria_id, tl.nombre
ORDER BY p.tipo_loteria_id, score_promedio DESC;

-- ========================================
-- FUNCIONES ÚTILES
-- ========================================

-- Función para obtener sorteos similares (versión temporal sin pgvector)
-- Una vez instalado pgvector, reemplazar con la versión vectorial
CREATE OR REPLACE FUNCTION buscar_sorteos_similares_temporal(
    numeros INTEGER[],  -- [primer_lugar, segundo_lugar, tercer_lugar]
    limite INTEGER DEFAULT 10
)
RETURNS TABLE(
    sorteo_id INTEGER,
    fecha DATE,
    primer_lugar INTEGER,
    segundo_lugar INTEGER,
    tercer_lugar INTEGER,
    similitud FLOAT
) AS $
BEGIN
    RETURN QUERY
    SELECT 
        s.id,
        s.fecha,
        s.primer_lugar,
        s.segundo_lugar,
        s.tercer_lugar,
        -- Similitud básica por coincidencias exactas
        CASE 
            WHEN s.primer_lugar = numeros[1] AND s.segundo_lugar = numeros[2] AND s.tercer_lugar = numeros[3] THEN 1.0
            WHEN (s.primer_lugar = numeros[1] AND s.segundo_lugar = numeros[2]) OR
                 (s.primer_lugar = numeros[1] AND s.tercer_lugar = numeros[3]) OR
                 (s.segundo_lugar = numeros[2] AND s.tercer_lugar = numeros[3]) THEN 0.67
            WHEN s.primer_lugar = numeros[1] OR s.segundo_lugar = numeros[2] OR s.tercer_lugar = numeros[3] THEN 0.33
            ELSE 0.0
        END as similitud
    FROM sorteos s
    WHERE s.primer_lugar = ANY(numeros) OR s.segundo_lugar = ANY(numeros) OR s.tercer_lugar = ANY(numeros)
    ORDER BY similitud DESC, s.fecha DESC
    LIMIT limite;
END;
$ LANGUAGE plpgsql;

-- Función para obtener estadísticas de un número específico
CREATE OR REPLACE FUNCTION estadisticas_numero(numero_consulta INTEGER)
RETURNS TABLE(
    posicion VARCHAR(20),
    frecuencia BIGINT,
    porcentaje NUMERIC,
    ultima_aparicion DATE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'Primer lugar'::VARCHAR(20),
        COUNT(*),
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sorteos), 2),
        MAX(fecha)
    FROM sorteos 
    WHERE primer_lugar = numero_consulta
    
    UNION ALL
    
    SELECT 
        'Segundo lugar'::VARCHAR(20),
        COUNT(*),
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sorteos), 2),
        MAX(fecha)
    FROM sorteos 
    WHERE segundo_lugar = numero_consulta
    
    UNION ALL
    
    SELECT 
        'Tercer lugar'::VARCHAR(20),
        COUNT(*),
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sorteos), 2),
        MAX(fecha)
    FROM sorteos 
    WHERE tercer_lugar = numero_consulta;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- DATOS DE EJEMPLO (OPCIONAL)
-- ========================================

-- Insertar algunos sorteos de ejemplo por tipo
-- INSERT INTO sorteos (fecha, tipo_loteria_id, primer_lugar, segundo_lugar, tercer_lugar, fuente_scraping) VALUES
-- ('2024-01-01', 1, 15, 37, 82, 'scraping_web'),  -- Lotería 2:30 PM
-- ('2024-01-01', 2, 03, 56, 91, 'scraping_web'),  -- Lotería 9:00 PM
-- ('2024-01-02', 1, 42, 18, 75, 'scraping_web'),  -- Lotería 2:30 PM
-- ('2024-01-02', 3, 67, 29, 44, 'scraping_web');  -- Gana Más

-- Insertar predicción de ejemplo
-- INSERT INTO predicciones (fecha_prediccion, tipo_loteria_id, metodo, numeros_recomendados, score) VALUES
-- ('2024-01-03', 1, 'red_neuronal', 
--  '{"primer_lugar": [{"numero": 25, "probabilidad": 0.15}, {"numero": 67, "probabilidad": 0.12}], 
--    "segundo_lugar": [{"numero": 43, "probabilidad": 0.18}, {"numero": 89, "probabilidad": 0.14}], 
--    "tercer_lugar": [{"numero": 12, "probabilidad": 0.16}, {"numero": 58, "probabilidad": 0.13}]}', 
--  0.85);

-- ========================================
-- COMENTARIOS ADICIONALES
-- ========================================

/*
NOTAS DE USO:

1. CONFIGURACIÓN PGVECTOR:
   - El vector tiene dimensión 128, ajusta según tus embeddings
   - HNSW es generalmente mejor para la mayoría de casos de uso
   - Para datasets > 1M vectores, considera IVFFlat

2. ESTRUCTURA JSONB EN PREDICCIONES:
   Formato sugerido para numeros_recomendados:
   {
     "primer_lugar": [{"numero": 25, "probabilidad": 0.15}, ...],
     "segundo_lugar": [{"numero": 43, "probabilidad": 0.18}, ...],
     "tercer_lugar": [{"numero": 12, "probabilidad": 0.16}, ...]
   }

3. OPTIMIZACIONES:
   - Los índices están optimizados para consultas temporales DESC
   - Considera particionamiento por año si el dataset crece mucho
   - Monitorea el rendimiento de las consultas vectoriales

4. MANTENIMIENTO:
   - Ejecuta VACUUM y ANALYZE regularmente
   - Para índices HNSW, ajusta ef_search según precisión vs velocidad
   - Considera cleanup de predicciones antiguas según tu política de retención

5. CONEXIÓN PYTHON:
   - Usa psycopg2 o asyncpg
   - Para vectores: numpy.array -> pgvector.Vector
   - Ejemplo: Vector(embedding_array.tolist())
*/