-- ========================================
-- ESQUEMA DE BASE DE DATOS MEJORADO PARA SISTEMA DE PREDICCIONES
-- LOTERÍA NACIONAL DOMINICANA
-- PostgreSQL 15+ con extensión pgvector
-- Versión optimizada para predicciones diarias
-- ========================================

-- Activar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

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

-- Insertar tipos de lotería
INSERT INTO tipos_loteria (nombre, descripcion, hora_sorteo) VALUES
('Gana Más', 'Sorteo Gana Más 2:30 PM (Lunes a Domingo)', '14:30:00'),
('Lotería Nacional', 'Sorteo Lotería Nacional 9:00 PM (Lunes a Sábado) y 6:00 PM (Domingo)', '21:00:00');

-- ========================================
-- TABLA: tipos_juego
-- Define los tipos de juegos de predicción
-- ========================================
CREATE TABLE tipos_juego (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL UNIQUE,
    descripcion TEXT,
    formato_numeros VARCHAR(100) NOT NULL, -- Ej: "XX" para quiniela, "XX-XX" para pale, "XX-XX-XX" para tripleta
    activo BOOLEAN DEFAULT true,
    creado_en TIMESTAMP DEFAULT NOW()
);

-- Insertar tipos de juego
INSERT INTO tipos_juego (nombre, descripcion, formato_numeros) VALUES
('quiniela', 'Predicción de número individual (00-99)', 'XX'),
('pale', 'Predicción de combinación de 2 números', 'XX-XX'),
('tripleta', 'Predicción de combinación de 3 números', 'XX-XX-XX');

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
-- TABLA: predicciones_quiniela
-- Almacena las 3 mejores predicciones de quiniela por día
-- ========================================
CREATE TABLE predicciones_quiniela (
    id SERIAL PRIMARY KEY,
    fecha_prediccion DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    posicion INTEGER NOT NULL CHECK (posicion IN (1, 2, 3)), -- P1, P2, P3
    numero_predicho INTEGER NOT NULL CHECK (numero_predicho >= 0 AND numero_predicho <= 99),
    probabilidad FLOAT CHECK (probabilidad >= 0 AND probabilidad <= 1),
    metodo_generacion VARCHAR(100) NOT NULL, -- 'neural_network', 'monte_carlo', 'statistical'
    score_confianza FLOAT CHECK (score_confianza >= 0 AND score_confianza <= 1),
    creado_en TIMESTAMP DEFAULT NOW(),
    
    -- Constraint único: solo una predicción por posición, fecha y tipo
    CONSTRAINT unique_quiniela_fecha_tipo_posicion UNIQUE (fecha_prediccion, tipo_loteria_id, posicion)
);

-- ========================================
-- TABLA: predicciones_pale
-- Almacena las 3 mejores predicciones de pale por día
-- ========================================
CREATE TABLE predicciones_pale (
    id SERIAL PRIMARY KEY,
    fecha_prediccion DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    posicion INTEGER NOT NULL CHECK (posicion IN (1, 2, 3)), -- P1, P2, P3
    numero_1 INTEGER NOT NULL CHECK (numero_1 >= 0 AND numero_1 <= 99),
    numero_2 INTEGER NOT NULL CHECK (numero_2 >= 0 AND numero_2 <= 99),
    probabilidad FLOAT CHECK (probabilidad >= 0 AND probabilidad <= 1),
    metodo_generacion VARCHAR(100) NOT NULL,
    score_confianza FLOAT CHECK (score_confianza >= 0 AND score_confianza <= 1),
    creado_en TIMESTAMP DEFAULT NOW(),
    
    -- Constraint único: solo una predicción por posición, fecha y tipo
    CONSTRAINT unique_pale_fecha_tipo_posicion UNIQUE (fecha_prediccion, tipo_loteria_id, posicion),
    
    -- Constraint: los números deben ser diferentes
    CONSTRAINT pale_numeros_diferentes CHECK (numero_1 != numero_2)
);

-- ========================================
-- TABLA: predicciones_tripleta
-- Almacena las 3 mejores predicciones de tripleta por día
-- ========================================
CREATE TABLE predicciones_tripleta (
    id SERIAL PRIMARY KEY,
    fecha_prediccion DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    posicion INTEGER NOT NULL CHECK (posicion IN (1, 2, 3)), -- P1, P2, P3
    numero_1 INTEGER NOT NULL CHECK (numero_1 >= 0 AND numero_1 <= 99),
    numero_2 INTEGER NOT NULL CHECK (numero_2 >= 0 AND numero_2 <= 99),
    numero_3 INTEGER NOT NULL CHECK (numero_3 >= 0 AND numero_3 <= 99),
    probabilidad FLOAT CHECK (probabilidad >= 0 AND probabilidad <= 1),
    metodo_generacion VARCHAR(100) NOT NULL,
    score_confianza FLOAT CHECK (score_confianza >= 0 AND score_confianza <= 1),
    creado_en TIMESTAMP DEFAULT NOW(),
    
    -- Constraint único: solo una predicción por posición, fecha y tipo
    CONSTRAINT unique_tripleta_fecha_tipo_posicion UNIQUE (fecha_prediccion, tipo_loteria_id, posicion),
    
    -- Constraint: los tres números deben ser diferentes
    CONSTRAINT tripleta_numeros_diferentes CHECK (numero_1 != numero_2 AND numero_1 != numero_3 AND numero_2 != numero_3)
);

-- ========================================
-- TABLA: metodos_prediccion
-- Almacena información sobre los métodos de predicción utilizados
-- ========================================
CREATE TABLE metodos_prediccion (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL UNIQUE,
    descripcion TEXT,
    version VARCHAR(50),
    parametros JSONB, -- Parámetros específicos del método
    activo BOOLEAN DEFAULT true,
    creado_en TIMESTAMP DEFAULT NOW()
);

-- Insertar métodos de predicción
INSERT INTO metodos_prediccion (nombre, descripcion, version) VALUES
('neural_network', 'Red neuronal profunda para predicciones', '1.0'),
('monte_carlo', 'Simulación Monte Carlo', '1.0'),
('statistical', 'Análisis estadístico avanzado', '1.0');

-- ========================================
-- TABLA: vectores
-- Almacena los embeddings vectoriales para búsquedas de similitud
-- ========================================
CREATE TABLE vectores (
    id SERIAL PRIMARY KEY,
    sorteo_id INTEGER NOT NULL REFERENCES sorteos(id) ON DELETE CASCADE,
    embedding VECTOR(128) NOT NULL, -- Vector de 128 dimensiones
    creado_en TIMESTAMP DEFAULT NOW()
);

-- ========================================
-- TABLA: resultados_predicciones
-- Almacena los resultados de las predicciones vs sorteos reales
-- ========================================
CREATE TABLE resultados_predicciones (
    id SERIAL PRIMARY KEY,
    fecha_sorteo DATE NOT NULL,
    tipo_loteria_id INTEGER NOT NULL REFERENCES tipos_loteria(id),
    tipo_juego_id INTEGER NOT NULL REFERENCES tipos_juego(id),
    prediccion_id INTEGER, -- ID de la predicción específica
    acierto BOOLEAN NOT NULL,
    tipo_acierto VARCHAR(50), -- 'exacto', 'parcial', 'ninguno'
    puntos_obtenidos INTEGER DEFAULT 0,
    creado_en TIMESTAMP DEFAULT NOW()
);

-- ========================================
-- ÍNDICES PARA OPTIMIZACIÓN
-- ========================================

-- Índices para tabla sorteos
CREATE INDEX idx_sorteos_fecha ON sorteos(fecha DESC);
CREATE INDEX idx_sorteos_tipo_loteria ON sorteos(tipo_loteria_id);
CREATE INDEX idx_sorteos_fecha_tipo ON sorteos(fecha DESC, tipo_loteria_id);
CREATE INDEX idx_sorteos_combinacion_tipo ON sorteos(tipo_loteria_id, primer_lugar, segundo_lugar, tercer_lugar);

-- Índices para predicciones_quiniela
CREATE INDEX idx_quiniela_fecha ON predicciones_quiniela(fecha_prediccion DESC);
CREATE INDEX idx_quiniela_tipo_loteria ON predicciones_quiniela(tipo_loteria_id);
CREATE INDEX idx_quiniela_fecha_tipo ON predicciones_quiniela(fecha_prediccion DESC, tipo_loteria_id);
CREATE INDEX idx_quiniela_posicion ON predicciones_quiniela(posicion);
CREATE INDEX idx_quiniela_numero ON predicciones_quiniela(numero_predicho);
CREATE INDEX idx_quiniela_metodo ON predicciones_quiniela(metodo_generacion);

-- Índices para predicciones_pale
CREATE INDEX idx_pale_fecha ON predicciones_pale(fecha_prediccion DESC);
CREATE INDEX idx_pale_tipo_loteria ON predicciones_pale(tipo_loteria_id);
CREATE INDEX idx_pale_fecha_tipo ON predicciones_pale(fecha_prediccion DESC, tipo_loteria_id);
CREATE INDEX idx_pale_posicion ON predicciones_pale(posicion);
CREATE INDEX idx_pale_numeros ON predicciones_pale(numero_1, numero_2);
CREATE INDEX idx_pale_metodo ON predicciones_pale(metodo_generacion);

-- Índices para predicciones_tripleta
CREATE INDEX idx_tripleta_fecha ON predicciones_tripleta(fecha_prediccion DESC);
CREATE INDEX idx_tripleta_tipo_loteria ON predicciones_tripleta(tipo_loteria_id);
CREATE INDEX idx_tripleta_fecha_tipo ON predicciones_tripleta(fecha_prediccion DESC, tipo_loteria_id);
CREATE INDEX idx_tripleta_posicion ON predicciones_tripleta(posicion);
CREATE INDEX idx_tripleta_numeros ON predicciones_tripleta(numero_1, numero_2, numero_3);
CREATE INDEX idx_tripleta_metodo ON predicciones_tripleta(metodo_generacion);

-- Índices para vectores
CREATE INDEX idx_vectores_sorteo_id ON vectores(sorteo_id);
CREATE INDEX idx_vectores_embedding_hnsw ON vectores 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Índices para resultados
CREATE INDEX idx_resultados_fecha ON resultados_predicciones(fecha_sorteo DESC);
CREATE INDEX idx_resultados_tipo_loteria ON resultados_predicciones(tipo_loteria_id);
CREATE INDEX idx_resultados_tipo_juego ON resultados_predicciones(tipo_juego_id);
CREATE INDEX idx_resultados_acierto ON resultados_predicciones(acierto);

-- ========================================
-- VISTAS ÚTILES PARA ANÁLISIS
-- ========================================

-- Vista para predicciones del día actual
CREATE OR REPLACE VIEW predicciones_hoy AS
SELECT 
    'quiniela' as tipo_juego,
    pq.fecha_prediccion,
    tl.nombre as tipo_loteria,
    pq.posicion,
    pq.numero_predicho::text as combinacion,
    pq.probabilidad,
    pq.metodo_generacion,
    pq.score_confianza
FROM predicciones_quiniela pq
JOIN tipos_loteria tl ON pq.tipo_loteria_id = tl.id
WHERE pq.fecha_prediccion = CURRENT_DATE

UNION ALL

SELECT 
    'pale' as tipo_juego,
    pp.fecha_prediccion,
    tl.nombre as tipo_loteria,
    pp.posicion,
    CONCAT(LPAD(pp.numero_1::text, 2, '0'), '-', LPAD(pp.numero_2::text, 2, '0')) as combinacion,
    pp.probabilidad,
    pp.metodo_generacion,
    pp.score_confianza
FROM predicciones_pale pp
JOIN tipos_loteria tl ON pp.tipo_loteria_id = tl.id
WHERE pp.fecha_prediccion = CURRENT_DATE

UNION ALL

SELECT 
    'tripleta' as tipo_juego,
    pt.fecha_prediccion,
    tl.nombre as tipo_loteria,
    pt.posicion,
    CONCAT(LPAD(pt.numero_1::text, 2, '0'), '-', LPAD(pt.numero_2::text, 2, '0'), '-', LPAD(pt.numero_3::text, 2, '0')) as combinacion,
    pt.probabilidad,
    pt.metodo_generacion,
    pt.score_confianza
FROM predicciones_tripleta pt
JOIN tipos_loteria tl ON pt.tipo_loteria_id = tl.id
WHERE pt.fecha_prediccion = CURRENT_DATE

ORDER BY tipo_loteria, tipo_juego, posicion;

-- Vista para estadísticas de efectividad por método
CREATE OR REPLACE VIEW estadisticas_metodos AS
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
    ROUND(AVG(CASE WHEN rp.acierto THEN 1.0 ELSE 0.0 END) * 100, 2) as porcentaje_acierto,
    SUM(rp.puntos_obtenidos) as puntos_totales
FROM resultados_predicciones rp
JOIN tipos_juego tj ON rp.tipo_juego_id = tj.id
LEFT JOIN predicciones_quiniela pq ON rp.tipo_juego_id = 1 AND rp.prediccion_id = pq.id
LEFT JOIN predicciones_pale pp ON rp.tipo_juego_id = 2 AND rp.prediccion_id = pp.id
LEFT JOIN predicciones_tripleta pt ON rp.tipo_juego_id = 3 AND rp.prediccion_id = pt.id
GROUP BY rp.tipo_juego_id, tj.nombre, metodo
ORDER BY rp.tipo_juego_id, porcentaje_acierto DESC;

-- ========================================
-- FUNCIONES ÚTILES
-- ========================================

-- Función para obtener las mejores predicciones del día
CREATE OR REPLACE FUNCTION obtener_predicciones_dia(
    fecha_consulta DATE DEFAULT CURRENT_DATE,
    tipo_loteria_consulta INTEGER DEFAULT NULL
)
RETURNS TABLE(
    tipo_juego VARCHAR(20),
    tipo_loteria VARCHAR(100),
    posicion INTEGER,
    combinacion TEXT,
    probabilidad FLOAT,
    metodo VARCHAR(100),
    confianza FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'quiniela'::VARCHAR(20),
        tl.nombre,
        pq.posicion,
        LPAD(pq.numero_predicho::text, 2, '0'),
        pq.probabilidad,
        pq.metodo_generacion,
        pq.score_confianza
    FROM predicciones_quiniela pq
    JOIN tipos_loteria tl ON pq.tipo_loteria_id = tl.id
    WHERE pq.fecha_prediccion = fecha_consulta
    AND (tipo_loteria_consulta IS NULL OR pq.tipo_loteria_id = tipo_loteria_consulta)
    
    UNION ALL
    
    SELECT 
        'pale'::VARCHAR(20),
        tl.nombre,
        pp.posicion,
        CONCAT(LPAD(pp.numero_1::text, 2, '0'), '-', LPAD(pp.numero_2::text, 2, '0')),
        pp.probabilidad,
        pp.metodo_generacion,
        pp.score_confianza
    FROM predicciones_pale pp
    JOIN tipos_loteria tl ON pp.tipo_loteria_id = tl.id
    WHERE pp.fecha_prediccion = fecha_consulta
    AND (tipo_loteria_consulta IS NULL OR pp.tipo_loteria_id = tipo_loteria_consulta)
    
    UNION ALL
    
    SELECT 
        'tripleta'::VARCHAR(20),
        tl.nombre,
        pt.posicion,
        CONCAT(LPAD(pt.numero_1::text, 2, '0'), '-', LPAD(pt.numero_2::text, 2, '0'), '-', LPAD(pt.numero_3::text, 2, '0')),
        pt.probabilidad,
        pt.metodo_generacion,
        pt.score_confianza
    FROM predicciones_tripleta pt
    JOIN tipos_loteria tl ON pt.tipo_loteria_id = tl.id
    WHERE pt.fecha_prediccion = fecha_consulta
    AND (tipo_loteria_consulta IS NULL OR pt.tipo_loteria_id = tipo_loteria_consulta)
    
    ORDER BY tipo_loteria, tipo_juego, posicion;
END;
$$ LANGUAGE plpgsql;

-- Función para limpiar predicciones antiguas
CREATE OR REPLACE FUNCTION limpiar_predicciones_antiguas(
    dias_antiguedad INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    registros_eliminados INTEGER := 0;
    fecha_limite DATE;
BEGIN
    fecha_limite := CURRENT_DATE - INTERVAL '1 day' * dias_antiguedad;
    
    -- Eliminar predicciones de quiniela
    DELETE FROM predicciones_quiniela WHERE fecha_prediccion < fecha_limite;
    GET DIAGNOSTICS registros_eliminados = ROW_COUNT;
    
    -- Eliminar predicciones de pale
    DELETE FROM predicciones_pale WHERE fecha_prediccion < fecha_limite;
    GET DIAGNOSTICS registros_eliminados = registros_eliminados + ROW_COUNT;
    
    -- Eliminar predicciones de tripleta
    DELETE FROM predicciones_tripleta WHERE fecha_prediccion < fecha_limite;
    GET DIAGNOSTICS registros_eliminados = registros_eliminados + ROW_COUNT;
    
    RETURN registros_eliminados;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- DATOS DE EJEMPLO
-- ========================================

-- Ejemplo de predicciones para hoy
/*
INSERT INTO predicciones_quiniela (fecha_prediccion, tipo_loteria_id, posicion, numero_predicho, probabilidad, metodo_generacion, score_confianza) VALUES
(CURRENT_DATE, 1, 1, 09, 0.85, 'neural_network', 0.92),
(CURRENT_DATE, 1, 2, 34, 0.78, 'neural_network', 0.88),
(CURRENT_DATE, 1, 3, 56, 0.72, 'neural_network', 0.85);

INSERT INTO predicciones_pale (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, probabilidad, metodo_generacion, score_confianza) VALUES
(CURRENT_DATE, 1, 1, 23, 45, 0.68, 'monte_carlo', 0.75),
(CURRENT_DATE, 1, 2, 34, 56, 0.65, 'monte_carlo', 0.72),
(CURRENT_DATE, 1, 3, 12, 78, 0.62, 'monte_carlo', 0.70);

INSERT INTO predicciones_tripleta (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, numero_3, probabilidad, metodo_generacion, score_confianza) VALUES
(CURRENT_DATE, 1, 1, 23, 54, 23, 0.45, 'statistical', 0.68),
(CURRENT_DATE, 1, 2, 12, 34, 56, 0.42, 'statistical', 0.65),
(CURRENT_DATE, 1, 3, 78, 90, 12, 0.40, 'statistical', 0.62);
*/

-- ========================================
-- COMENTARIOS FINALES
-- ========================================

/*
ESTRUCTURA DE LA BASE DE DATOS:

1. TABLAS PRINCIPALES:
   - tipos_loteria: Gana Más, Lotería Nacional
   - tipos_juego: quiniela, pale, tripleta
   - sorteos: resultados históricos
   - predicciones_quiniela: 3 predicciones diarias de números individuales
   - predicciones_pale: 3 predicciones diarias de combinaciones de 2 números
   - predicciones_tripleta: 3 predicciones diarias de combinaciones de 3 números

2. RESTRICCIONES:
   - Solo 3 predicciones por tipo de juego por día
   - Números válidos: 0-99
   - Posiciones: 1, 2, 3 (P1, P2, P3)
   - Probabilidades: 0.0 - 1.0

3. FORMATO DE PREDICCIONES:
   - Quiniela: P1:09, P2:34, P3:56
   - Pale: P1:23-45, P2:34-56, P3:12-78
   - Tripleta: P1:23-54-67, P2:12-34-56, P3:78-90-12

4. USO DESDE PYTHON:
   - Insertar 3 predicciones por tipo por día
   - Consultar predicciones con la función obtener_predicciones_dia()
   - Evaluar resultados en tabla resultados_predicciones
   - Mantener limpia la base con limpiar_predicciones_antiguas()
*/