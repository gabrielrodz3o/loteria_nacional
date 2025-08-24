-- ========================================
-- EJEMPLO DE PREDICCIONES DIARIAS PARA LOTERÍA NACIONAL
-- Fecha: 20-09-2025
-- ========================================

-- Primero, insertar las predicciones del día 20-09-2025 para Lotería Nacional (tipo_loteria_id = 2)

-- ========================================
-- PREDICCIONES DE QUINIELA (3 números individuales)
-- Python elige el mejor método para cada predicción
-- ========================================
INSERT INTO predicciones_quiniela (fecha_prediccion, tipo_loteria_id, posicion, numero_predicho, probabilidad, metodo_generacion, score_confianza) VALUES
('2025-09-20', 2, 1, 09, 0.87, 'neural_network', 0.92),  -- P1: 09 (Red neuronal ganó)
('2025-09-20', 2, 2, 34, 0.84, 'monte_carlo', 0.89),     -- P2: 34 (Monte Carlo ganó)
('2025-09-20', 2, 3, 56, 0.81, 'statistical', 0.86);     -- P3: 56 (Estadístico ganó)

-- ========================================
-- PREDICCIONES DE PALE (3 combinaciones de 2 números)
-- Python elige el mejor método para cada predicción
-- ========================================
INSERT INTO predicciones_pale (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, probabilidad, metodo_generacion, score_confianza) VALUES
('2025-09-20', 2, 1, 23, 45, 0.75, 'statistical', 0.83),   -- P1: 23-45 (Estadístico ganó)
('2025-09-20', 2, 2, 34, 56, 0.71, 'neural_network', 0.80), -- P2: 34-56 (Red neuronal ganó)
('2025-09-20', 2, 3, 12, 78, 0.68, 'monte_carlo', 0.77);   -- P3: 12-78 (Monte Carlo ganó)

-- ========================================
-- PREDICCIONES DE TRIPLETA (3 combinaciones de 3 números)
-- Python elige el mejor método para cada predicción
-- ========================================
INSERT INTO predicciones_tripleta (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, numero_3, probabilidad, metodo_generacion, score_confianza) VALUES
('2025-09-20', 2, 1, 23, 54, 67, 0.58, 'monte_carlo', 0.74), -- P1: 23-54-67 (Monte Carlo ganó)
('2025-09-20', 2, 2, 12, 34, 89, 0.54, 'neural_network', 0.71), -- P2: 12-34-89 (Red neuronal ganó)
('2025-09-20', 2, 3, 45, 76, 91, 0.51, 'statistical', 0.68); -- P3: 45-76-91 (Estadístico ganó)

-- ========================================
-- CONSULTAR LAS PREDICCIONES DEL DÍA
-- ========================================

-- Opción 1: Usar la función personalizada
SELECT * FROM obtener_predicciones_dia('2025-09-20', 2);

-- Opción 2: Consulta directa con formato legible
SELECT 
    'LOTERÍA NACIONAL - 20/09/2025' as titulo,
    'QUINIELA' as tipo_juego,
    CONCAT('P', posicion, ': ', LPAD(numero_predicho::text, 2, '0')) as prediccion,
    CONCAT(ROUND(probabilidad * 100, 1), '%') as probabilidad_pct,
    metodo_generacion as metodo,
    CONCAT(ROUND(score_confianza * 100, 1), '%') as confianza_pct
FROM predicciones_quiniela 
WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
ORDER BY posicion

UNION ALL

SELECT 
    'LOTERÍA NACIONAL - 20/09/2025' as titulo,
    'PALE' as tipo_juego,
    CONCAT('P', posicion, ': ', LPAD(numero_1::text, 2, '0'), '-', LPAD(numero_2::text, 2, '0')) as prediccion,
    CONCAT(ROUND(probabilidad * 100, 1), '%') as probabilidad_pct,
    metodo_generacion as metodo,
    CONCAT(ROUND(score_confianza * 100, 1), '%') as confianza_pct
FROM predicciones_pale 
WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
ORDER BY posicion

UNION ALL

SELECT 
    'LOTERÍA NACIONAL - 20/09/2025' as titulo,
    'TRIPLETA' as tipo_juego,
    CONCAT('P', posicion, ': ', LPAD(numero_1::text, 2, '0'), '-', LPAD(numero_2::text, 2, '0'), '-', LPAD(numero_3::text, 2, '0')) as prediccion,
    CONCAT(ROUND(probabilidad * 100, 1), '%') as probabilidad_pct,
    metodo_generacion as metodo,
    CONCAT(ROUND(score_confianza * 100, 1), '%') as confianza_pct
FROM predicciones_tripleta 
WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
ORDER BY posicion;

-- ========================================
-- RESULTADO ESPERADO DE LA CONSULTA:
-- ========================================
/*
TITULO                          | TIPO_JUEGO | PREDICCION    | PROBABILIDAD | METODO         | CONFIANZA
--------------------------------|------------|---------------|--------------|----------------|----------
LOTERÍA NACIONAL - 20/09/2025  | QUINIELA   | P1: 09        | 87.0%        | neural_network | 92.0%
LOTERÍA NACIONAL - 20/09/2025  | QUINIELA   | P2: 34        | 84.0%        | monte_carlo    | 89.0%
LOTERÍA NACIONAL - 20/09/2025  | QUINIELA   | P3: 56        | 81.0%        | statistical    | 86.0%
LOTERÍA NACIONAL - 20/09/2025  | PALE       | P1: 23-45     | 75.0%        | statistical    | 83.0%
LOTERÍA NACIONAL - 20/09/2025  | PALE       | P2: 34-56     | 71.0%        | neural_network | 80.0%
LOTERÍA NACIONAL - 20/09/2025  | PALE       | P3: 12-78     | 68.0%        | monte_carlo    | 77.0%
LOTERÍA NACIONAL - 20/09/2025  | TRIPLETA   | P1: 23-54-67  | 58.0%        | monte_carlo    | 74.0%
LOTERÍA NACIONAL - 20/09/2025  | TRIPLETA   | P2: 12-34-89  | 54.0%        | neural_network | 71.0%
LOTERÍA NACIONAL - 20/09/2025  | TRIPLETA   | P3: 45-76-91  | 51.0%        | statistical    | 68.0%
*/

-- ========================================
-- FORMATO SIMPLIFICADO PARA MOSTRAR AL USUARIO:
-- ========================================
/*
📅 PREDICCIONES LOTERÍA NACIONAL - 20/09/2025

🎯 QUINIELA (Números individuales):
   P1: 09 (87.0% probabilidad - Red Neuronal)
   P2: 34 (84.0% probabilidad - Monte Carlo)
   P3: 56 (81.0% probabilidad - Estadístico)

🎲 PALE (Combinaciones de 2 números):
   P1: 23-45 (75.0% probabilidad - Estadístico)
   P2: 34-56 (71.0% probabilidad - Red Neuronal)
   P3: 12-78 (68.0% probabilidad - Monte Carlo)

🎪 TRIPLETA (Combinaciones de 3 números):
   P1: 23-54-67 (58.0% probabilidad - Monte Carlo)
   P2: 12-34-89 (54.0% probabilidad - Red Neuronal)
   P3: 45-76-91 (51.0% probabilidad - Estadístico)

⏰ Sorteo: 9:00 PM
🤖 Generado automáticamente por el sistema de IA
*/

-- ========================================
-- CONSULTA PARA PYTHON (JSON FORMAT)
-- ========================================
SELECT 
    json_build_object(
        'fecha', '2025-09-20',
        'loteria', 'Lotería Nacional',
        'hora_sorteo', '21:00',
        'predicciones', json_build_object(
            'quiniela', (
                SELECT json_agg(
                    json_build_object(
                        'posicion', posicion,
                        'numero', LPAD(numero_predicho::text, 2, '0'),
                        'probabilidad', ROUND(probabilidad * 100, 1),
                        'metodo', metodo_generacion,
                        'confianza', ROUND(score_confianza * 100, 1)
                    ) ORDER BY posicion
                )
                FROM predicciones_quiniela 
                WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
            ),
            'pale', (
                SELECT json_agg(
                    json_build_object(
                        'posicion', posicion,
                        'combinacion', CONCAT(LPAD(numero_1::text, 2, '0'), '-', LPAD(numero_2::text, 2, '0')),
                        'probabilidad', ROUND(probabilidad * 100, 1),
                        'metodo', metodo_generacion,
                        'confianza', ROUND(score_confianza * 100, 1)
                    ) ORDER BY posicion
                )
                FROM predicciones_pale 
                WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
            ),
            'tripleta', (
                SELECT json_agg(
                    json_build_object(
                        'posicion', posicion,
                        'combinacion', CONCAT(LPAD(numero_1::text, 2, '0'), '-', LPAD(numero_2::text, 2, '0'), '-', LPAD(numero_3::text, 2, '0')),
                        'probabilidad', ROUND(probabilidad * 100, 1),
                        'metodo', metodo_generacion,
                        'confianza', ROUND(score_confianza * 100, 1)
                    ) ORDER BY posicion
                )
                FROM predicciones_tripleta 
                WHERE fecha_prediccion = '2025-09-20' AND tipo_loteria_id = 2
            )
        )
    ) as predicciones_json;

-- ========================================
-- CÓDIGO PYTHON PARA INSERTAR PREDICCIONES
-- ========================================
/*
import psycopg2
from datetime import date

def insertar_predicciones_dia(fecha, tipo_loteria_id, predicciones_data):
    """
    Inserta las predicciones del día en la base de datos
    
    Args:
        fecha: date object (ej: date(2025, 9, 20))
        tipo_loteria_id: int (1=Gana Más, 2=Lotería Nacional)
        predicciones_data: dict con las predicciones
    """
    conn = psycopg2.connect(
        host="localhost",
        database="loteria_nacional",
        user="tu_usuario",
        password="tu_password"
    )
    
    try:
        with conn.cursor() as cur:
            # Insertar quinielas
            for i, pred in enumerate(predicciones_data['quiniela'], 1):
                cur.execute(
                    "INSERT INTO predicciones_quiniela (fecha_prediccion, tipo_loteria_id, posicion, numero_predicho, probabilidad, metodo_generacion, score_confianza) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (fecha, tipo_loteria_id, i, pred['numero'], pred['probabilidad'], pred['metodo'], pred['confianza'])
                )
            
            # Insertar pales
            for i, pred in enumerate(predicciones_data['pale'], 1):
                cur.execute(
                    "INSERT INTO predicciones_pale (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, probabilidad, metodo_generacion, score_confianza) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (fecha, tipo_loteria_id, i, pred['numero_1'], pred['numero_2'], pred['probabilidad'], pred['metodo'], pred['confianza'])
                )
            
            # Insertar tripletas
            for i, pred in enumerate(predicciones_data['tripleta'], 1):
                cur.execute(
                    "INSERT INTO predicciones_tripleta (fecha_prediccion, tipo_loteria_id, posicion, numero_1, numero_2, numero_3, probabilidad, metodo_generacion, score_confianza) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (fecha, tipo_loteria_id, i, pred['numero_1'], pred['numero_2'], pred['numero_3'], pred['probabilidad'], pred['metodo'], pred['confianza'])
                )
            
            conn.commit()
            print(f"Predicciones insertadas exitosamente para {fecha}")
            
    except Exception as e:
        conn.rollback()
        print(f"Error al insertar predicciones: {e}")
    finally:
        conn.close()

# Ejemplo de uso con métodos mixtos (Python elige el mejor):
predicciones_ejemplo = {
    'quiniela': [
        {'numero': 9, 'probabilidad': 0.87, 'metodo': 'neural_network', 'confianza': 0.92},
        {'numero': 34, 'probabilidad': 0.84, 'metodo': 'monte_carlo', 'confianza': 0.89},
        {'numero': 56, 'probabilidad': 0.81, 'metodo': 'statistical', 'confianza': 0.86}
    ],
    'pale': [
        {'numero_1': 23, 'numero_2': 45, 'probabilidad': 0.75, 'metodo': 'statistical', 'confianza': 0.83},
        {'numero_1': 34, 'numero_2': 56, 'probabilidad': 0.71, 'metodo': 'neural_network', 'confianza': 0.80},
        {'numero_1': 12, 'numero_2': 78, 'probabilidad': 0.68, 'metodo': 'monte_carlo', 'confianza': 0.77}
    ],
    'tripleta': [
        {'numero_1': 23, 'numero_2': 54, 'numero_3': 67, 'probabilidad': 0.58, 'metodo': 'monte_carlo', 'confianza': 0.74},
        {'numero_1': 12, 'numero_2': 34, 'numero_3': 89, 'probabilidad': 0.54, 'metodo': 'neural_network', 'confianza': 0.71},
        {'numero_1': 45, 'numero_2': 76, 'numero_3': 91, 'probabilidad': 0.51, 'metodo': 'statistical', 'confianza': 0.68}
    ]
}

insertar_predicciones_dia(date(2025, 9, 20), 2, predicciones_ejemplo)
*/