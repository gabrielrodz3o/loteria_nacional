"""Predictions package - FIXED VERSION."""

import logging
logger = logging.getLogger(__name__)

print("[INIT] Starting predictions package import - FIXED VERSION...")

# COMENTAR LOS IMPORTS ANTIGUOS PARA EVITAR CONFLICTOS:
# from . import gradient_boosting  # ← COMENTADO
# from . import statistical        # ← COMENTADO  
# from . import random_forest      # ← COMENTADO
# from . import monte_carlo        # ← COMENTADO

print("[INIT] ⚠️  Modelos antiguos DESHABILITADOS para evitar conflictos")
print("[INIT] ✅ Usando SOLO los 6 modelos registrados en model_registry")

# Solo importar el motor de predicciones
from .predictor_engine import predictor_engine

print("[INIT] Predictions package loaded - FIXED VERSION")

__all__ = ['predictor_engine']