
from pydantic_settings import BaseSettings
from typing import List

class TempSettings(BaseSettings):
    priority_models: List[str] = [
        "frequencyanalysis", "randomforest", "montecarlo", "bayesian", "lightgbm", "xgboost"
    ]

# Sobrescribir la instancia global
settings = TempSettings()
