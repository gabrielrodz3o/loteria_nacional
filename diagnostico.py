#!/usr/bin/env python3
import sys
sys.path.append('.')

print("🔍 DIAGNÓSTICO DE MODELOS")
print("=" * 40)

try:
    from models.prediction_models import model_registry
    from config.settings import settings
    
    print(f"Settings priority_models: {settings.priority_models}")
    print(f"Registered models: {model_registry.list_available_models()}")
    print(f"Total registered: {len(model_registry.list_available_models())}")
    
    # Test model creation
    for modelo in model_registry.list_available_models():
        try:
            instance = model_registry.create_model(modelo)
            print(f"{'✅' if instance else '❌'} {modelo}")
        except Exception as e:
            print(f"❌ {modelo} - {e}")
            
except Exception as e:
    print(f"ERROR: {e}")