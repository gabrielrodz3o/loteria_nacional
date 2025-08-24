#!/usr/bin/env python3
"""
SOLUCIÓN RÁPIDA SIN ERRORES DE INDENTACIÓN
Arregla el bug donde todas las predicciones tienen método 'neuralnetwork'
"""

import sys
import os
sys.path.append('.')

print("🔧 SOLUCIÓN RÁPIDA - BUG DE MÉTODOS")
print("=" * 50)

def quick_fix_method_assignment():
    """Solución rápida editando directamente el predictor_engine.py"""
    
    file_path = "predictions/predictor_engine.py"
    backup_path = "predictions/predictor_engine_backup.py"
    
    try:
        print("📝 Haciendo backup del archivo original...")
        
        # Hacer backup
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        print(f"✅ Backup creado: {backup_path}")
        
        # CORRECCIÓN PRINCIPAL: Buscar y reemplazar la línea problemática
        print("🔧 Aplicando corrección...")
        
        # Patrón a buscar (la línea que causa problemas)
        old_pattern = '''# CORRECCIÓN: Obtener nombre del modelo correctamente
            model_name = getattr(model, 'trained_model_name', 
                         getattr(model, 'model_name', 
                         getattr(model, 'name', 'unknown')))'''
        
        # Nuevo código corregido
        new_pattern = '''# CORRECCIÓN: Obtener nombre del modelo correctamente
            model_name = getattr(model, 'trained_model_name', 
                                getattr(model, 'model_name', 
                                        getattr(model, 'name', 'unknown')))
            
            # CORRECCIÓN ADICIONAL: Usar clase si no hay nombre
            if not model_name or model_name == 'unknown':
                class_name = model.__class__.__name__.lower()
                if 'realrandomforest' in class_name:
                    model_name = 'realrandomforest'
                elif 'realxgboost' in class_name:
                    model_name = 'realxgboost'
                elif 'reallightgbm' in class_name:
                    model_name = 'reallightgbm'
                elif 'neuralnetwork' in class_name:
                    model_name = 'neuralnetwork'
                elif 'frequencyanalysis' in class_name:
                    model_name = 'frequencyanalysis'
                elif 'randomforest' in class_name:
                    model_name = 'randomforest'
                elif 'montecarlo' in class_name:
                    model_name = 'montecarlo'
                elif 'bayesian' in class_name:
                    model_name = 'bayesian'
                elif 'lightgbm' in class_name:
                    model_name = 'lightgbm'
                elif 'xgboost' in class_name:
                    model_name = 'xgboost'
                else:
                    model_name = class_name'''
        
        # Aplicar el reemplazo
        if old_pattern in original_content:
            updated_content = original_content.replace(old_pattern, new_pattern)
            print("✅ Patrón encontrado y reemplazado")
        else:
            # Si no encuentra el patrón exacto, buscar versiones similares
            print("⚠️  Patrón exacto no encontrado, buscando alternativas...")
            
            # Buscar líneas problemáticas y corregir indentación
            lines = original_content.split('\n')
            updated_lines = []
            
            for i, line in enumerate(lines):
                # Buscar la línea problemática
                if 'model_name = getattr(model,' in line and 'trained_model_name' in line:
                    # Corregir indentación
                    base_indent = len(line) - len(line.lstrip())
                    
                    # Reemplazar con código corregido
                    updated_lines.append(' ' * base_indent + '# CORRECCIÓN: Obtener nombre del modelo correctamente')
                    updated_lines.append(' ' * base_indent + 'model_name = getattr(model, "trained_model_name",')
                    updated_lines.append(' ' * (base_indent + 4) + 'getattr(model, "model_name",')
                    updated_lines.append(' ' * (base_indent + 8) + 'getattr(model, "name", "unknown")))')
                    updated_lines.append(' ' * base_indent + '')
                    updated_lines.append(' ' * base_indent + '# CORRECCIÓN ADICIONAL: Usar clase si no hay nombre')
                    updated_lines.append(' ' * base_indent + 'if not model_name or model_name == "unknown":')
                    updated_lines.append(' ' * (base_indent + 4) + 'class_name = model.__class__.__name__.lower()')
                    updated_lines.append(' ' * (base_indent + 4) + 'if "realrandomforest" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "realrandomforest"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "realxgboost" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "realxgboost"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "reallightgbm" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "reallightgbm"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "neuralnetwork" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "neuralnetwork"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "frequencyanalysis" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "frequencyanalysis"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "randomforest" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "randomforest"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "montecarlo" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "montecarlo"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "bayesian" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "bayesian"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "lightgbm" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "lightgbm"')
                    updated_lines.append(' ' * (base_indent + 4) + 'elif "xgboost" in class_name:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = "xgboost"')
                    updated_lines.append(' ' * (base_indent + 4) + 'else:')
                    updated_lines.append(' ' * (base_indent + 8) + 'model_name = class_name')
                    
                    # Saltar las siguientes líneas del getattr original
                    skip_next = 0
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if 'getattr(' in lines[j] or '))' in lines[j]:
                            skip_next = j - i
                        else:
                            break
                    
                    # Saltar líneas del patrón original
                    for _ in range(skip_next):
                        if i + 1 < len(lines):
                            i += 1
                    
                    print(f"✅ Corregida línea {i + 1}")
                    
                else:
                    updated_lines.append(line)
            
            updated_content = '\n'.join(updated_lines)
        
        # Escribir archivo corregido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Archivo corregido: {file_path}")
        print("🔄 Reinicia Python para aplicar los cambios")
        
        return True
        
    except Exception as e:
        print(f"❌ Error aplicando corrección: {e}")
        
        # Restaurar backup si hay error
        try:
            if os.path.exists(backup_path):
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
                print("🔄 Archivo restaurado desde backup")
        except:
            print("❌ No se pudo restaurar el backup")
        
        return False

def test_fix():
    """Probar si la corrección funciona."""
    
    print("\n🧪 PROBANDO LA CORRECCIÓN...")
    
    try:
        # Reimportar para cargar cambios
        import importlib
        import sys
        
        modules_to_reload = [
            'predictions.predictor_engine'
        ]
        
        for module in modules_to_reload:
            if module in sys.modules:
                del sys.modules[module]
        
        from predictions.predictor_engine import predictor_engine
        from datetime import date
        
        print("🔮 Generando predicciones de prueba...")
        
        preds = predictor_engine.generar_predicciones_diarias(
            date.today().strftime('%Y-%m-%d'), 2
        )
        
        methods = set()
        total_preds = 0
        
        for game in ['quiniela', 'pale', 'tripleta']:
            for pred in preds.get(game, []):
                method = pred.get('metodo_generacion', 'unknown')
                methods.add(method)
                total_preds += 1
        
        print(f"📊 Resultado del test:")
        print(f"   Total predicciones: {total_preds}")
        print(f"   Métodos únicos: {len(methods)}")
        print(f"   Lista métodos: {sorted(methods)}")
        
        if len(methods) >= 3:
            print("✅ ¡CORRECCIÓN EXITOSA! Hay diversidad de métodos")
            return True
        else:
            print("⚠️  Aún hay problemas, todos los métodos son iguales")
            return False
            
    except Exception as e:
        print(f"❌ Error probando corrección: {e}")
        return False

def main():
    print("🚀 INICIANDO SOLUCIÓN RÁPIDA...")
    
    # Aplicar corrección
    if quick_fix_method_assignment():
        print("\n✅ Corrección aplicada exitosamente")
        
        # Probar corrección
        if test_fix():
            print("\n🎉 ¡BUG SOLUCIONADO!")
            print("✅ Ahora las predicciones tendrán métodos correctos")
        else:
            print("\n⚠️  La corrección se aplicó pero aún hay problemas")
            print("🔧 Es posible que necesites reiniciar Python completamente")
    else:
        print("\n❌ Error aplicando corrección")
        return False
    
    print(f"\n🎯 PARA VERIFICAR EN FUTURAS EJECUCIONES:")
    print("python3 -c \"")
    print("from predictions.predictor_engine import predictor_engine")
    print("from datetime import date")
    print("preds = predictor_engine.generar_predicciones_diarias(date.today().strftime('%Y-%m-%d'), 2)")
    print("methods = set()")
    print("[methods.add(pred.get('metodo_generacion')) for game in ['quiniela', 'pale', 'tripleta'] for pred in preds.get(game, [])]")
    print("print(f'Métodos únicos: {len(methods)} - {sorted(methods)}')\"")

if __name__ == "__main__":
    main()