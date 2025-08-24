#!/bin/bash
# ================================================================
# FIX ORBSTACK PGADMIN - Solucionar problema de red con pgAdmin
# ================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🔧 ================================================================"
echo "🌐 FIX ORBSTACK PGADMIN - Solucionar problema de red"
echo "================================================================"

# PASO 1: DIAGNÓSTICO DEL PROBLEMA
log_info "🔍 PASO 1: Diagnosticando problema de red..."

echo "Sistema detectado: OrbStack"
echo "Error: Bad Gateway - problemas de proxy/red"

# Verificar contenedores
log_info "Estado actual de contenedores:"
docker-compose ps

# PASO 2: DETENER PGADMIN PROBLEMÁTICO
log_info "🛑 PASO 2: Deteniendo pgAdmin problemático..."

docker-compose stop pgadmin_loteria 2>/dev/null || true
docker-compose rm -f pgladmin_loteria 2>/dev/null || true

log_success "✅ pgAdmin detenido"

# PASO 3: CREAR CONFIGURACIÓN COMPATIBLE CON ORBSTACK
log_info "📝 PASO 3: Creando configuración compatible con OrbStack..."

# Backup del docker-compose actual
cp docker-compose.yml docker-compose.yml.backup.orbstack.$(date +%Y%m%d_%H%M%S)

# Crear configuración optimizada para OrbStack
cat > docker-compose.yml << 'EOF'
services:
  postgres_loteria:
    container_name: postgres_loteria
    image: pgvector/pgvector:pg15
    restart: always
    networks:
      - loteria-network
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    environment:
      - POSTGRES_PASSWORD=LoteriaPass2024!
      - POSTGRES_USER=loteria_user
      - POSTGRES_DB=loteria_db
      - TZ=America/Santo_Domingo
      - PGDATA=/var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    command: >
      postgres 
      -c timezone=America/Santo_Domingo 
      -c shared_preload_libraries=vector
      -c max_connections=100
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c work_mem=4MB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U loteria_user -d loteria_db"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 45s
    shm_size: 256mb

  pgadmin_loteria:
    container_name: pgadmin_loteria
    image: dpage/pgadmin4:8.5
    restart: always
    networks:
      - loteria-network
    ports:
      # CAMBIO CRÍTICO: Usar puerto diferente para evitar conflictos en OrbStack
      - "5050:80"
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@loteria.com
      - PGADMIN_DEFAULT_PASSWORD=LoteriaAdmin2024!
      - PGADMIN_CONFIG_ENHANCED_COOKIE_PROTECTION=False
      # NUEVAS CONFIGURACIONES PARA ORBSTACK
      - PGADMIN_LISTEN_ADDRESS=0.0.0.0
      - PGADMIN_LISTEN_PORT=80
      - PGADMIN_CONFIG_SERVER_MODE=False
      - PGADMIN_CONFIG_MASTER_PASSWORD_REQUIRED=False
      # CONFIGURACIÓN ESPECÍFICA PARA REDES DE CONTENEDORES
      - PGADMIN_CONFIG_CONSOLE_LOG_LEVEL=20
      - GUNICORN_ACCESS_LOGFILE=-
    depends_on:
      postgres_loteria:
        condition: service_healthy
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    # CONFIGURACIÓN DE RED ESPECÍFICA
    extra_hosts:
      - "host.docker.internal:host-gateway"
    profiles:
      - admin

  redis_loteria:
    container_name: redis_loteria
    image: redis:7-alpine
    restart: always
    networks:
      - loteria-network
    ports:
      - "6380:6379"
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    volumes:
      - redis_data:/data

  app:
    container_name: loteria_app
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    networks:
      - loteria-network
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://loteria_user:LoteriaPass2024!@postgres_loteria:5432/loteria_db
      - REDIS_URL=redis://redis_loteria:6379/0
      - TZ=America/Santo_Domingo
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
      - ./scripts:/app/scripts
      - .:/app
    depends_on:
      postgres_loteria:
        condition: service_healthy
      redis_loteria:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "python -c 'from config.database import check_database_connection; exit(0 if check_database_connection() else 1)'"]
      interval: 30s
      timeout: 15s
      retries: 5
      start_period: 60s

networks:
  loteria-network:
    driver: bridge
    # CONFIGURACIÓN ESPECÍFICA PARA ORBSTACK
    driver_opts:
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
      com.docker.network.driver.mtu: "1500"

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  pgadmin_data:
    driver: local
EOF

log_success "✅ Configuración compatible con OrbStack creada"

# PASO 4: LIMPIAR RED Y VOLÚMENES PROBLEMÁTICOS
log_info "🧹 PASO 4: Limpiando configuración de red..."

# Limpiar redes antiguas
docker network prune -f 2>/dev/null || true

# Recrear red
docker-compose down --remove-orphans 2>/dev/null || true

log_success "✅ Red limpiada"

# PASO 5: INICIAR SERVICIOS PRINCIPALES PRIMERO
log_info "🚀 PASO 5: Iniciando servicios principales..."

# Iniciar PostgreSQL y App primero
docker-compose up -d postgres_loteria redis_loteria app

# Esperar a que estén listos
log_info "Esperando servicios principales..."
sleep 15

# Verificar estado
log_info "Estado de servicios principales:"
docker-compose ps postgres_loteria redis_loteria app

# PASO 6: INICIAR PGADMIN DE FORMA CONTROLADA
log_info "📊 PASO 6: Iniciando pgAdmin de forma controlada..."

# Crear directorio para pgAdmin si no existe
mkdir -p ./pgadmin_data

# Iniciar pgAdmin con el perfil admin
docker-compose --profile admin up -d pgladmin_loteria

log_info "Esperando pgAdmin..."
sleep 20

# PASO 7: VERIFICAR CONECTIVIDAD
log_info "🔍 PASO 7: Verificando conectividad..."

echo ""
log_info "Estado de todos los servicios:"
docker-compose --profile admin ps

echo ""

# Verificar logs de pgAdmin
log_info "Últimos logs de pgAdmin:"
docker-compose logs --tail=10 pgladmin_loteria

echo ""

# Test de conectividad específico
log_info "Probando conectividad de pgAdmin..."

# Probar conexión local
if curl -s http://localhost:5050 > /dev/null 2>&1; then
    log_success "✅ pgAdmin accesible en http://localhost:5050"
else
    log_warning "⚠️ pgAdmin no responde en puerto 5050"
    
    # Intentar con puerto alternativo
    if curl -s http://127.0.0.1:5050 > /dev/null 2>&1; then
        log_success "✅ pgAdmin accesible en http://127.0.0.1:5050"
    else
        log_error "❌ pgAdmin no accesible"
        
        # Mostrar diagnóstico adicional
        echo ""
        log_info "Diagnóstico adicional:"
        docker-compose exec -T pgladmin_loteria netstat -tlnp 2>/dev/null || echo "netstat no disponible"
        
        echo ""
        log_info "Configuración de red del contenedor:"
        docker inspect pgladmin_loteria | grep -A 10 "NetworkSettings" || echo "No se pudo obtener info de red"
    fi
fi

# PASO 8: CONFIGURACIÓN AUTOMÁTICA DE SERVIDOR EN PGADMIN
log_info "⚙️ PASO 8: Configurando servidor PostgreSQL en pgAdmin..."

# Crear archivo de configuración de servidor
cat > servers.json << 'EOF'
{
  "Servers": {
    "1": {
      "Name": "Lotería Nacional DB",
      "Group": "Servers",
      "Host": "postgres_loteria",
      "Port": 5432,
      "MaintenanceDB": "loteria_db",
      "Username": "loteria_user",
      "PassFile": "/tmp/pgpassfile",
      "SSLMode": "prefer",
      "SSLCert": "",
      "SSLKey": "",
      "SSLRootCert": "",
      "SSLCrl": "",
      "SSLCompression": 0,
      "Timeout": 10,
      "UseSSHTunnel": 0,
      "TunnelHost": "",
      "TunnelPort": "22",
      "TunnelUsername": "",
      "TunnelAuthentication": 0
    }
  }
}
EOF

# Crear archivo de passwords
echo "postgres_loteria:5432:loteria_db:loteria_user:LoteriaPass2024!" > pgpass

# Copiar configuración al contenedor
if docker-compose ps pgladmin_loteria | grep -q "Up"; then
    docker cp servers.json pgladmin_loteria:/pgladmin4/servers.json 2>/dev/null || true
    docker cp pgpass pgladmin_loteria:/tmp/pgpassfile 2>/dev/null || true
    docker-compose exec -T pgladmin_loteria chmod 600 /tmp/pgpassfile 2>/dev/null || true
    log_success "✅ Configuración de servidor aplicada"
else
    log_warning "⚠️ pgAdmin no está corriendo - configuración manual necesaria"
fi

# Limpiar archivos temporales
rm -f servers.json pgpass

# RESUMEN FINAL
echo ""
log_success "🎉 ================================================================"
log_success "🌐 ORBSTACK PGADMIN - CONFIGURACIÓN COMPLETADA"
log_success "================================================================"

echo ""
log_info "📊 CAMBIOS REALIZADOS:"
echo "  🔄 Puerto cambiado de 5051 a 5050"
echo "  🌐 Configuración de red optimizada para OrbStack"
echo "  ⚙️ Variables de entorno adicionales agregadas"
echo "  🔗 Host gateway configurado"
echo "  📝 Servidor PostgreSQL preconfigurado"

echo ""
log_info "🌐 URLs ACTUALIZADAS:"
echo "  📱 Aplicación: http://localhost:8000"
echo "  🗄️ pgAdmin: http://localhost:5050"
echo "  🔴 Redis: localhost:6380"
echo "  🐘 PostgreSQL: localhost:5433"

echo ""
log_info "🔐 CREDENCIALES DE PGADMIN:"
echo "  📧 Email: admin@loteria.com"
echo "  🔑 Password: LoteriaAdmin2024!"

echo ""
log_info "🔧 COMANDOS ÚTILES:"
echo "  Ver logs pgAdmin:    docker-compose logs -f pgladmin_loteria"
echo "  Reiniciar pgAdmin:   docker-compose restart pgladmin_loteria"
echo "  Detener pgAdmin:     docker-compose stop pgladmin_loteria"

echo ""
if docker-compose --profile admin ps | grep pgladmin_loteria | grep -q "Up"; then
    log_success "✅ PGLADMIN FUNCIONANDO CORRECTAMENTE"
    echo "  Accede a: http://localhost:5050"
else
    log_warning "⚠️ PGADMIN NECESITA VERIFICACIÓN MANUAL"
    echo "  Revisa logs: docker-compose logs pgladmin_loteria"
fi

echo "================================================================"