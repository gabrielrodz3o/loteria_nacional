# Makefile for Lottery Prediction System

.PHONY: help build up down logs shell test clean backup restore

# Default target
help:
	@echo "Lottery Prediction System - Available Commands:"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make up         - Start all services"
	@echo "  make up-dev     - Start with development profile"
	@echo "  make up-admin   - Start with admin tools (pgAdmin)"
	@echo "  make up-full    - Start with all profiles"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo ""
	@echo "🛠️  Build & Setup:"
	@echo "  make build      - Build application image"
	@echo "  make setup      - Initial setup (build + migrate + data)"
	@echo "  make migrate    - Run database migrations"
	@echo "  make seed       - Seed database with sample data"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  make logs       - View application logs"
	@echo "  make logs-db    - View database logs"
	@echo "  make status     - Check service status"
	@echo "  make health     - Check application health"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run unit tests only"
	@echo "  make test-api   - Run API tests only"
	@echo "  make lint       - Run code linting"
	@echo ""
	@echo "🔧 Maintenance:"
	@echo "  make shell      - Access application shell"
	@echo "  make db-shell   - Access database shell"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make backup     - Backup database"
	@echo "  make restore    - Restore database from backup"
	@echo ""
	@echo "🤖 ML Operations:"
	@echo "  make train      - Train ML models"
	@echo "  make predict    - Generate predictions"
	@echo "  make scrape     - Run scraping test"

# Service management
up:
	docker-compose up -d

up-dev:
	docker-compose up -d --build

up-admin:
	docker-compose --profile admin up -d

up-monitoring:
	docker-compose --profile monitoring up -d

up-full:
	docker-compose --profile admin --profile monitoring --profile production up -d

down:
	docker-compose down

restart:
	docker-compose restart

# Build and setup
build:
	docker-compose build

setup: build migrate seed
	@echo "✅ Setup completed!"

# Database operations
migrate:
	docker-compose exec loteria_app alembic upgrade head

seed:
	docker-compose exec loteria_app python -c "\
	from scraping.data_cleaner import seed_sample_data; \
	seed_sample_data()"

# Monitoring and logs
logs:
	docker-compose logs -f loteria_app

logs-db:
	docker-compose logs -f postgres_loteria

logs-all:
	docker-compose logs -f

status:
	docker-compose ps

health:
	@echo "🔍 Checking application health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Application not responding"
	@echo ""
	@echo "🔍 Checking database..."
	@docker-compose exec postgres_loteria pg_isready -U loteria_user || echo "❌ Database not ready"

# Testing
test:
	docker-compose exec loteria_app pytest tests/ -v

test-unit:
	docker-compose exec loteria_app pytest tests/test_predictions.py -v

test-api:
	docker-compose exec loteria_app pytest tests/test_api.py -v

test-coverage:
	docker-compose exec loteria_app pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	docker-compose exec loteria_app flake8 .
	docker-compose exec loteria_app black --check .
	docker-compose exec loteria_app isort --check-only .

format:
	docker-compose exec loteria_app black .
	docker-compose exec loteria_app isort .

# Shell access
shell:
	docker-compose exec loteria_app bash

db-shell:
	docker-compose exec postgres_loteria psql -U loteria_user -d loteria_db

redis-shell:
	docker-compose exec redis_loteria redis-cli

# ML Operations
train:
	@echo "🤖 Training ML models..."
	curl -X POST "http://localhost:8000/entrenar-modelos?forzar=true"

predict:
	@echo "🔮 Generating predictions..."
	curl -X POST "http://localhost:8000/generar-predicciones?tipo_loteria_id=1"

scrape:
	@echo "🕷️  Testing scraping..."
	docker-compose exec loteria_app python -c "\
	from scraping.scraper import lottery_scraper; \
	import json; \
	result = lottery_scraper.test_scraping(); \
	print(json.dumps(result, indent=2, default=str))"

# Backup and restore
backup:
	@echo "💾 Creating database backup..."
	@mkdir -p backups
	docker-compose exec postgres_loteria pg_dump -U loteria_user -d loteria_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

restore:
	@echo "⚠️  This will restore database from latest backup"
	@read -p "Continue? [y/N]: " confirm && [ "$$confirm" = "y" ]
	@latest_backup=$$(ls -t backups/*.sql | head -1); \
	if [ -n "$$latest_backup" ]; then \
		echo "📥 Restoring from $$latest_backup"; \
		cat "$$latest_backup" | docker-compose exec -T postgres_loteria psql -U loteria_user -d loteria_db; \
		echo "✅ Database restored"; \
	else \
		echo "❌ No backup files found"; \
	fi

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

clean-all: clean
	docker-compose down --rmi all -v
	docker system prune -a -f

# Development helpers
install-dev:
	pip install -r requirements.txt
	pip install -e .

run-local:
	python main.py

# Cache management
clear-cache:
	curl -X DELETE "http://localhost:8000/cache/limpiar"

# System info
info:
	@echo "📊 System Information:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Docker Compose Version:"
	@docker-compose version
	@echo ""
	@echo "Running Services:"
	@docker-compose ps
	@echo ""
	@echo "Disk Usage:"
	@docker system df
	@echo ""
	@echo "API Endpoints:"
	@curl -s http://localhost:8000/tipos-loteria | python -m json.tool || echo "API not available"

# Performance monitoring
monitor:
	@echo "📈 Performance Monitoring:"
	@echo "API: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000 (admin/LoteriaGrafana2024!)"
	@echo "Prometheus: http://localhost:9090"
	@echo "pgAdmin: http://localhost:5051 (admin@loteria.com/LoteriaAdmin2024!)"

# Quick start for new users
quickstart:
	@echo "🚀 Quick Start Guide:"
	@echo "1. Copy environment file: cp .env.example .env"
	@echo "2. Start services: make up"
	@echo "3. Run migrations: make migrate"
	@echo "4. Check health: make health"
	@echo "5. View API docs: http://localhost:8000/docs"
	@echo ""
	@echo "Run 'make help' for more commands"

# Development workflow
dev-setup: build migrate seed train
	@echo "🎯 Development environment ready!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

# Production deployment
deploy-prod:
	@echo "🚀 Deploying to production..."
	docker-compose --profile production up -d --build
	sleep 10
	make migrate
	make health

# Update and restart
update:
	git pull
	make build
	make restart
	make migrate
	@echo "✅ System updated!"

# Emergency commands
emergency-stop:
	docker-compose kill
	docker-compose down

emergency-restart:
	make emergency-stop
	make up
	sleep 10
	make health

# Export/Import
export-predictions:
	@echo "📤 Exporting predictions..."
	@mkdir -p exports
	docker-compose exec postgres_loteria pg_dump -U loteria_user -d loteria_db \
		-t predicciones_quiniela -t predicciones_pale -t predicciones_tripleta \
		> exports/predictions_$(shell date +%Y%m%d).sql

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@command -v xdg-open >/dev/null && xdg-open http://localhost:8000/docs || \
	 command -v open >/dev/null && open http://localhost:8000/docs || \
	 echo "Open http://localhost:8000/docs in your browser"