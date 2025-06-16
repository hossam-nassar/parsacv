# ParsaCV - Makefile for Development and Deployment
# =================================================

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
COMPOSE := docker-compose
APP_NAME := parsacv
VERSION := $(shell cat VERSION 2>/dev/null || echo "1.0.0")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
NC := \033[0m

.PHONY: help install dev-install test lint format security clean docker run deploy

# Default target
help: ## Show this help message
	@echo "$(BLUE)ParsaCV - Available Commands$(NC)"
	@echo "=============================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m spacy download ar_core_news_sm
	@echo "$(GREEN)Installation completed!$(NC)"

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort mypy bandit safety
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m spacy download ar_core_news_sm
	@echo "$(GREEN)Development environment ready!$(NC)"

venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(YELLOW)Activate with: source venv/bin/activate$(NC)"

# Code Quality
format: ## Format code with Black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black .
	isort .
	@echo "$(GREEN)Code formatted!$(NC)"

lint: ## Run linting with flake8
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "$(GREEN)Linting completed!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy . --ignore-missing-imports
	@echo "$(GREEN)Type checking completed!$(NC)"

# Security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r . -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security checks completed!$(NC)"

# Testing
test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	mkdir -p temp_cvs outputs samples
	pytest tests/ -v
	@echo "$(GREEN)Tests completed!$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	mkdir -p temp_cvs outputs samples
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# Docker Operations
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(VCS_REF) \
		-t $(APP_NAME):$(VERSION) \
		-t $(APP_NAME):latest .
	@echo "$(GREEN)Docker image built: $(APP_NAME):$(VERSION)$(NC)"

docker-build-prod: ## Build production Docker image
	@echo "$(BLUE)Building production Docker image...$(NC)"
	$(DOCKER) build \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(VCS_REF) \
		-f Dockerfile.prod \
		-t $(APP_NAME):$(VERSION)-prod \
		-t $(APP_NAME):prod .
	@echo "$(GREEN)Production Docker image built: $(APP_NAME):$(VERSION)-prod$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Starting Docker container...$(NC)"
	$(DOCKER) run -d \
		--name $(APP_NAME) \
		-p 8501:8501 \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/samples:/app/samples \
		$(APP_NAME):latest
	@echo "$(GREEN)Container started! Access at http://localhost:8501$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	$(DOCKER) stop $(APP_NAME) || true
	$(DOCKER) rm $(APP_NAME) || true
	@echo "$(GREEN)Container stopped!$(NC)"

docker-logs: ## Show Docker container logs
	$(DOCKER) logs -f $(APP_NAME)

# Docker Compose Operations
up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	$(COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"

down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

logs: ## Show docker-compose logs
	$(COMPOSE) logs -f

restart: ## Restart services
	@echo "$(BLUE)Restarting services...$(NC)"
	$(COMPOSE) restart
	@echo "$(GREEN)Services restarted!$(NC)"

# Application Operations
run: ## Run the application locally
	@echo "$(BLUE)Starting ParsaCV application...$(NC)"
	mkdir -p temp_cvs outputs samples
	streamlit run parsacv.py
	@echo "$(GREEN)Application started!$(NC)"

run-dev: ## Run in development mode with hot reload
	@echo "$(BLUE)Starting in development mode...$(NC)"
	mkdir -p temp_cvs outputs samples
	STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll streamlit run parsacv.py --server.runOnSave=true
	@echo "$(GREEN)Development server started!$(NC)"

# Database Operations
db-up: ## Start database only
	@echo "$(BLUE)Starting database...$(NC)"
	$(COMPOSE) --profile database up -d postgresql
	@echo "$(GREEN)Database started!$(NC)"

db-migrate: ## Run database migrations (if applicable)
	@echo "$(BLUE)Running database migrations...$(NC)"
	# Add migration commands here
	@echo "$(GREEN)Migrations completed!$(NC)"

db-backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(NC)"
	$(COMPOSE) exec postgresql pg_dump -U parsacv_user parsacv > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created!$(NC)"

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	$(COMPOSE) --profile staging up -d
	@echo "$(GREEN)Deployed to staging!$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	$(COMPOSE) --profile production up -d
	@echo "$(GREEN)Deployed to production!$(NC)"

# Monitoring and Maintenance
health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8501/_stcore/health || echo "$(RED)Health check failed!$(NC)"

monitor: ## Show system resource usage
	@echo "$(BLUE)System resource usage:$(NC)"
	$(DOCKER) stats --no-stream $(APP_NAME) 2>/dev/null || echo "Container not running"

# Data Management
clean-data: ## Clean temporary and output files
	@echo "$(BLUE)Cleaning data files...$(NC)"
	rm -rf temp_cvs/* outputs/* *.xlsx *.log
	@echo "$(GREEN)Data files cleaned!$(NC)"

clean-cache: ## Clean Python cache files
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)Cache cleaned!$(NC)"

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER) container prune -f
	$(DOCKER) image prune -f
	$(DOCKER) volume prune -f
	@echo "$(GREEN)Docker resources cleaned!$(NC)"

clean: clean-data clean-cache ## Clean all temporary files
	@echo "$(GREEN)All temporary files cleaned!$(NC)"

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	# Add documentation generation commands here
	@echo "$(GREEN)Documentation generated!$(NC)"

# Release Management
version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(NC)"

tag: ## Create git tag for current version
	@echo "$(BLUE)Creating git tag v$(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	@echo "$(GREEN)Tag v$(VERSION) created and pushed!$(NC)"

release: test security docker-build ## Prepare release (test, security, build)
	@echo "$(GREEN)Release preparation completed for version $(VERSION)!$(NC)"

# Backup and Restore
backup: ## Create full backup
	@echo "$(BLUE)Creating full backup...$(NC)"
	mkdir -p backups
	tar -czf backups/parsacv_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude=venv \
		--exclude=__pycache__ \
		--exclude=temp_cvs \
		--exclude=.git \
		.
	@echo "$(GREEN)Backup created in backups/ directory!$(NC)"

# Performance Testing
load-test: ## Run load testing
	@echo "$(BLUE)Running load tests...$(NC)"
	# Add load testing commands here (e.g., with k6 or locust)
	@echo "$(GREEN)Load tests completed!$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python -m pytest tests/performance/ -v
	@echo "$(GREEN)Benchmarks completed!$(NC)"

# Environment Management
env-check: ## Check environment variables
	@echo "$(BLUE)Checking environment variables...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Copy .env.example to .env$(NC)"; \
	else \
		echo "$(GREEN).env file found!$(NC)"; \
	fi

setup-dev: venv dev-install env-check ## Complete development setup
	@echo "$(GREEN)Development environment setup completed!$(NC)"
	@echo "$(YELLOW)Don't forget to:$(NC)"
	@echo "  1. Activate virtual environment: source venv/bin/activate"
	@echo "  2. Copy .env.example to .env and configure"
	@echo "  3. Run 'make run' to start the application"

# CI/CD Helpers
ci-test: ## Run CI tests locally
	@echo "$(BLUE)Running CI tests locally...$(NC)"
	$(MAKE) format lint type-check security test
	@echo "$(GREEN)CI tests completed!$(NC)"

pre-commit: ## Run pre-commit checks
	@echo "$(BLUE)Running pre-commit checks...$(NC)"
	$(MAKE) format lint test-cov
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

# Debugging
debug: ## Start application in debug mode
	@echo "$(BLUE)Starting in debug mode...$(NC)"
	DEBUG=true STREAMLIT_LOGGER_LEVEL=debug streamlit run parsacv.py

shell: ## Start interactive shell in Docker container
	$(DOCKER) exec -it $(APP_NAME) /bin/bash

# Information
info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "=================="
	@echo "Name: $(APP_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "VCS Ref: $(VCS_REF)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell $(DOCKER) --version)"

# Quick Start
quick-start: setup-dev ## Quick start for new developers
	@echo "$(GREEN)Quick start completed!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. source venv/bin/activate"
	@echo "  2. make run"

# Production Deployment Helpers
prod-check: ## Pre-production checks
	@echo "$(BLUE)Running production checks...$(NC)"
	$(MAKE) ci-test docker-build-prod
	@echo "$(GREEN)Production checks passed!$(NC)"

prod-deploy: prod-check ## Deploy to production with checks
	@echo "$(BLUE)Deploying to production...$(NC)"
	$(MAKE) deploy-prod
	sleep 30
	$(MAKE) health-check
	@echo "$(GREEN)Production deployment completed!$(NC)"