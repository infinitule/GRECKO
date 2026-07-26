# GRECKO — developer & operator entrypoints.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install dev test verify demo figures serve gif lint clean \
        docker docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PY) -m pip install -e .

dev: ## Install with dev + viz extras
	$(PY) -m pip install -e ".[dev,viz]"

test: ## Run the full acceptance suite
	$(PY) -m pytest sim/tests -q

verify: ## Run the architectural invariant gate (CI gate)
	$(PY) -m tools.verify_invariants

demo: ## Run the headline cost-exchange study (fast)
	$(PY) demo.py --fast

figures: ## (Re)generate docs/figures static assets
	$(PY) -m tools.make_figures

gif: ## Render the swarm-for-swarm demo animation
	$(PY) -m tools.make_demo_gif

serve: ## Start the C2 WebSocket bridge (port 8765)
	$(PY) -m sim.bridge.server --host 0.0.0.0 --port 8765

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker: ## Build the C2-server container image
	docker build -t grecko/c2-server:1.0.0 .

docker-up: ## Build and start the full stack (server + console)
	docker compose up --build -d

docker-down: ## Stop the stack
	docker compose down
