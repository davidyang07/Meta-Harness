# Meta-Harness — one entry point per thing a reviewer might want to run.
#
# `make verify` is the important one: everything this repository can prove
# without provider credentials and without spending anything. The real
# implementation lives in scripts/verify.sh so it runs identically for
# anyone without `make` (Git Bash on Windows, for instance):
#
#     bash scripts/verify.sh

# make execs SHELL directly, so it must be a path to a binary rather than
# an `env` invocation with arguments.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help verify verify-fast install postgres test test-postgres \
        replay evidence wandb docker docker-up docker-down serve \
        dashboard clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the uv workspace (sdk/ + backend/)
	uv sync

postgres: ## Start just the Postgres container
	docker compose -f infra/docker-compose.yml up -d postgres

verify: ## Everything provable without credentials or paid model calls
	bash scripts/verify.sh

verify-fast: ## Same, minus the Docker image build and the dashboard build
	SKIP_DOCKER=1 SKIP_FRONTEND=1 bash scripts/verify.sh

test: ## Backend test suite
	cd backend && uv run pytest tests/ -q

test-postgres: ## Only the suites that require a real Postgres
	ONLY=postgres bash scripts/verify.sh

replay: ## Recording + exact recorded-execution replay
	ONLY=replay bash scripts/verify.sh

evidence: ## Regenerate docs/CAPABILITY_EVIDENCE.md from the artifacts
	uv run meta-harness report capability-evidence

wandb: ## Probe the optional W&B adapter offline
	ONLY=wandb bash scripts/verify.sh

docker: ## Build the backend image and prove the container serves
	bash scripts/docker_smoke.sh

docker-up: ## Bring up Postgres + the backend
	docker compose -f infra/docker-compose.yml up -d --build

docker-down: ## Stop the stack (add ARGS=-v to wipe the volume)
	docker compose -f infra/docker-compose.yml down $(ARGS)

serve: ## Run the API on :8000 (never `uvicorn` directly — see README)
	cd backend && uv run meta-harness serve --port 8000

dashboard: ## Run the Next.js dashboard in dev mode
	cd frontend/dashboard && npm run dev

clean: ## Remove caches and generated run artifacts (never committed evidence)
	rm -rf backend/.pytest_cache backend/.pytest-tmp
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
	rm -rf benchmark-results
