SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install project dependencies
	./bin/install

.PHONY: test
test:	## Run project tests
	./bin/test

.PHONY: lint
lint:  ## Lint project code
	./bin/lint

.PHONY: format
format:  ## Format project code
	./bin/format

.PHONY: run
run: ## Run project
	./bin/run

.PHONY: depends
depends: ## Add poetry dependencies from requirements.txt.
	./bin/depends
