.PHONY: conda format style types black test link check notebooks
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.10
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = jaxsw

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Formatting
black:  ## Format code in-place using black.
		black ${PKGROOT}/ -l 79 .

isort:  ## Format imports in-place using isort.
		isort ${PKGROOT}/ 

format: ## Code styling - black, isort
		black ${PKGROOT}/
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		isort ${PKGROOT}/
		@printf "\033[1;34misort passes!\033[0m\n\n"
		ruff ${PKGROOT}/ --fix --ignore=I001,E402,E721,E731,E741,F722,E501
		@printf "\033[1;34misort passes!\033[0m\n\n"

##@ Testing
test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v jaxsw tests
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"

precommit: ## Run recommit
		@printf "\033[1;34mRunning precommit routine...\033[0m\n\n"
		pre-commit run --all-files
		@printf "\033[1;34mFinished!\033[0m\n\n"

##@ JupyterBook
jb_build: ## Build Jupyterbook
	rm -rf docs/_build/
	jupyter-book build jbook --all

jb_clean: ## Clean JupyterBook
	jupyter-book clean jbook

jb_publish: ## Clean JupyterBook
	ghp-import -n -p -f _build/html