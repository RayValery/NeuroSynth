# Everything derived from the per-subject results in results/per_subject/.
#
#   make install   install pinned dependencies and the neurosynth package
#   make figures   rebuild all figures and summary tables (seconds, CPU only)
#   make test      leakage-invariant tests
#   make collect   copy freshly trained result files from models/checkpoints/ into results/per_subject/
#   make all       figures + test
#
# After re-running any training arm:  make collect figures

# Uses the project virtualenv if present (no need to activate it), otherwise python3.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: all install figures test collect

all: figures test

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q

collect:
	cp models/checkpoints/per_subject_results_*.json results/per_subject/
