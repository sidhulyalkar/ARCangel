.PHONY: test preflight notebooks check public-smoke

test:
	python -m pytest -q

preflight:
	PYTHONPATH=src python scripts/preflight.py

notebooks:
	python scripts/build_kaggle_notebooks.py

check: notebooks preflight test
	python -m compileall -q src scripts

public-smoke:
	@test -n "$(ENV_DIR)" || (echo "Set ENV_DIR=/path/to/environment_files" && exit 2)
	PYTHONPATH=src python scripts/run_public.py --env-dir "$(ENV_DIR)" --policy structural --games vc33 --max-actions 20 --workers 1 --output artifacts/smoke.json
