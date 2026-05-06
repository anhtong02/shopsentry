.PHONY: test lint fmt up down spark attack normal inspect apply clean

test:
	python -m pytest tests/unit/ tests/integration/ -v

lint:
	ruff check .
	mypy simulator pipeline tests api models --install-types --non-interactive

fmt:
	ruff format .

up:
	docker-compose up -d

down:
	docker-compose down

# --- NEW WEEK 3 COMMANDS ---

spark:
	python pipeline/spark_streaming.py

attack:
	python -m simulator.run --scenario bot_attack

normal:
	python -m simulator.run --scenario normal_day

mixed:
	python -m simulator.run --scenario mixed
mixedL:
	python -m simulator.run --scenario mixed_large

inspect:
	python tests/integration/test_data_quality.py

apply:
	cd feature_repo/feature_repo && feast apply

activate:
	.venv\Scripts\activate

mlflow:
	mlflow server --host 0.0.0.0 --port 5000
# ---------------------------


iso:
	python -m models.anomaly.isolation_forest
clean: 
	-powershell -Command "Remove-Item -Recurse -Force __pycache__, .pytest_cache, .mypy_cache, .ruff_cache -ErrorAction Ignore"

#-----week 4-5-----

api:
	uvicorn api.main:app --port 8000 --workers 4

pipeline:
	python -m models.run_pipeline

load-test:
	locust -f load_tests/locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 20 --run-time 60s --headless