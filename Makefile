.PHONY: test lint fmt up down spark attack normal inspect apply clean

test:
	python -m pytest tests/unit/ -v

lint:
	ruff check .
	mypy simulator pipeline tests --install-types --non-interactive

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

inspect:
	python tests/integration/test_data_quality.py

apply:
	cd feature_repo/feature_repo && feast apply

activate:
	.venv\Scripts\activate
# ---------------------------

clean: 
	-powershell -Command "Remove-Item -Recurse -Force __pycache__, .pytest_cache, .mypy_cache, .ruff_cache -ErrorAction Ignore"