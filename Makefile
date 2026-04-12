.PHONY: test lint fmt up down produce consume

test:
	python -m pytest tests/unit/ -v
lint:
	ruff check .
	mypy simulator pipeline tests
fmt:
	ruff format .

up:
	docker-compose up -d

down:
	docker-compose down

produce:
	python -m simulator.producer

consume:
	python -m simulator.consumer

clean: 
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
