.PHONY: test lint fmt up down produce consume

test:
	python -m pytest tests/unit/test_event_schema.py -v
	python -m pytest tests/unit/test_agents.py -v

lint:
	ruff check .
	mypy .

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
