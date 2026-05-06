"""Shared pytest fixtures.

We use TestClient with mocked models/Feast so tests don't require
docker, MLflow, Redis, or a populated feature store. This is the
standard pattern for unit/integration testing FastAPI services.
"""
from unittest.mock import MagicMock
import pytest
from api.feast_client import DEFAULT_FEATURES, FeastClient
from api.main import app
from api.model_loader import LoadedModels
from typing import Generator
from fastapi.testclient import TestClient

@pytest.fixture
def mock_models() -> LoadedModels:
    """A LoadedModels with predictable fake predict() output."""
    models = MagicMock(spec=LoadedModels)
    models.classifier_version = "test_classifier:v1"
    # Default: returns (0.1, False) — looks normal
    models.predict.return_value = (0.1, False)
    return models


@pytest.fixture
def mock_feast_with_data() -> FeastClient:
    """Feast client that returns valid features (simulates Redis hit)."""
    client = MagicMock(spec=FeastClient)
    client.get_features.return_value = (dict(DEFAULT_FEATURES), False)
    return client


@pytest.fixture
def mock_feast_unavailable() -> FeastClient:
    """Feast client that always falls back (simulates Redis down)."""
    client = MagicMock(spec=FeastClient)
    client.get_features.return_value = (dict(DEFAULT_FEATURES), True)
    return client


@pytest.fixture
def client(mock_models: LoadedModels, mock_feast_with_data: FeastClient) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        c.app.state.models = mock_models  # type: ignore[attr-defined]
        c.app.state.feast = mock_feast_with_data  # type: ignore[attr-defined]
        yield c


@pytest.fixture
def client_feast_down(mock_models: LoadedModels, mock_feast_unavailable: FeastClient) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        c.app.state.models = mock_models  # type: ignore[attr-defined]
        c.app.state.feast = mock_feast_unavailable  # type: ignore[attr-defined]
        yield c