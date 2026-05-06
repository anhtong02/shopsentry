"""Shared pytest fixtures.

We use TestClient with mocked models/Feast so tests don't require
docker, MLflow, Redis, or a populated feature store. This is the
standard pattern for unit/integration testing FastAPI services.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.feast_client import DEFAULT_FEATURES, FeastClient
from api.main import app
from api.model_loader import LoadedModels


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
def client(mock_models, mock_feast_with_data) -> TestClient:
    """TestClient with happy-path mocks. Lifespan runs but we override state after."""
    with TestClient(app) as c:
        # Override after lifespan fires (lifespan runs on context entry)
        c.app.state.models = mock_models
        c.app.state.feast = mock_feast_with_data
        yield c


@pytest.fixture
def client_feast_down(mock_models, mock_feast_unavailable) -> TestClient:
    """Same as client but Feast lookups all return defaults."""
    with TestClient(app) as c:
        c.app.state.models = mock_models
        c.app.state.feast = mock_feast_unavailable
        yield c