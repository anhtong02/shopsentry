"""Integration tests for the prediction API.

These use FastAPI's TestClient — same code path as production but in-process.
Models and Feast are mocked so tests run anywhere (no docker required).
"""
import pytest


# ============================================================
# Happy path
# ============================================================

def test_root_returns_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "shopsentry"


def test_predict_with_explicit_features_returns_score(client):
    r = client.post(
        "/predict/anomaly",
        json={
            "session_id": "test_explicit",
            "features": {
                "events_per_minute": 5.0,
                "unique_pages_visited": 3.0,
                "avg_time_between_events": 15.0,
                "cart_to_purchase_ratio": 0.5,
                "session_duration_seconds": 120.0,
                "event_type_diversity": 4,
                "has_payment": 0,
                "signup_to_purchase_speed": 0.0,
                "page_revisit_ratio": 0.1,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "test_explicit"
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert body["model_version"] == "test_classifier:v1"


def test_predict_with_session_id_only_does_feast_lookup(client, mock_feast_with_data):
    r = client.post("/predict/anomaly", json={"session_id": "looked_up_session"})
    assert r.status_code == 200
    # Verify Feast was actually called
    mock_feast_with_data.get_features.assert_called_once_with("looked_up_session")


def test_predict_returns_anomaly_when_model_says_so(client, mock_models):
    mock_models.predict.return_value = (0.95, True)
    r = client.post(
        "/predict/anomaly",
        json={
            "session_id": "bot_session",
            "features": {
                "events_per_minute": 80.0,
                "unique_pages_visited": 1.0,
                "avg_time_between_events": 0.8,
                "cart_to_purchase_ratio": 0.0,
                "session_duration_seconds": 200.0,
                "event_type_diversity": 2,
                "has_payment": 0,
                "signup_to_purchase_speed": 0.0,
                "page_revisit_ratio": 0.0,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_anomaly"] is True
    assert body["anomaly_score"] == 0.95


# ============================================================
# Bad input — Pydantic validation
# ============================================================

def test_predict_rejects_negative_feature(client):
    r = client.post(
        "/predict/anomaly",
        json={"features": {"events_per_minute": -5}},  # ge=0 should reject
    )
    assert r.status_code == 422


def test_predict_rejects_missing_required_field(client):
    r = client.post(
        "/predict/anomaly",
        json={"features": {"events_per_minute": 5}},  # other fields missing
    )
    assert r.status_code == 422


def test_predict_rejects_invalid_has_payment_value(client):
    r = client.post(
        "/predict/anomaly",
        json={
            "features": {
                "events_per_minute": 5,
                "unique_pages_visited": 3,
                "avg_time_between_events": 15,
                "cart_to_purchase_ratio": 0.5,
                "session_duration_seconds": 120,
                "event_type_diversity": 4,
                "has_payment": 5,  # should be 0 or 1
                "signup_to_purchase_speed": 0,
                "page_revisit_ratio": 0.1,
            },
        },
    )
    assert r.status_code == 422


def test_predict_rejects_empty_request(client):
    r = client.post("/predict/anomaly", json={})
    # No session_id, no features → 400
    assert r.status_code == 400


# ============================================================
# Graceful degradation — Feast/Redis down
# ============================================================

def test_predict_works_when_feast_returns_defaults(client_feast_down):
    """When Feast can't find a session, defaults are used and prediction still works."""
    r = client_feast_down.post(
        "/predict/anomaly",
        json={"session_id": "session_not_in_redis"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "session_not_in_redis"
    assert "anomaly_score" in body


# ============================================================
# Observability endpoints
# ============================================================

def test_health_returns_status(client):
    r = client.get("/health")
    # 200 = ok, 503 = degraded — both acceptable depending on Redis state
    assert r.status_code in (200, 503)
    body = r.json()
    assert "status" in body
    assert "checks" in body
    assert "model" in body["checks"]


def test_metrics_endpoint_returns_prometheus_format(client):
    # Hit the API once first so counters are non-zero
    client.post(
        "/predict/anomaly",
        json={
            "session_id": "metrics_test",
            "features": {
                "events_per_minute": 5,
                "unique_pages_visited": 3,
                "avg_time_between_events": 15,
                "cart_to_purchase_ratio": 0.5,
                "session_duration_seconds": 120,
                "event_type_diversity": 4,
                "has_payment": 0,
                "signup_to_purchase_speed": 0,
                "page_revisit_ratio": 0.1,
            },
        },
    )

    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus format is plain text with HELP/TYPE comments
    text = r.text
    assert "shopsentry_requests_total" in text
    assert "shopsentry_request_latency_seconds" in text
    assert "shopsentry_model_loaded" in text