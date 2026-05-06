import logging
import time
import redis
from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest
)

logger = logging.getLogger(__name__)
router = APIRouter()

#counter is to increase the number (and only goes up)
REQUEST_COUNT = Counter(
    "shopsentry_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

#histogram is bucketed distributions. Every observation falls into a bucket.
#lets u compute the percentiles at query time. Too many buckets might be expensive,
#too few can have imprecise percentiles
REQUEST_LATENCY = Histogram(
    "shopsentry_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_COUNT = Counter(
    "shopsentry_predictions_total",
    "Predictions by outcome",
    ["is_anomaly"],
)

FEAST_FALLBACK_COUNT = Counter(
    "shopsentry_feast_fallback_total",
    "Times graceful degradation kicked in",
)

# gauge can go up or down
MODEL_LOADED = Gauge(
    "shopsentry_model_loaded",
    "1 if model is loaded, 0 otherwise",
)

#@router.get(...) means ti register the function as an endpoint on this router.

@router.get("/health")
def health(request: Request) -> Response:
    checks = {}
    overall_ok = True

    # 1. Model loaded?
    try:
        models = request.app.state.models
        checks["model"] = {
            "ok": models is not None,
            "version": models.classifier_version if models else None,
        }
    except AttributeError:
        checks["model"] = {"ok": False, "error": "not loaded"}
        overall_ok = False

    # 2. Redis reachable??
    try:
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        checks["redis"] = {
            "ok": True
        }
    except Exception as e:
        checks["redis"] = {
            "ok": False,
            "error": str(e)
        }

    # 3. Feast client initialized?
    try:
        feast = request.app.state.feast
        checks["feast"] = {"ok": feast._store is not None}
    except AttributeError:
        checks["feast"] = {"ok": False, "error": "not initialized"}
 
    status_code = 200 if overall_ok else 503
    body = {"status": "ok" if overall_ok else "degraded", "checks": checks}
 
    import json
    return Response(
        content=json.dumps(body),
        status_code=status_code,
        media_type="application/json",
    )

@router.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)