"""ShopSentry prediction API. Models from MLflow, features from Feast, observability via Prometheus."""
import logging
import time
from contextlib import asynccontextmanager
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from typing import AsyncIterator
from api.feast_client import FeastClient
from api.model_loader import FEATURE_ORDER, LoadedModels, load_from_registry
from api.observability import (
    FEAST_FALLBACK_COUNT,
    MODEL_LOADED,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    router as observability_router,
)
from api.schemas import PredictRequest, PredictResponse, SessionFeatures
from typing import Awaitable, Callable
from fastapi import Response
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    without lifespan, we can load the model at wrong time for example:
    1. At import time: slows everything
    2. Inside the endpt: reloads model every request makes each request take more than 5 sec

    with lifespan, model loads 1 time at start up, lives in app.state.models
    """
    if os.getenv("SHOPSENTRY_TEST_MODE") == "1":
        logger.info("Test mode — skipping model/Feast load")
        yield
        return
    
    logger.info("Loading models from MLflow Registry...")
    app.state.models = load_from_registry()
    MODEL_LOADED.set(1)
    logger.info("Initializing Feast client...")
    app.state.feast = FeastClient()
    logger.info("Startup complete")
    yield # Code b4 yield runs at startup; after runs at shutdown
    MODEL_LOADED.set(0)
    logger.info("Shutting down")


app = FastAPI(
    title="ShopSentry Anomaly Detection API",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(observability_router)


"""
A middleware runs automatically on every request;it tracks every endpoint forever 
— /predict/anomaly, /health, /metrics
Without middleware, i'd have to manually add timing code to every endpoint:
def predict_anomaly(...):
    start = time.perf_counter()
    # ... 
    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(...).observe(latency)

def health(...):
    start = time.perf_counter()
    # ... actual logic ...
    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(...).observe(latency)
"""
@app.middleware("http")
async def track_metrics(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Wraps every request: times it, counts it, exposes via /metrics."""
    start = time.perf_counter()
    endpoint = request.url.path

    # Skip metrics on /metrics itself to avoid recursion noise
    if endpoint == "/metrics":
        return await call_next(request)

    response = await call_next(request)
    latency = time.perf_counter() - start

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "shopsentry", "version": "0.4.0"}


@app.post("/predict/anomaly", response_model=PredictResponse)
def predict_anomaly(request: PredictRequest, http_request: Request) -> PredictResponse:

    #http because we want to reach app state to get the models and feast
    t0 = time.perf_counter()
    models: LoadedModels = http_request.app.state.models
    feast: FeastClient = http_request.app.state.feast

    if request.features is not None:
        feature_dict = request.features.model_dump()
    elif request.session_id is not None:
        feature_dict, used_defaults = feast.get_features(request.session_id)
        if used_defaults:
            FEAST_FALLBACK_COUNT.inc()
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either session_id (lookup in Feast) or features (explicit)",
        )
    t_feast = time.perf_counter()

    try:
        validated = SessionFeatures(**feature_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid feature values: {e}")
    t_validate = time.perf_counter()

    feature_array = np.array(
        [[validated.model_dump()[name] for name in FEATURE_ORDER]],
        dtype=np.float64,
    )
    t_array = time.perf_counter()

    anomaly_score, is_anomaly = models.predict(feature_array)
    t_predict = time.perf_counter()

    PREDICTION_COUNT.labels(is_anomaly=str(is_anomaly).lower()).inc()

    logger.info(
        f"TIMING feast={(t_feast-t0)*1000:.1f}ms "
        f"validate={(t_validate-t_feast)*1000:.1f}ms "
        f"array={(t_array-t_validate)*1000:.1f}ms "
        f"predict={(t_predict-t_array)*1000:.1f}ms "
        f"total={(t_predict-t0)*1000:.1f}ms"
    )

    return PredictResponse(
        session_id=request.session_id,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        model_version=models.classifier_version,
    )