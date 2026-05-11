# ShopSentry

## What is it
It's a real time anomaly detector for simulated e-commerce event streams. It detects bot traffic and fraud rings using a full ML pipeline: simulated traffic -> streaming feature engineering -> multi model evaluation -> serve the best model with observability

## What this project shows
- **End-to-end ML system, not just a notebook!** From Event simulation, streaming feature pipeline, to model training with experiment tracking, model registry, real-time serving, monitoring, integration tests, load testing.
- **With pseudo-labeling and XGBoost for model** Compares 5 labeling strategies (Isolation Forest, Autoencoder, Heuristics, Ensemble, Oracle) and evaluates the cost of not having ground-truth labels.
- **Production-grade serving.** FastAPI with Pydantic validation, MLflow Registry version pinning, Feast online lookup with graceful degradation, Prometheus metrics, integration tests that run anywhere without infrastructure.
- **Measured performance under load.** Distributed Locust load testing across two machines, with documented latency-vs-concurrency curves.

## Architecture
 
```
┌──────────────┐    Kafka     ┌──────────┐   Streaming   ┌─────────────────┐
│  Simulator   │─────────────▶│ Redpanda │──────────────▶│  Spark          │
│ (4 agent     │              │          │               │  feature engine │
│  types)      │              └──────────┘               └────────┬────────┘
└──────────────┘                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │     Feast    │
                                                          │ (Redis +     │
                                                          │  Parquet)    │
                                                          └──────┬───────┘
                                                                 │
                              ┌──────────────────────────────────┤
                              │                                  │
                              ▼                                  ▼
                       ┌─────────────┐                  ┌────────────────┐
                       │ Training    │                  │  FastAPI       │
                       │ Pipeline    │   MLflow         │  /predict/     │
                       │ (5 models)  │──Registry───────▶│  anomaly       │
                       └─────────────┘                  │                │
                                                        │  Prometheus    │
                                                        │  /metrics      │
                                                        └────────────────┘
```

## Results
 
### Model comparison
 
| Labeler | Pseudo-label F1 | XGBoost test F1 | Bot recall | Fraud recall |
|---|---|---|---|---|
| Isolation Forest | 0.40 | 0.49 | 100% | 27% |
| Autoencoder (normal-only) | 0.97 | 0.84 | 100% | 100% |
| Heuristic | 1.00 | 1.00 | 100% | 100% |
| Ensemble | 0.95 | 0.93 | 100% | 100% |
| Oracle (true labels) | 1.00 | 1.00 | 100% | 100% |
 
Autoencoder selected as production model — see [docs/model_comparison.md](docs/model_comparison.md) for full analysis.
 
### API latency under load
 
Distributed test: Locust on macOS hitting FastAPI on Windows over LAN, 16 uvicorn workers, 0 failures across 30K+ requests.
 
| Concurrent users | p50 | p95 | p99 | RPS |
|---|---|---|---|---|
| 20 | **32 ms** | 88 ms | 160 ms | ~190 |
| 30 | 60 ms | 150 ms | 240 ms | ~240 |
| 50 | 83 ms | 190 ms | 260 ms | ~295 |
 
Direct measurement (no load): ~3 ms p50.
 
When 20 or fewer users hit the API at same time, it is < 50ms p50. When more than 30 users hit at once, the wait time becomes bigger than actual work time. Kubernetes, and AWS EKS can help with this problem in real production.
 
## Quick Start
### Prerequisites
- Docker Desktop running
- Python 3.12+
- 4GB free RAM

### 1. Bring up infrastructure
```bash
.venv\Scripts\activate
docker compose up -d
# Starts: MLflow, Redis, Redpanda, Spark
```
 
### 2. Generate training data
```bash
# Terminal 1: streaming feature engine
python -m pipeline.spark_streaming

# Terminal 2: traffic simulator
python -m simulator.run --scenario mixed_large
```
 
Wait 2 minutes for Spark to drain.
 
### 3. Train and register models
```bash
python -m models.run_pipeline
```
 
Open MLflow UI at http://localhost:5000 → register the autoencoder run as `shopsentry_classifier` and `shopsentry_scaler` (v1).
 
### 4. Serve predictions
```bash
uvicorn api.main:app --port 8000 --workers 8
```
 
API docs at http://localhost:8000/docs
 
### 5. Test
```bash
make test          # 30+ unit and integration tests
make load-test     # Locust against local API
```
 
## Folders tree
 
```
shopsentry/
├── api/                    # FastAPI service (predict, health, metrics)
│   ├── main.py
│   ├── model_loader.py     # MLflow Registry client
│   ├── feast_client.py     # Online feature lookup with fallback
│   ├── observability.py    # Prometheus metrics + /health
│   └── schemas.py          # Pydantic request/response
├── models/                 # ML pipeline
│   ├── pseudo_labelers/    # 4 labeling strategies + ensemble
│   ├── classifiers/        # XGBoost wrapper
│   └── run_pipeline.py     # Orchestrator
├── pipeline/               # Spark streaming feature engine
├── simulator/              # 4 agent types (normal, bot, fraud ring, churning)
├── feature_repo/           # Feast definitions
├── load_tests/             # Locust scenarios
├── tests/
│   ├── unit/
│   └── integration/        # Mocked, runs without infra
└── docs/
    └── model_comparison.md # Full model analysis
```
 
## Tech stack
 
- **Streaming:** Redpanda (Kafka), Spark Structured Streaming
- **Feature store:** Feast (Redis online + Parquet offline)
- **Models:** XGBoost, TensorFlow autoencoder, scikit-learn (IsolationForest, StandardScaler)
- **Experiment tracking:** MLflow (server, registry, model logging)
- **Serving:** FastAPI + Uvicorn workers
- **Validation:** Pydantic v2
- **Observability:** Prometheus client (counters, histograms, gauges)
- **Testing:** pytest + TestClient + mocks
- **Load testing:** Locust (distributed master/worker)
- **Quality gates:** ruff, mypy, GitHub Actions CI

## Limitations

This is **phase 1**. The simulator's agents are obvious, which makes the detection problem easier than real life fraud based on the fact that i created them like this:
 
- Bots make 50-120 events/min consistently
- Fraud rings acts immediately on session 1
- Only session-level features → fraud rings detected per-session, not as groups

**Phase 2 (planned):** introduce evasive agents — throttled bots, fraud rings that browse before striking, account aging, IP rotation with realistic geo distribution. This should make F1 scores of every model drops. Then build phase 3 fixes (heuristics, cross-session features, ensemble re-weighting).
 
That phase 2 → phase 3 arc is the project I actually want to build. Phase 1 is the platform that makes that experiment possible.
 
## Read more
 
- [docs/model_comparison.md](docs/model_comparison.md) — Full model evaluation, why autoencoder won, what each labeler is good and bad at
- API docs auto-generated at `/docs` when running uvicorn
---
 
Built by Anh Tong | [\[LinkedIn\]](https://www.linkedin.com/in/anhtongxt/) | anhtongxt@gmail.com