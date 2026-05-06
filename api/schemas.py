from pydantic import BaseModel, Field


class SessionFeatures(BaseModel):
    events_per_minute: float = Field(..., ge=0)
    unique_pages_visited: float = Field(..., ge=0)
    avg_time_between_events: float = Field(..., ge=0)
    cart_to_purchase_ratio: float = Field(..., ge=0)
    session_duration_seconds: float = Field(..., ge=0)
    event_type_diversity: float = Field(..., ge=0)
    has_payment: int = Field(..., ge=0, le=1)
    signup_to_purchase_speed: float = Field(..., ge=0)
    page_revisit_ratio: float = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
    session_id: str | None = None
    features: SessionFeatures | None = None


class PredictResponse(BaseModel):
    session_id: str | None
    anomaly_score: float
    is_anomaly: bool
    model_version: str
