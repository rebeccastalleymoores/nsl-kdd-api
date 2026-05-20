"""
FastAPI application for the NSL-KDD intrusion detection service.

Endpoints:
    GET  /          — basic service info
    GET  /health    — health check (used by deployment platforms)
    POST /predict   — submit network connection features, get a classification

Run locally:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import IntrusionDetector


# ----- Pydantic schemas -----
# These define the shape of requests and responses. FastAPI uses them for
# validation, documentation, and auto-generated interactive docs.


class ConnectionFeatures(BaseModel):
    """A single network connection record, as accepted by /predict."""

    # Categorical features (text)
    protocol_type: Literal["tcp", "udp", "icmp"]
    service: str = Field(..., description="Network service, e.g. 'http', 'smtp'")
    flag: str = Field(..., description="TCP connection state flag, e.g. 'SF', 'S0'")

    # Numerical features
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 0
    num_compromised: float = 0
    root_shell: float = 0
    num_root: float = 0
    num_file_creations: float = 0
    num_shells: float = 0
    num_access_files: float = 0
    is_guest_login: float = 0
    count: float = 0
    srv_count: float = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    srv_diff_host_rate: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate: float = 0
    dst_host_srv_rerror_rate: float = 0


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increased", "decreased"]


class PredictionResponse(BaseModel):
    predicted_class: Literal["DoS", "Normal", "Probe", "R2L", "U2R"]
    confidence: float
    probabilities: dict[str, float]
    top_contributing_features: list[FeatureContribution]


# ----- Lifespan: load model artifacts once at startup -----

detector: IntrusionDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, before the server starts handling requests."""
    global detector
    detector = IntrusionDetector().load()
    yield
    # No special teardown needed; Python will clean up.


# ----- FastAPI app -----

app = FastAPI(
    title="NSL-KDD Intrusion Detection API",
    description=(
        "Multi-class network intrusion detection. Submit raw connection features "
        "and receive a predicted attack category with SHAP-based explanation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "NSL-KDD Intrusion Detection API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Used by deployment platforms to verify the service is alive."""
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ConnectionFeatures) -> dict:
    """Classify a single network connection and return a SHAP-based explanation."""
    return detector.predict(features.model_dump())