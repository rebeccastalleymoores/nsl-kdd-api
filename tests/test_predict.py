"""
Tests for the NSL-KDD intrusion detection pipeline.

Run from project root:
    pytest

Covers:
    - Preprocessing: fit/transform shape and train/serve consistency
    - IntrusionDetector: load and predict
    - API endpoints: /health, /predict (valid + invalid requests)
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.preprocessing import NSLKDDPreprocessor, COLUMN_NAMES
from src.predict import IntrusionDetector
from api.main import app


# ---------- Fixtures ----------
# A pytest fixture is a reusable object built once and injected into tests
# that ask for it (by listing it as a parameter). Keeps tests clean.


@pytest.fixture(scope="module")
def training_sample() -> pd.DataFrame:
    """A small subset of the real training data, for preprocessing tests."""
    df = pd.read_csv("data/KDDTrain+.txt", names=COLUMN_NAMES, header=None)
    return df.head(1000).copy()


@pytest.fixture(scope="module")
def detector() -> IntrusionDetector:
    """Loaded IntrusionDetector. Loaded once, reused across tests."""
    return IntrusionDetector().load()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A test client that calls the FastAPI app without needing a running server."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_connection() -> dict:
    """A known-Normal HTTP connection payload."""
    return {
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "duration": 0,
        "src_bytes": 232,
        "dst_bytes": 8153,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "is_guest_login": 0,
        "count": 5,
        "srv_count": 5,
        "serror_rate": 0.2,
        "srv_serror_rate": 0.2,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 30,
        "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.03,
        "dst_host_srv_diff_host_rate": 0.04,
        "dst_host_serror_rate": 0.03,
        "dst_host_srv_serror_rate": 0.01,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.01,
    }


# ---------- Preprocessing tests ----------


def test_preprocessor_fit_transform_shape(training_sample):
    """After fitting and transforming, we get a model-ready feature matrix."""
    preprocessor = NSLKDDPreprocessor()
    preprocessor.fit(training_sample)
    X = preprocessor.transform(training_sample)

    assert X.shape[0] == len(training_sample)
    assert X.shape[1] == len(preprocessor.feature_columns)
    assert len(preprocessor.feature_columns) > 0


def test_preprocessor_single_row_matches_bulk(training_sample):
    """A single-row transform must produce identical columns to a bulk transform."""
    preprocessor = NSLKDDPreprocessor()
    preprocessor.fit(training_sample)

    X_bulk = preprocessor.transform(training_sample)
    X_single = preprocessor.transform(training_sample.iloc[[0]])

    # This is the train/serve consistency check — critical for the API.
    assert X_single.columns.tolist() == X_bulk.columns.tolist()
    assert X_single.shape == (1, X_bulk.shape[1])


# ---------- Detector tests ----------


def test_detector_loads(detector):
    """Artifacts load and all internal state is populated."""
    assert detector.model is not None
    assert detector.preprocessor is not None
    assert detector.label_encoder is not None
    assert detector.explainer is not None


def test_detector_predict_returns_expected_structure(detector, sample_connection):
    """A prediction includes all required fields with sensible types."""
    result = detector.predict(sample_connection)

    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "top_contributing_features" in result

    assert result["predicted_class"] in {"DoS", "Normal", "Probe", "R2L", "U2R"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 5
    assert len(result["top_contributing_features"]) == 5


# ---------- API endpoint tests ----------


def test_health_endpoint(client):
    """/health returns 200 and reports the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint_valid_request(client, sample_connection):
    """A valid POST to /predict returns 200 and a structurally valid response."""
    response = client.post("/predict", json=sample_connection)
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in {"DoS", "Normal", "Probe", "R2L", "U2R"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_contributing_features"]) == 5


def test_predict_endpoint_rejects_invalid_protocol(client, sample_connection):
    """Invalid protocol_type should be rejected by Pydantic validation."""
    bad = sample_connection.copy()
    bad["protocol_type"] = "smtp"  # not one of tcp/udp/icmp

    response = client.post("/predict", json=bad)
    assert response.status_code == 422  # FastAPI's validation error code