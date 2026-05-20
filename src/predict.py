"""
End-to-end prediction logic for NSL-KDD intrusion detection.

Loads trained artifacts (preprocessor, model, label encoder) and exposes
a single class — IntrusionDetector — that handles prediction and SHAP
explanation for any input connection.

Kept separate from the FastAPI layer so the prediction logic can be
unit-tested without spinning up a web server.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocessing import NSLKDDPreprocessor


# Default location for saved artifacts (relative to project root)
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


class IntrusionDetector:
    """
    Loaded model + preprocessor + label encoder, with predict and explain.

    Designed to be instantiated once at API startup; the loaded artifacts
    stay in memory and are reused across every request.
    """

    def __init__(self, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.preprocessor: NSLKDDPreprocessor = None
        self.model = None
        self.label_encoder = None
        self.explainer: shap.TreeExplainer = None

    def load(self) -> 'IntrusionDetector':
        """Load all artifacts from disk and initialise the SHAP explainer."""
        self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.pkl")
        self.model = joblib.load(self.artifacts_dir / "model.pkl")
        self.label_encoder = joblib.load(self.artifacts_dir / "label_encoder.pkl")

        # TreeExplainer is fast for tree-based models (milliseconds per prediction)
        self.explainer = shap.TreeExplainer(self.model)

        return self

    def predict(self, raw_features: dict) -> dict:
        """
        Predict the attack category for a single connection.

        Args:
            raw_features: Dictionary with raw NSL-KDD feature names as keys.
                          Must include 'service', 'protocol_type', 'flag', and
                          the numerical features. 'attack_type' and
                          'difficulty_level' are not required (and ignored if
                          provided).

        Returns:
            Dictionary with:
                - predicted_class: str (e.g. "DoS")
                - confidence: float (probability of the predicted class)
                - probabilities: dict mapping each class to its probability
                - top_contributing_features: list of (feature, signed_shap_value)
                  for the top 5 features driving this prediction
        """
        if self.model is None:
            raise RuntimeError("IntrusionDetector must be load()ed before predict().")

        # Convert single-row dict to DataFrame for the preprocessor
        df = pd.DataFrame([raw_features])

        # Apply preprocessing
        X = self.preprocessor.transform(df)

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[0]  # shape (5,)
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.label_encoder.classes_[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Map probabilities to class names
        probability_map = {
            cls: float(prob)
            for cls, prob in zip(self.label_encoder.classes_, probabilities)
        }

        # SHAP explanation for the predicted class
        top_features = self._explain(X, predicted_idx, raw_features)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_map,
            "top_contributing_features": top_features,
        }

    def _explain(
        self,
        X: pd.DataFrame,
        predicted_idx: int,
        raw_features: dict,
        top_n: int = 5,
    ) -> list:
        """
        Return top N features driving the predicted class, with human-readable
        service mapping where applicable.
        """
        # shap_values for multi-class is a list of arrays (one per class).
        # We want the explanation for the predicted class.
        shap_values = self.explainer.shap_values(X)

        # shap_values can be a list (older API) or 3D array (newer API).
        # Handle both.
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_idx][0]  # shape (n_features,)
        else:
            class_shap = shap_values[0, :, predicted_idx]

        feature_names = self.preprocessor.feature_columns

        # Sort features by absolute SHAP value (most influential first)
        order = np.argsort(np.abs(class_shap))[::-1][:top_n]

        result = []
        for idx in order:
            feature = feature_names[idx]
            shap_value = float(class_shap[idx])

            # Make service_encoded human-readable by surfacing the original service
            if feature == "service_encoded" and "service" in raw_features:
                feature_display = f"service ({raw_features['service']})"
            else:
                feature_display = feature

            result.append({
                "feature": feature_display,
                "shap_value": shap_value,
                "direction": "increased" if shap_value > 0 else "decreased",
            })

        return result