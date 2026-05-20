# nsl-kdd-api
Containerised FastAPI service deploying an XGBoost multi-class intrusion detection model (NSL-KDD) with SHAP-based explanations for each prediction

This API deploys the model developed and evaluated in the classification project (https://github.com/rebeccastalleymoores/nsl-kdd-intrusion-detection-classification). See that repo for methodology, model selection, and performance analysis.

Operational considerations
This is a portfolio project demonstrating ML deployment engineering, not a production-ready intrusion detection system. The underlying model achieves macro F1 of 0.641 on the NSL-KDD test set, with minority class recall (R2L 30%, U2R 18%) that would be insufficient for operational use. These limitations reflect the inherent difficulty of the NSL-KDD benchmark — particularly the test set containing attack subtypes absent from training — rather than the engineering quality of the deployment.
A production intrusion detection system would require ensemble approaches (e.g., supervised classifier paired with unsupervised anomaly detection for zero-day coverage), continuous retraining infrastructure, and integration with downstream SOC tooling. This project demonstrates the foundational deployment work that such a system would build upon.
