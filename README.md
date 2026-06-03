# nsl-kdd-api

Containerised FastAPI service deploying an XGBoost multi-class intrusion detection model (NSL-KDD) with SHAP-based explanations for each prediction.

**Live demo:** https://nsl-kdd-api-m7an.onrender.com

**API docs:** https://nsl-kdd-api-m7an.onrender.com/docs

This project demonstrates end-to-end ML deployment engineering: modular Python, a tested REST API, Docker containerisation, and cloud deployment. The model itself was developed and evaluated in the accompanying classification project — see that repo for methodology, model selection, and performance analysis: https://github.com/rebeccastalleymoores/nsl-kdd-intrusion-detection-classification

---

## Project structure
nsl-kdd-api/
├── api/main.py          # FastAPI app — /predict and /health endpoints
├── src/preprocessing.py # NSLKDDPreprocessor (fit/transform pattern)
├── src/predict.py       # IntrusionDetector class with SHAP explainability
├── artifacts/           # Trained model, preprocessor, and label encoder
├── templates/index.html # Interactive frontend
├── tests/               # 7 pytest tests covering prediction and API behaviour
├── train.py             # Reproducible training pipeline
└── Dockerfile           # Python 3.12-slim image

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive frontend |
| `/predict` | POST | Returns predicted class, confidence, class probabilities, and top 5 SHAP features |
| `/health` | GET | Returns service status |
| `/docs` | GET | Interactive Swagger UI |

---

## Deployment

The service is deployed on Render using Docker. On each push to `main`, Render rebuilds and redeploys automatically.

To run locally:

```bash
docker build -t nsl-kdd-api .
docker run -p 8000:8000 nsl-kdd-api
```

Then visit http://127.0.0.1:8000

---

## Operational considerations

This is a portfolio project demonstrating ML deployment engineering, not a production-ready intrusion detection system. The model achieves macro F1 of 0.641 on the NSL-KDD test set, with minority class recall (R2L 30%, U2R 18%) that would be insufficient for operational use. These limitations reflect the inherent difficulty of the NSL-KDD benchmark — particularly the test set containing attack subtypes absent from training — rather than the engineering quality of the deployment.

A production intrusion detection system would require ensemble approaches (e.g., a supervised classifier paired with unsupervised anomaly detection for zero-day coverage), continuous retraining infrastructure, and integration with downstream SOC tooling. This project demonstrates the foundational deployment work that such a system would build upon.

**Deployment note:** The API is hosted on Render's free tier, which spins down the service after 15 minutes of inactivity. The first request after a period of inactivity will take 30–60 seconds to wake up — this is a free tier limitation, not a bug. Subsequent requests are fast. A production deployment would use a paid tier to keep the service warm.