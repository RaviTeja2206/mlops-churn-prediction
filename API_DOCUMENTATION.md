# API Documentation

The Customer Churn Prediction API provides REST endpoints for real-time churn prediction. When running locally or in Docker, interactive docs (Swagger UI) are available at **http://localhost:8000/docs**.

## Base URL

- **Local/Docker:** `http://localhost:8000`

## Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | Detailed health status and model load status |

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "timestamp": "2025-02-08T10:00:00"
}
```

### Single Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict churn for a single customer |

**Required fields:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`  
**Optional fields:** `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and other one-hot encoded features (default to 0)

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "SeniorCitizen": 0}'
```

**Response:**
```json
{
  "churn_prediction": "Will Not Churn",
  "churn_probability": 0.2341,
  "confidence": "High",
  "recommendation": "Low risk. Customer likely to stay. Focus on upselling opportunities.",
  "timestamp": "2025-02-08T10:00:00"
}
```

### Batch Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/batch` | Predict churn for multiple customers |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "SeniorCitizen": 0}]'
```

### Model Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/model/info` | Get model metadata (type, features, metrics) |

**Response:**
```json
{
  "model_type": "XGBoost Classifier",
  "training_platform": "AWS SageMaker",
  "features_count": 28,
  "accuracy": 0.816,
  "f1_score": 0.611
}
```

## Interactive Docs

When the API is running:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
