# MLOps Churn Prediction Pipeline

> **Production-ready MLOps pipeline** for customer churn prediction with automated drift monitoring, model retraining, and deployment.

**Note:** Keep secrets (API keys, passwords, SMTP credentials) in `.env` only—never commit them. Use `.env.example` as a template.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 What This Project Does

This project demonstrates a **complete MLOps pipeline** that:
- **Predicts customer churn** using XGBoost machine learning
- **Monitors data drift** to detect when the model needs retraining
- **Provides a production API** for real-time predictions
- **Tracks experiments** and manages model versions
- **Deploys with Docker** for consistent environments
- **Includes CI/CD** for automated testing and deployment

**Perfect for**: Data scientists, ML engineers, and anyone learning MLOps best practices.

## 🚀 Quick Demo (2 minutes)

```bash
# Clone and run the complete pipeline
git clone https://github.com/RaviTeja2206/mlops-churn-prediction.git
cd mlops-churn-prediction

# One-command deployment
./docker/manage.sh build && ./docker/manage.sh run

# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "SeniorCitizen": 0}'

# View interactive API docs
open http://localhost:8000/docs
```

**Expected Output**: `{"churn_prediction": "Will Not Churn", "churn_probability": 0.23, "confidence": "High"}`

## 🏗️ How It Works

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Preprocessing  │───▶│  Model Training │
│   (CSV/S3)      │    │   Pipeline      │    │   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Drift Monitoring│◀───│   Predictions   │◀───│   FastAPI       │
│  (Evidently)    │    │     (Logs)      │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Dashboard     │    │   Docker        │
                       │  (Streamlit)    │    │  Container      │
                       └─────────────────┘    └─────────────────┘
```

**The Pipeline:**
1. **Data Processing**: Clean and prepare telecom customer data (7,043 customers, 20 features)
2. **Model Training**: Train XGBoost classifier with MLflow tracking
3. **API Deployment**: Serve predictions via FastAPI
4. **Monitoring**: Track data drift and model performance
5. **Retraining**: Automatically retrain when drift detected

## 📊 Model Performance

The model is trained on telecom customer data and uses **XGBoost** for classification.

**Why XGBoost?**
- Excellent performance on structured/tabular data
- Built-in handling of missing values
- Feature importance insights
- Robust against overfitting
- Industry-standard for churn prediction

**Training Process:**
- Compares Logistic Regression, Random Forest, and XGBoost
- Uses MLflow for experiment tracking
- Selects best model based on validation metrics
- Includes comprehensive evaluation reports

## 🛠️ Technology Stack

**ML & Data**: Python, Scikit-learn, XGBoost, Pandas, NumPy  
**MLOps**: MLflow, Evidently AI, Docker, FastAPI  
**Cloud**: AWS SageMaker, S3  
**DevOps**: GitHub Actions, Docker Compose  
**Monitoring**: Logging, Health Checks, Drift Detection  

## 🎯 Key Features

### **🤖 Smart Churn Prediction**
- XGBoost model chosen for its superior performance on tabular data
- Handles 20 customer features (tenure, charges, services, demographics)
- Returns probability scores and confidence levels with business recommendations

### **📊 Automated Monitoring**
- Real-time data drift detection with Evidently AI
- Email alerts when model performance degrades
- HTML reports with detailed analysis

### **🚀 Production-Ready API**
- FastAPI with automatic OpenAPI documentation
- Input validation and comprehensive error handling
- Health checks and monitoring endpoints

### **🔄 MLOps Pipeline**
- MLflow experiment tracking and model registry
- Automated retraining triggers
- CI/CD pipeline with GitHub Actions

### **🐳 Easy Deployment**
- Docker containerization with one-command setup
- Multi-stage builds for production optimization
- Horizontal scaling support

## 🚀 Getting Started

### Prerequisites
- **Docker Desktop** - [Install Docker Desktop](https://docs.docker.com/get-docker/)
  - **Mac (Intel/Apple Silicon)**: Docker Desktop for Mac (auto-detects architecture)
  - **Windows**: Docker Desktop for Windows (requires WSL2)
  - **Linux**: Docker Engine or Docker Desktop
- **OR Python 3.11+** - [Install Python](https://python.org)
- **Git** - [Install Git](https://git-scm.com/)

**Important**: Make sure Docker Desktop is running before executing Docker commands.

### Option 1: Docker (Recommended - Works on All Platforms)

#### Quick Start (Universal Commands)
```bash
# Clone the repository
git clone https://github.com/RaviTeja2206/mlops-churn-prediction.git
cd mlops-churn-prediction

# Build (Docker auto-detects your platform)
docker build -t churn-prediction-api:latest .

# Run with volume mount for logs
docker run -d --name churn-api -p 8000:8000 -v "$(pwd)/logs:/app/logs" churn-prediction-api:latest

# Test the API
python api/test_api.py
# OR
curl http://localhost:8000/health

# View API docs: http://localhost:8000/docs
```

#### Platform-Specific Commands

**Mac (Intel & Apple Silicon):**
```bash
# Use the management script (auto-detects architecture)
./docker/manage.sh build
./docker/manage.sh run
./docker/manage.sh test
```

**Windows (PowerShell/CMD):**
```bash
# Build and run
docker build -t churn-prediction-api:latest .
docker run -d --name churn-api -p 8000:8000 -v "%cd%/logs:/app/logs" churn-prediction-api:latest

# Test
python api/test_api.py
```

**Linux:**
```bash
# Same as Mac
./docker/manage.sh build
./docker/manage.sh run
./docker/manage.sh test
```

### Option 2: Local Development
```bash
# Clone and setup
git clone https://github.com/RaviTeja2206/mlops-churn-prediction.git
cd mlops-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Start the API
cd api && uvicorn main:app --reload

# In another terminal, run tests
pytest tests/
```

### ✅ Verify Installation (All Platforms)

After setup, test these endpoints to confirm everything works:

```bash
# 1. Health check
curl http://localhost:8000/health
# Expected: {"status": "healthy", "model": "loaded"}

# 2. Simple prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "SeniorCitizen": 0}'
# Expected: JSON with churn_prediction and probability

# 3. API documentation (open in browser)
# http://localhost:8000/docs
```

**If any test fails:**
1. Check Docker Desktop is running
2. Verify container is running: `docker ps`
3. Check logs: `docker logs churn-api`
4. Try different port: `docker run -p 8001:8000 ...`

### 🔧 Optional: Advanced Features

#### AWS SageMaker (Cloud Training)
For cloud-based model training with cost optimization:

1. **Install AWS CLI**: `pip install awscli`
2. **Configure credentials**: `aws configure`
3. **Update IAM role** in `sagemaker/train_sagemaker.py`:
   ```python
   role_arn = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
   ```
4. **Run cloud training**: `python sagemaker/train_sagemaker.py`

#### Email Alerts (Drift Monitoring)
For automated drift detection emails:

1. **Create environment file**: `cp .env.example .env`
2. **Add your Gmail credentials** to `.env`:
   ```
   SMTP_EMAIL=your.email@gmail.com
   SMTP_PASSWORD=your_gmail_app_password
   ```
3. **Enable alerts** in `monitoring/drift_check.py` (uncomment email line)

## 📈 Usage Examples

### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "PaperlessBilling": 1
  }'

# Expected response:
# {
#   "churn_prediction": "Will Not Churn",
#   "churn_probability": 0.2341,
#   "confidence": "High",
#   "recommendation": "Low risk. Customer likely to stay. Focus on upselling opportunities.",
#   "timestamp": "2024-02-04T10:30:00"
# }
```

### Train Models Locally
```bash
# 1. Preprocess data
python src/data_preprocessing.py

# 2. Train models with MLflow tracking
python src/train_models.py

# 3. Start MLflow UI to view experiments
mlflow server --host 127.0.0.1 --port 5000
# Visit: http://localhost:5000

# 4. Rebuild API with new model
./docker/manage.sh stop
./docker/manage.sh build
./docker/manage.sh run
```

### Monitor Data Drift
```bash
# Generate test predictions
python generate_predictions.py

# Check for data drift
python monitoring/drift_check.py

# View drift report
open monitoring/drift_report.html
```

### Retrain Model
```bash
# Quick retrain with existing data
python retrain_model.py

# Or full pipeline retrain
python src/train_models.py

# Rebuild and restart API with new model
./docker/manage.sh stop
./docker/manage.sh build
./docker/manage.sh run
```

## 🛠️ Development

### Project Structure
```
├── api/                    # FastAPI application
├── data/                   # Training data (7,043 telecom customers)
│   ├── raw/               # Original dataset (20 features + target)
│   ├── processed/         # Train/validation/test splits
│   └── sagemaker/         # AWS SageMaker format
├── docker/                # Docker management scripts
├── models/                # Trained models (.joblib files)
├── monitoring/            # Drift monitoring with Evidently AI
├── notebooks/             # Jupyter notebooks for EDA
├── sagemaker/            # AWS SageMaker training scripts
├── src/                  # Source code (preprocessing, training)
└── tests/                # Test files
```

### Docker Commands
```bash
# Universal commands (work on all platforms)
docker build -t churn-prediction-api:latest .
docker run -d --name churn-api -p 8000:8000 -v "$(pwd)/logs:/app/logs" churn-prediction-api:latest
docker stop churn-api && docker rm churn-api
docker logs churn-api

# Platform-specific management script
./docker/manage.sh build   # Mac/Linux
./docker/manage.sh run     # Mac/Linux  
./docker/manage.sh stop    # Mac/Linux
./docker/manage.sh test    # Mac/Linux
./docker/manage.sh logs    # Mac/Linux
./docker/manage.sh clean   # Mac/Linux
```

**Windows Volume Mount**: Use `%cd%` instead of `$(pwd)`:
```bash
docker run -d --name churn-api -p 8000:8000 -v "%cd%/logs:/app/logs" churn-prediction-api:latest
```

### 🔧 Troubleshooting

**Docker Issues:**

**All Platforms:**
```bash
# Check if Docker is running
docker --version

# Build without cache if issues
docker build --no-cache -t churn-prediction-api:latest .

# Check container status
docker ps -a

# If port 8000 is busy
docker run -d --name churn-api -p 8001:8000 churn-prediction-api:latest
```

**Mac (Intel):**
```bash
# Standard build (Docker auto-detects x86_64)
docker build -t churn-prediction-api:latest .
```

**Mac (Apple Silicon):**
```bash
# Docker automatically uses ARM64, but you can force it:
docker build --platform linux/arm64 -t churn-prediction-api:latest .
```

**Windows:**
```bash
# If WSL2 issues, try running in PowerShell as Administrator
# Make sure Docker Desktop is running
# Use Windows-style paths for volume mounts
docker run -d --name churn-api -p 8000:8000 -v "%cd%/logs:/app/logs" churn-prediction-api:latest
```

**Linux:**
```bash
# If permission issues
sudo docker build -t churn-prediction-api:latest .
# Or add user to docker group
sudo usermod -aG docker $USER
```

**API Issues:**
```bash
# Check if API is running
curl http://localhost:8000/health

# View API logs
./docker/manage.sh logs

# Test with minimal data
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "SeniorCitizen": 0}'
```

**Model Training Issues:**
```bash
# If MLflow server not running
mlflow server --host 127.0.0.1 --port 5000

# If data files missing
python src/data_preprocessing.py

# Check model files
ls -la models/
```

## 📚 Documentation

- [**Docker Management**](DOCKER_MANAGEMENT.md) - Container operations guide
- [**Model Retraining**](RETRAIN_SUMMARY.md) - MLOps pipeline details
- [**API Documentation**](API_DOCUMENTATION.md) - Endpoints and usage (interactive docs at `/docs` when running locally)

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows that run when you push to `main` or open a pull request.

### What Runs

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **MLOps CI** | Every push/PR to `main` | Runs unit tests, API tests, and builds the Docker image |
| **Drift Monitoring** | Every Monday 2am UTC | Runs drift detection and uploads the report as an artifact |

### About the Docker Build

When you push to *your* fork or clone, the CI builds a Docker image and pushes it to *your* Docker Hub account. That image runs the **churn prediction REST API**—a FastAPI service that serves predictions, health checks, and interactive docs at `/docs`.

**What you get (in your Docker Hub):**
- Pre-built image: `your-dockerhub-username/churn-prediction-api:latest`
- No need to run `docker build` on your server—pull and run
- Useful for deploying to a cloud VM, EC2, Kubernetes, etc.

**Setup:** Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` as [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) in your repo settings. Without these, the Docker build step will fail (tests will still pass).

### Using Your Pre-Built Image

Once CI has pushed an image to your Docker Hub:

```bash
# Pull and run (replace with your Docker Hub username)
docker pull your-dockerhub-username/churn-prediction-api:latest
docker run -d -p 8000:8000 your-dockerhub-username/churn-prediction-api:latest

# Test it
curl http://localhost:8000/health
# Open http://localhost:8000/docs for interactive API documentation
```

The container serves the FastAPI endpoints (`/predict`, `/health`, `/docs`, etc.)—no separate web UI, just the REST API.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Evidently AI](https://evidentlyai.com/) for drift monitoring
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

---

⭐ **If you find this project helpful, please give it a star!**