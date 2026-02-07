# Docker Management Guide

This project uses `docker/manage.sh` for simplified Docker operations.

## Quick Start

```bash
# Make script executable (first time only)
chmod +x docker/manage.sh

# Build, run, and test
./docker/manage.sh build
./docker/manage.sh run
./docker/manage.sh test
```

## Available Commands

### 1. Build Docker Image
```bash
./docker/manage.sh build
```
- Builds Docker image for ARM64 (M1 Mac)
- Tags as `churn-prediction-api:latest`
- Shows image size and details

**Output:**
```
🐳 Building Docker image for ARM64 (M1 Mac)...
✅ Build complete
REPOSITORY             TAG       IMAGE ID       CREATED          SIZE
churn-prediction-api   latest    70dc1be56d5f   14 minutes ago   1.74GB
```

### 2. Run Container
```bash
./docker/manage.sh run
```
- Starts container in detached mode
- Maps port 8000:8000
- Mounts `./logs` directory for prediction logging
- Waits 5 seconds for startup
- Shows container status

**Output:**
```
🚀 Starting container...
⏳ Waiting for API to start...
✅ Container started
📍 API available at: http://localhost:8000
📖 API docs: http://localhost:8000/docs
```

**Access Points:**
- API: http://localhost:8000
- Health Check: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Stop Container
```bash
./docker/manage.sh stop
```
- Stops running container
- Removes container (keeps image)

**Output:**
```
🛑 Stopping container...
✅ Container stopped and removed
```

### 4. View Logs
```bash
./docker/manage.sh logs
```
- Shows real-time container logs
- Press `Ctrl+C` to exit

**Example Output:**
```
📋 Container logs:
INFO:     Started server process [1]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Model loaded from: ../models/churn_model_fixed.joblib
INFO:     192.168.65.1:38503 - "GET /health HTTP/1.1" 200 OK
```

### 5. Test API
```bash
./docker/manage.sh test
```
- Runs comprehensive API test suite
- Tests health check, model info, predictions, and batch predictions
- Requires Python environment with requests library

**Output:**
```
🧪 Testing API...
============================================================
FASTAPI TESTING SUITE
============================================================
✅ Health check passed
✅ Model info retrieved
✅ High-risk customer prediction successful
✅ Low-risk customer prediction successful
✅ Batch prediction successful
```

### 6. Open Shell in Container
```bash
./docker/manage.sh shell
```
- Opens interactive bash shell inside running container
- Useful for debugging
- Type `exit` to leave shell

**Example:**
```
🐚 Opening shell in container...
appuser@c4bb0837727c:/app/api$ ls
main.py  test_api.py  test_model.py
appuser@c4bb0837727c:/app/api$ exit
```

### 7. Clean Up Everything
```bash
./docker/manage.sh clean
```
- Stops and removes container
- Removes Docker image
- Frees up disk space

**Output:**
```
🧹 Cleaning up...
✅ Cleanup complete
```

## Common Workflows

### Initial Setup
```bash
# Build and start
./docker/manage.sh build
./docker/manage.sh run

# Verify it's working
./docker/manage.sh test
```

### Development Cycle
```bash
# After code changes
./docker/manage.sh stop
./docker/manage.sh build
./docker/manage.sh run
./docker/manage.sh test
```

### Debugging
```bash
# View logs
./docker/manage.sh logs

# Or open shell
./docker/manage.sh shell
```

### Complete Rebuild
```bash
# Clean everything and rebuild
./docker/manage.sh clean
./docker/manage.sh build
./docker/manage.sh run
```

### After Model Retraining
```bash
# Rebuild with new model
./docker/manage.sh stop
./docker/manage.sh build
./docker/manage.sh run

# Test new model
./docker/manage.sh test
python generate_predictions.py
python monitoring/drift_check.py
```

## Volume Mounts

The container mounts the following directories:
- `./logs:/app/logs` - Prediction logs for drift monitoring

## Container Details

- **Container Name:** `churn-api`
- **Image Name:** `churn-prediction-api:latest`
- **Platform:** `linux/arm64` (M1 Mac optimized)
- **Port:** `8000:8000`
- **User:** `appuser` (non-root for security)

## Troubleshooting

### Container won't start
```bash
# Check logs
./docker/manage.sh logs

# Clean and rebuild
./docker/manage.sh clean
./docker/manage.sh build
./docker/manage.sh run
```

### Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Stop the container
./docker/manage.sh stop
```

### Model not loading
```bash
# Check if model files exist
ls -lh models/

# Open shell and check
./docker/manage.sh shell
ls -lh /app/models/
```

### Logs not being created
```bash
# Verify volume mount
docker inspect churn-api | grep -A 5 Mounts

# Check logs directory permissions
ls -la logs/
```

## Integration with Other Tools

### With MLflow
```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# In another terminal
./docker/manage.sh run
```

### With Drift Monitoring
```bash
# Generate predictions
python generate_predictions.py

# Check drift
python monitoring/drift_check.py

# View drift report
open monitoring/drift_report.html
```

### With Streamlit Dashboard
```bash
# Start API
./docker/manage.sh run

# Start dashboard (in another terminal)
cd dashboard
streamlit run app.py
```

## Performance Tips

1. **Build Cache:** Docker caches layers. Only changed layers rebuild.
2. **Image Size:** Current image is ~1.74GB. Consider multi-stage builds for smaller size.
3. **Logs:** Use `./docker/manage.sh logs` sparingly in production.
4. **Health Checks:** Container has built-in health checks (30s interval).

## Security Notes

- Container runs as non-root user (`appuser`)
- No sensitive data in image
- Environment variables for secrets (not hardcoded)
- Regular security updates recommended

## Current Status

✅ **Container:** Running  
✅ **Model:** Loaded (churn_model_fixed.joblib)  
✅ **API:** Healthy  
✅ **Logs:** Mounted and working  
✅ **Tests:** Passing
