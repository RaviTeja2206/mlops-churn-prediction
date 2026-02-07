# Model Setup Instructions

This public repository excludes trained model files (*.joblib) to keep the repo size manageable.

## Quick Setup (Recommended)

1. **Train models locally**:
   ```bash
   # Preprocess data
   python src/data_preprocessing.py
   
   # Train models (creates model files in models/ directory)
   python src/train_models.py
   ```

2. **Build and run the API**:
   ```bash
   ./docker/manage.sh build
   ./docker/manage.sh run
   ```

## Alternative: Use Pre-trained Models

If you want to skip training and use pre-trained models:

1. Download from the releases page (when available)
2. Place in the `models/` directory:
   - `churn_model_xgb.joblib`
   - `preprocessor.joblib`
   - `model_metadata.txt`

## AWS SageMaker Setup (Optional)

For cloud training with AWS SageMaker:

1. **Update AWS Account ID** in `sagemaker/train_sagemaker.py`:
   ```python
   role_arn = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
   ```
   Replace `YOUR_ACCOUNT_ID` with your actual AWS account ID.

2. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

3. **Run SageMaker training**:
   ```bash
   python sagemaker/train_sagemaker.py
   ```

## Email Alerts Setup (Optional)

For drift monitoring email alerts:

1. **Update email in `monitoring/drift_check.py`**:
   ```python
   msg["From"] = "your.actual.email@gmail.com"
   msg["To"] = "your.actual.email@gmail.com"
   smtp.login("your.actual.email@gmail.com", "your_gmail_app_password")
   ```

2. **Uncomment the email line** in `drift_check.py`:
   ```python
   send_email(REPORT_PATH, dataset_drift)  # Remove the # comment
   ```

## Verify Setup

Test that everything works:
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", "model": "loaded"}
```

The training process takes about 2-3 minutes on a modern laptop.
