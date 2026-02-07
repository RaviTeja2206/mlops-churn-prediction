# Model Retraining Summary

## Trigger
- **Reason:** Data drift detected (58.8% of features drifted)
- **Drift Report:** monitoring/drift_report.html

## Training Process

### Data
- **Training Samples:** 4,225
- **Validation Samples:** 1,056
- **Test Samples:** 1,127
- **Features:** 28

### Models Trained
All three models were trained and logged to MLflow:

1. **Logistic Regression (Baseline)**
   - Test F1: 0.6017
   - Test Accuracy: 0.8160

2. **Random Forest**
   - Test F1: 0.5740
   - Test Accuracy: 0.8098

3. **XGBoost (Selected for Production)**
   - Test F1: 0.5804
   - Test Accuracy: 0.8160
   - Test Precision: 0.6110
   - Test Recall: 0.5530
   - Test ROC-AUC: 0.8450

## Deployment

### Files Updated
- ✅ `models/churn_model_xgb.joblib` - Primary model
- ✅ `models/churn_model_fixed.joblib` - API model
- ✅ `monitoring/reference_sample.csv` - Updated reference data (1000 rows)
- ✅ `models/model_metadata.txt` - Metadata log

### Docker Deployment
```bash
docker-compose build
docker-compose up -d
```

### Verification
- ✅ API health check passed
- ✅ Model loaded successfully
- ✅ Test predictions working
- ✅ Prediction logging functional

## MLflow Tracking

**Experiment:** customer-churn-prediction  
**MLflow UI:** http://127.0.0.1:5000

**Run IDs:**
- Logistic Regression: `1fea1b31fec44c1eacf10827a924eb20`
- Random Forest: `019cfd383da1470d83131cce890108db`
- XGBoost: `36e018d7a0754dec96150e3ef2745be8`

## Artifacts Generated
- `confusion_matrix_xgb.png` - Visual performance analysis
- `classification_report_xgb.txt` - Detailed metrics
- `feature_importance_rf.csv` - Feature importance from RF

## Post-Retrain Drift Check
- **New Predictions Generated:** 50
- **Drift Status:** Still detected (70.6%) - Expected with synthetic test data
- **Note:** In production, drift should reduce after retraining with real production data

## Next Steps

1. **Monitor Performance**
   - Track prediction accuracy in production
   - Monitor drift metrics weekly
   - Review MLflow dashboard regularly

2. **Data Collection**
   - Continue logging predictions to `logs/predictions.csv`
   - Collect ground truth labels for validation
   - Build feedback loop for continuous improvement

3. **Scheduled Retraining**
   - Set up automated drift monitoring (GitHub Actions)
   - Define retraining triggers (drift threshold, performance degradation)
   - Implement A/B testing for model comparison

4. **Documentation**
   - Update README with new model version
   - Document any changes in model behavior
   - Keep model_metadata.txt updated

## Commands Reference

### View MLflow UI
```bash
mlflow server --host 127.0.0.1 --port 5000
# Visit: http://localhost:5000
```

### Retrain Model
```bash
python src/train_models.py
```

### Test API
```bash
python api/test_api.py
```

### Generate Test Predictions
```bash
python generate_predictions.py
```

### Check Drift
```bash
python monitoring/drift_check.py
```

### Restart Docker API
```bash
docker-compose restart
```

## Notes
- All experiments tracked in MLflow for reproducibility
- Model versioning maintained through git commits
- Reference data updated to match new training distribution
- Full pipeline executed (not partial retraining)
- Docker image rebuilt with new model artifacts
