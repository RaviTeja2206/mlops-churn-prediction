"""
Retrain the customer churn model
Simplified version without MLflow server dependency
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, dataset_name="Test"):
    """Calculate all evaluation metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost model"""
    
    print("\n🚀 Training XGBoost model...")
    
    # Model parameters
    params = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    print(f"  Parameters: {params}")
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on validation set
    print("\n📊 Validation Set Performance:")
    val_metrics, _, _ = evaluate_model(model, X_val, y_val, "val")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on test set
    print("\n📊 Test Set Performance:")
    test_metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, "test")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - XGBoost\nRetrained: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_xgb.png')
    print(f"\n✅ Confusion matrix saved: confusion_matrix_xgb.png")
    plt.close()
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    with open('classification_report_xgb.txt', 'w') as f:
        f.write(f"Model Retrained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"✅ Classification report saved: classification_report_xgb.txt")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as primary model
    joblib.dump(model, 'models/churn_model_xgb.joblib')
    print(f"\n✅ Model saved: models/churn_model_xgb.joblib")
    
    # Save timestamped backup
    backup_path = f'models/churn_model_xgb_{timestamp}.joblib'
    joblib.dump(model, backup_path)
    print(f"✅ Backup saved: {backup_path}")
    
    # Update the fixed model (used by API)
    joblib.dump(model, 'models/churn_model_fixed.joblib')
    print(f"✅ API model updated: models/churn_model_fixed.joblib")
    
    return model, test_metrics

def update_reference_data(X_train):
    """Update reference data for drift monitoring"""
    print("\n📊 Updating drift monitoring reference data...")
    
    # Sample 1000 rows for reference
    sample_size = min(1000, len(X_train))
    reference_sample = X_train.sample(n=sample_size, random_state=42)
    reference_sample.to_csv('monitoring/reference_sample.csv', index=False)
    
    print(f"✅ Reference sample updated: {sample_size} rows")
    print(f"   Location: monitoring/reference_sample.csv")

if __name__ == "__main__":
    print("="*60)
    print("CUSTOMER CHURN MODEL RETRAINING")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Train model
    model, metrics = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Update reference data for drift monitoring
    update_reference_data(X_train)
    
    print("\n" + "="*60)
    print("🎉 MODEL RETRAINING COMPLETE!")
    print("="*60)
    print(f"\n📈 Final Test F1 Score: {metrics['f1_score']:.4f}")
    print(f"📈 Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"📈 Final Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\n📝 Next Steps:")
    print("  1. Review confusion_matrix_xgb.png")
    print("  2. Review classification_report_xgb.txt")
    print("  3. Restart Docker API to use new model:")
    print("     docker-compose restart")
    print("  4. Test predictions with new model")
    print("  5. Run drift check again to verify improvement")
