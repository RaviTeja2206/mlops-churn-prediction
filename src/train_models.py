import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-prediction")

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, dataset_name="Test"):
    """Calculate all evaluation metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
        f'{dataset_name}_precision': precision_score(y, y_pred),
        f'{dataset_name}_recall': recall_score(y, y_pred),
        f'{dataset_name}_f1_score': f1_score(y, y_pred),
        f'{dataset_name}_roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression baseline"""
    
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        # Log parameters
        params = {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'}
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, "val")
        mlflow.log_metrics(val_metrics)
        
        # Evaluate on test set
        test_metrics, _, _ = evaluate_model(model, X_test, y_test, "test")
        mlflow.log_metrics(test_metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Logistic Regression - Test F1: {test_metrics['test_f1_score']:.4f}")
        
        return model

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest"""
    
    with mlflow.start_run(run_name="random_forest_v1"):
        # Log parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, "val")
        mlflow.log_metrics(val_metrics)
        
        test_metrics, _, _ = evaluate_model(model, X_test, y_test, "test")
        mlflow.log_metrics(test_metrics)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance_rf.csv', index=False)
        mlflow.log_artifact('feature_importance_rf.csv')
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Random Forest - Test F1: {test_metrics['test_f1_score']:.4f}")
        
        return model

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost (Best model)"""
    
    with mlflow.start_run(run_name="xgboost_tuned"):
        # Log parameters
        params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, "val")
        mlflow.log_metrics(val_metrics)
        
        test_metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, "test")
        mlflow.log_metrics(test_metrics)
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - XGBoost')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix_xgb.png')
        mlflow.log_artifact('confusion_matrix_xgb.png')
        plt.close()
        
        # Log classification report
        report = classification_report(y_test, y_pred)
        with open('classification_report_xgb.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report_xgb.txt')
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save best model locally
        joblib.dump(model, 'models/churn_model_xgb.joblib')
        print(f"✅ XGBoost - Test F1: {test_metrics['test_f1_score']:.4f}")
        print(f"✅ Model saved to models/churn_model_xgb.joblib")
        
        return model

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Train all models
    print("\n🚀 Training models...")
    
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    rf_model = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n🎉 All models trained! Check MLflow UI at http://localhost:5000")
