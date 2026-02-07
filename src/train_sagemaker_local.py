"""
Local training script that simulates SageMaker environment
Test this before deploying to AWS!
"""
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os

def train_local():
    """Train model locally with same structure as SageMaker"""
    
    print("🏠 Running LOCAL training (SageMaker simulation)...")
    
    # Load data (simulating SageMaker channels)
    print("Loading data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    
    # Hyperparameters (same as we'll use on SageMaker)
    hyperparameters = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    print(f"Training with hyperparameters: {hyperparameters}")
    
    # Train model
    model = XGBClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"\n📊 Validation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save model (simulating SageMaker model artifacts)
    model_dir = 'models/sagemaker_local'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    print("💡 If results look good, proceed to AWS SageMaker training!")
    
    return model, accuracy, f1

if __name__ == "__main__":
    model, accuracy, f1 = train_local()
    
    # Estimate AWS costs
    print("\n" + "="*60)
    print("💰 ESTIMATED AWS SAGEMAKER COSTS")
    print("="*60)
    print(f"Instance: ml.m5.xlarge")
    print(f"On-Demand: $0.23/hour")
    print(f"Spot Instance: ~$0.07/hour (70% savings)")
    print(f"\nEstimated training time: ~5 minutes")
    print(f"Estimated cost (spot): $0.006 (~₹0.50)")
    print("="*60)
