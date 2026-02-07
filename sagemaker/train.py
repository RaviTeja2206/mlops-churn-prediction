"""
SageMaker training script
This runs on AWS SageMaker instances
"""
import argparse
import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
import json

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=150)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    return parser.parse_args()

def load_data(train_path, val_path):
    """Load training and validation data"""
    print(f"Loading training data from {train_path}")
    print(f"Loading validation data from {val_path}")
    
    # Load CSVs
    train_files = [f for f in os.listdir(train_path) if f.endswith('.csv')]
    val_files = [f for f in os.listdir(val_path) if f.endswith('.csv')]
    
    train_data = pd.read_csv(os.path.join(train_path, train_files[0]), header=None)
    val_data = pd.read_csv(os.path.join(val_path, val_files[0]), header=None)
    
    # Separate features and target (label is first column)
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    X_val = val_data.iloc[:, 1:]
    y_val = val_data.iloc[:, 0]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val

def train_model(args, X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n🚀 Training XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'  # Faster on CPU
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1)
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return metrics

def save_model(model, model_dir):
    """Save model to model directory"""
    print(f"\n💾 Saving model to {model_dir}")
    
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    print(f"✅ Model saved successfully")

if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("AWS SAGEMAKER TRAINING JOB")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(args.train, args.validation)
    
    # Train model
    model = train_model(args, X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    
    # Save metrics
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Save model
    save_model(model, args.model_dir)
    
    print("\n🎉 Training complete!")
