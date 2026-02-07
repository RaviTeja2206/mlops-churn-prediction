"""
Test the trained model locally before creating API
"""
import joblib
import pandas as pd
import numpy as np
import os

def find_model():
    """Find the trained model"""
    
    # Check multiple locations
    locations = [
        'models/sagemaker_production/xgboost-model',
        'models/churn_model_xgb.joblib',
        'models/model.joblib'
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            return loc
    
    raise FileNotFoundError("Model not found! Run download_trained_model.py first")

def load_model():
    """Load trained model"""
    model_path = find_model()
    print(f"📦 Loading model from: {model_path}")
    
    # XGBoost models from SageMaker are in pickle format
    import pickle
    
    if model_path.endswith('.joblib'):
        model = joblib.load(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    print("✅ Model loaded successfully")
    return model

def test_prediction():
    """Test model with sample data"""
    
    # Load model
    model = load_model()
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"\n📊 Testing on {len(X_test)} samples...")
    
    # Make predictions
    predictions = model.predict(X_test[:10])
    probabilities = model.predict_proba(X_test[:10])
    
    print("\n🎯 Sample Predictions:")
    print("-" * 60)
    
    for i in range(10):
        actual = "Churn" if y_test[i] == 1 else "No Churn"
        predicted = "Churn" if predictions[i] == 1 else "No Churn"
        confidence = probabilities[i][1]
        
        status = "✅" if actual == predicted else "❌"
        print(f"{status} Sample {i+1}: Actual={actual:10s} | Predicted={predicted:10s} | Confidence={confidence:.2%}")
    
    # Overall accuracy
    accuracy = (predictions[:10] == y_test[:10]).mean()
    print(f"\n📈 Accuracy on 10 samples: {accuracy:.2%}")
    
    print("\n✅ Model is working! Ready for API deployment")
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("MODEL TESTING")
    print("="*60)
    
    try:
        model = test_prediction()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've:")
        print("   1. Run preprocessing: python src/data_preprocessing.py")
        print("   2. Downloaded model: python sagemaker/download_trained_model.py")
