"""
Retrain model locally with proper feature names
This ensures feature names match between training and inference
"""
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def retrain_with_feature_names():
    """Retrain model with explicit feature names"""
    
    print("="*60)
    print("RETRAINING MODEL WITH PROPER FEATURE NAMES")
    print("="*60)
    
    # Load training data
    print("\n📦 Loading training data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Standardize feature names (match API format)
    X_train.columns = [name.replace(' ', '_').replace('(', '').replace(')', '') for name in X_train.columns]
    X_test.columns = [name.replace(' ', '_').replace('(', '').replace(')', '') for name in X_test.columns]
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {list(X_train.columns)}")
    
    # Train XGBoost with feature names
    print("\n🚀 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        enable_categorical=False  # Important: keep False
    )
    
    # Fit the model
    model.fit(X_train, y_train, verbose=False)
    
    # Set feature names for the model (match API format)
    model._Booster.feature_names = list(X_train.columns)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = 'models/churn_model_fixed.joblib'
    joblib.dump(model, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print("   Feature names are now preserved!")
    
    # Verify feature names
    print("\n🔍 Verifying feature names...")
    loaded_model = joblib.load(model_path)
    print(f"   Feature names in model: {loaded_model._Booster.feature_names}")
    
    return model

if __name__ == "__main__":
    model = retrain_with_feature_names()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Update API to use new model: models/churn_model_fixed.joblib")
    print("2. Restart API: uvicorn main:app --reload")
    print("3. Rerun tests: python api/test_api.py")
    print("="*60)
