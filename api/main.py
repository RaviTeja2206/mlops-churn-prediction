"""
FastAPI application for churn prediction
M1 Mac compatible
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import csv, threading
import joblib
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using ML model trained on AWS SageMaker",
    version="1.0.0"
)

# Global model variable
model = None

LOG_FILE = '../logs/predictions.csv'
LOCK = threading.Lock()

def log_prediction(data, response, status, exec_time):
    """Append request/response details to CSV"""
    row = {**data, 'prediction': response['churn_prediction'],
           'probability': response['churn_probability'],
           'status': status, 'exec_time': exec_time, 'timestamp': datetime.now().isoformat()}
    with LOCK:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0: writer.writeheader()
            writer.writerow(row)


def load_model_at_startup():
    """Load model when API starts"""
    global model

    model_paths = [
        '../models/churn_model_fixed.joblib',  # New model with fixed feature names
        '../models/sagemaker_production/model.joblib',  # Correct relative path from api/
        '/Users/apple/Documents/Practice ML/customer-churn-mlops/models/sagemaker_production/model.joblib'  # Absolute path as fallback
    ]
    
    for path in model_paths:
        print(f"Checking path: {path}")
        print(os.path.abspath(path))
        if os.path.exists(path):
            print(f"✅ Model found at: {path}")
            try:
                if path.endswith('.joblib'):
                    model = joblib.load(path)
                elif path.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    continue
                
                print(f"✅ Model loaded from: {path}")
                return
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
    
    raise Exception("Model not found!")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model_at_startup()

# Input data model
class CustomerData(BaseModel):
    """Customer features for prediction"""
    tenure: int = Field(..., ge=0, le=100, description="Months with company")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Dependents: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    PhoneService: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    PaperlessBilling: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    
    # Add categorical features (one-hot encoded)
    gender_Male: int = Field(0, ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(0, ge=0, le=1)
    MultipleLines_Yes: int = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1)
    InternetService_No: int = Field(0, ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(0, ge=0, le=1)
    OnlineSecurity_Yes: int = Field(0, ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(0, ge=0, le=1)
    OnlineBackup_Yes: int = Field(0, ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(0, ge=0, le=1)
    DeviceProtection_Yes: int = Field(0, ge=0, le=1)
    TechSupport_No_internet_service: int = Field(0, ge=0, le=1)
    TechSupport_Yes: int = Field(0, ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(0, ge=0, le=1)
    StreamingTV_Yes: int = Field(0, ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(0, ge=0, le=1)
    StreamingMovies_Yes: int = Field(0, ge=0, le=1)
    Contract_One_year: int = Field(0, ge=0, le=1)
    Contract_Two_year: int = Field(0, ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0,
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "PhoneService": 1,
                "PaperlessBilling": 1,
                "gender_Male": 1,
                "Contract_Two_year": 1,
                "PaymentMethod_Electronic_check": 0
            }
        }

# Response model
class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    confidence: str
    recommendation: str
    timestamp: str

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict customer churn probability
    
    Returns:
    - churn_prediction: "Will Churn" or "Will Not Churn"
    - churn_probability: Probability of churning (0-1)
    - confidence: "High", "Medium", or "Low"
    - recommendation: Action recommendation
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start = datetime.now()

    try:
        # Convert input to DataFrame
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Reorder columns to match model feature names
        input_df = input_df[model._Booster.feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Determine confidence level
        if probability > 0.7 or probability < 0.3:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate recommendation
        if probability > 0.7:
            recommendation = "High churn risk! Contact customer immediately with retention offer."
        elif probability > 0.5:
            recommendation = "Moderate risk. Monitor customer satisfaction and engagement."
        else:
            recommendation = "Low risk. Customer likely to stay. Focus on upselling opportunities."
        
        log_prediction(input_dict, {"churn_prediction":..., "churn_probability":...}, "success",
                    (datetime.now()-start).total_seconds())

        return PredictionResponse(
            churn_prediction="Will Churn" if prediction == 1 else "Will Not Churn",
            churn_probability=round(float(probability), 4),
            confidence=confidence,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        log_prediction(input_dict, {"error": str(e)}, "error",
                    (datetime.now()-start).total_seconds())

@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """
    Batch prediction for multiple customers
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for customer in customers:
        try:
            input_dict = customer.dict()
            input_df = pd.DataFrame([input_dict])
            
            # Reorder columns to match model feature names
            input_df = input_df[model._Booster.feature_names]
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            results.append({
                "churn_prediction": "Will Churn" if prediction == 1 else "Will Not Churn",
                "churn_probability": round(float(probability), 4)
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "count": len(results)}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "XGBoost Classifier",
        "training_platform": "AWS SageMaker",
        "features_count": 28,
        "accuracy": 0.816,
        "f1_score": 0.611,
        "training_cost": "$0.0006",
        "spot_savings": "69%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
