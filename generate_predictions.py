"""
Generate 50+ varied predictions for drift monitoring testing
"""
import requests
import random
import time
from datetime import datetime

API_URL = "http://localhost:8000/predict"

# Define varied customer profiles for realistic testing
customer_profiles = [
    # High-risk profiles (likely to churn)
    {
        "name": "Short tenure, high charges, month-to-month",
        "tenure": random.randint(1, 6),
        "MonthlyCharges": random.uniform(80, 120),
        "TotalCharges": lambda t, m: t * m,
        "SeniorCitizen": 1,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "gender_Male": random.randint(0, 1),
        "InternetService_Fiber_optic": 1,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Electronic_check": 1,
        "OnlineSecurity_Yes": 0,
        "TechSupport_Yes": 0,
    },
    # Low-risk profiles (likely to stay)
    {
        "name": "Long tenure, low charges, two-year contract",
        "tenure": random.randint(48, 72),
        "MonthlyCharges": random.uniform(20, 50),
        "TotalCharges": lambda t, m: t * m,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 1,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "gender_Male": random.randint(0, 1),
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "PaymentMethod_Credit_card_automatic": 1,
        "OnlineSecurity_Yes": 1,
        "TechSupport_Yes": 1,
    },
    # Medium-risk profiles
    {
        "name": "Medium tenure, medium charges, one-year contract",
        "tenure": random.randint(12, 36),
        "MonthlyCharges": random.uniform(50, 80),
        "TotalCharges": lambda t, m: t * m,
        "SeniorCitizen": random.randint(0, 1),
        "Partner": random.randint(0, 1),
        "Dependents": random.randint(0, 1),
        "PhoneService": 1,
        "PaperlessBilling": random.randint(0, 1),
        "gender_Male": random.randint(0, 1),
        "InternetService_Fiber_optic": random.randint(0, 1),
        "Contract_One_year": 1,
        "Contract_Two_year": 0,
        "PaymentMethod_Mailed_check": 1,
    },
    # New customers
    {
        "name": "Very new customer",
        "tenure": random.randint(0, 3),
        "MonthlyCharges": random.uniform(60, 100),
        "TotalCharges": lambda t, m: t * m if t > 0 else random.uniform(0, 100),
        "SeniorCitizen": 0,
        "Partner": random.randint(0, 1),
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "gender_Male": random.randint(0, 1),
        "InternetService_Fiber_optic": 1,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Electronic_check": 1,
    },
    # Loyal customers
    {
        "name": "Very loyal customer",
        "tenure": random.randint(60, 72),
        "MonthlyCharges": random.uniform(30, 70),
        "TotalCharges": lambda t, m: t * m,
        "SeniorCitizen": random.randint(0, 1),
        "Partner": 1,
        "Dependents": 1,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "gender_Male": random.randint(0, 1),
        "InternetService_No": 1,
        "Contract_Two_year": 1,
        "PaymentMethod_Credit_card_automatic": 1,
        "OnlineSecurity_Yes": 1,
        "TechSupport_Yes": 1,
        "StreamingTV_Yes": 1,
        "StreamingMovies_Yes": 1,
    },
]

def create_customer_data(profile):
    """Create a complete customer data dict from a profile"""
    # Start with all features set to 0
    customer = {
        "tenure": 0,
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 0,
        "PaperlessBilling": 0,
        "gender_Male": 0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 0,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 0,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0,
    }
    
    # Update with profile values
    for key, value in profile.items():
        if key == "name":
            continue
        if callable(value):
            continue
        if isinstance(value, int):
            customer[key] = value
        elif isinstance(value, float):
            customer[key] = value
    
    # Calculate TotalCharges if it's a function
    if "TotalCharges" in profile and callable(profile["TotalCharges"]):
        customer["TotalCharges"] = profile["TotalCharges"](
            customer["tenure"], customer["MonthlyCharges"]
        )
    
    return customer

def make_prediction(customer_data, profile_name):
    """Make a prediction via API"""
    try:
        response = requests.post(API_URL, json=customer_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {profile_name}: {result['churn_prediction']} "
                  f"(prob: {result['churn_probability']:.3f})")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the API running? Start it with:")
        print("   docker-compose up -d")
        print("   or: cd api && uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Generating 50+ Predictions for Drift Monitoring")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code != 200:
            print("❌ API health check failed!")
            return
        print("✅ API is healthy and ready\n")
    except:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("Please start the API first:")
        print("  Option 1: docker-compose up -d")
        print("  Option 2: cd api && uvicorn main:app --reload")
        return
    
    successful = 0
    failed = 0
    
    # Generate 50 predictions (10 from each profile)
    for i in range(50):
        profile = random.choice(customer_profiles)
        customer_data = create_customer_data(profile.copy())
        
        print(f"[{i+1}/50] ", end="")
        if make_prediction(customer_data, profile["name"]):
            successful += 1
        else:
            failed += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print(f"Completed: {successful} successful, {failed} failed")
    print(f"Predictions logged to: logs/predictions.csv")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check logs/predictions.csv")
    print("2. Run drift check: python monitoring/drift_check.py")

if __name__ == "__main__":
    main()
