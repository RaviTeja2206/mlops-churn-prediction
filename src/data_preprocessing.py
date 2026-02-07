import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    """
    Preprocessing pipeline for Telco Customer Churn dataset
    """
    
    def __init__(self, data_path):
        """Load raw data"""
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_data(self):
        """Handle missing values and data type conversions"""
        print("Starting data cleaning...")
        
        # TotalCharges has some spaces, convert to numeric
        self.data['TotalCharges'] = pd.to_numeric(
            self.data['TotalCharges'], 
            errors='coerce'
        )
        
        # Fill missing TotalCharges with 0 (new customers)
        self.data['TotalCharges'].fillna(0, inplace=True)
        
        # Drop customerID (not useful for prediction)
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)
        
        print(f"Cleaned data shape: {self.data.shape}")
        return self
    
    def feature_engineering(self):
        """Create new features and encode categorical variables"""
        print("Performing feature engineering...")
        
        # Binary encoding for Yes/No columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 
                       'PaperlessBilling']
        
        for col in binary_cols:
            self.data[col] = self.data[col].map({'Yes': 1, 'No': 0})
        
        # Handle multi-category columns
        categorical_cols = ['gender', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod']
        
        # One-hot encode categorical features
        self.data = pd.get_dummies(
            self.data, 
            columns=categorical_cols,
            drop_first=True  # Avoid multicollinearity
        )
        
        # Encode target variable (Churn)
        self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})
        
        print(f"Features after engineering: {self.data.shape[1]}")
        return self
    
    def scale_features(self):
        """Scale numerical features"""
        print("Scaling numerical features...")
        
        # Identify numerical columns (excluding target)
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        self.data[numerical_cols] = self.scaler.fit_transform(
            self.data[numerical_cols]
        )
        
        return self
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split into train, validation, and test sets"""
        print("Splitting data...")
        
        # Separate features and target
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class balance
        )
        
        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, path='models/preprocessor.joblib'):
        """Save scaler for future use"""
        joblib.dump(self.scaler, path)
        print(f"Preprocessor saved to {path}")

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Run preprocessing pipeline
    preprocessor.clean_data()\
                .feature_engineering()\
                .scale_features()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n✅ Preprocessing complete!")
