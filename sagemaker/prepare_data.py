"""
Prepare and upload data to S3 for SageMaker training
"""
import pandas as pd
import boto3
from sagemaker import Session
import os

def prepare_sagemaker_data():
    """Combine features and target for SageMaker"""
    
    print("📦 Preparing data for SageMaker...")
    
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv')
    
    # Combine features and target (label first for XGBoost)
    train_data = pd.concat([y_train, X_train], axis=1)
    val_data = pd.concat([y_val, X_val], axis=1)
    
    # Create directory
    os.makedirs('data/sagemaker', exist_ok=True)
    
    # Save without header for SageMaker XGBoost
    train_data.to_csv('data/sagemaker/train.csv', index=False, header=False)
    val_data.to_csv('data/sagemaker/validation.csv', index=False, header=False)
    
    print(f"✅ Training data: {train_data.shape}")
    print(f"✅ Validation data: {val_data.shape}")
    
    return train_data, val_data

def upload_to_s3():
    """Upload data to S3"""
    
    print("\n☁️  Uploading data to S3...")
    
    try:
        # Initialize SageMaker session
        sagemaker_session = Session()
        bucket = sagemaker_session.default_bucket()
        prefix = 'churn-prediction'
        
        print(f"Using S3 bucket: {bucket}")
        
        # Upload training data
        train_s3 = sagemaker_session.upload_data(
            path='data/sagemaker/train.csv',
            bucket=bucket,
            key_prefix=f'{prefix}/train'
        )
        
        # Upload validation data
        val_s3 = sagemaker_session.upload_data(
            path='data/sagemaker/validation.csv',
            bucket=bucket,
            key_prefix=f'{prefix}/validation'
        )
        
        print(f"\n✅ Training data uploaded to:")
        print(f"   {train_s3}")
        print(f"✅ Validation data uploaded to:")
        print(f"   {val_s3}")
        
        return train_s3, val_s3, bucket
        
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        print("\n💡 Make sure AWS credentials are configured:")
        print("   aws configure")
        raise

if __name__ == "__main__":
    # Prepare data
    train_data, val_data = prepare_sagemaker_data()
    
    # Upload to S3
    train_s3, val_s3, bucket = upload_to_s3()
    
    print("\n" + "="*60)
    print("🎯 Next Steps:")
    print("="*60)
    print("1. Data is ready in S3")
    print("2. Run: python sagemaker/train_sagemaker.py")
    print("="*60)
