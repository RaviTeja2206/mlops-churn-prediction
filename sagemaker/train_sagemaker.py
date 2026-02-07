"""
Launch SageMaker training job with SPOT INSTANCES (70% cost savings!)
"""
import sagemaker
from sagemaker.xgboost import XGBoost
import boto3
import time

def get_execution_role():
    """Get SageMaker execution role"""
    
    # Try to get role from SageMaker notebook
    try:
        role = sagemaker.get_execution_role()
        return role
    except:
        # If running locally, use IAM role ARN
        # Replace with your role ARN from Step 3.2.2
        role_arn = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
        print(f"⚠️  Using IAM role: {role_arn}")
        print("   Make sure to replace YOUR_ACCOUNT_ID with your AWS account ID!")
        return role_arn

def launch_training_job(use_spot=True):
    """Launch SageMaker training job"""
    
    print("="*60)
    print("🚀 LAUNCHING SAGEMAKER TRAINING JOB")
    print("="*60)
    
    # Get execution role
    role = get_execution_role()
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    prefix = 'churn-prediction'
    
    print(f"\n📋 Configuration:")
    print(f"   S3 Bucket: {bucket}")
    print(f"   Prefix: {prefix}")
    print(f"   Role: {role}")
    
    # Training data locations
    train_input = f's3://{bucket}/{prefix}/train'
    val_input = f's3://{bucket}/{prefix}/validation'
    
    # Create estimator
    if use_spot:
        print(f"\n💰 Using SPOT INSTANCES (70% cost savings!)")
        print(f"   On-demand: $0.23/hour")
        print(f"   Spot: ~$0.07/hour")
        
        estimator = XGBoost(
            entry_point='train.py',
            source_dir='sagemaker/',
            role=role,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            framework_version='1.7-1',
            
            # SPOT INSTANCE CONFIGURATION
            use_spot_instances=True,
            max_run=3600,  # 1 hour max training time
            max_wait=3600,  # 1 hour max wait time
            checkpoint_s3_uri=f's3://{bucket}/{prefix}/checkpoints/',
            
            hyperparameters={
                'n-estimators': 150,
                'max-depth': 5,
                'learning-rate': 0.1,
                'subsample': 0.8,
                'colsample-bytree': 0.8
            }
        )
    else:
        print(f"\n💵 Using ON-DEMAND instances")
        print(f"   Cost: $0.23/hour")
        
        estimator = XGBoost(
            entry_point='train.py',
            source_dir='sagemaker/',
            role=role,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            framework_version='1.7-1',
            
            hyperparameters={
                'n-estimators': 150,
                'max-depth': 5,
                'learning-rate': 0.1,
                'subsample': 0.8,
                'colsample-bytree': 0.8
            }
        )
    
    # Start training
    print(f"\n⏱️  Starting training job at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    print(f"   This will take approximately 5-10 minutes")
    print(f"   Estimated cost: $0.01 - $0.02 (~₹0.84 - ₹1.68)")
    
    estimator.fit({
        'train': train_input,
        'validation': val_input
    })
    
    print("\n✅ Training job completed!")
    print(f"   Model artifacts: {estimator.model_data}")
    
    # Print cost information
    job_name = estimator.latest_training_job.name
    print(f"\n💰 View training job costs:")
    print(f"   AWS Console → SageMaker → Training Jobs → {job_name}")
    
    return estimator

if __name__ == "__main__":
    import sys
    
    # Ask user confirmation
    print("\n⚠️  WARNING: This will incur AWS charges!")
    print("Estimated cost: $0.01 - $0.02 (~₹0.84 - ₹1.68)")
    
    response = input("\nProceed with SageMaker training? (yes/no): ")
    
    if response.lower() == 'yes':
        estimator = launch_training_job(use_spot=True)
        
        print("\n" + "="*60)
        print("🎉 SAGEMAKER TRAINING COMPLETE!")
        print("="*60)
        print("Next: Deploy model as FastAPI (Phase 4)")
    else:
        print("\n❌ Training cancelled")
        print("💡 Test locally first: python src/train_sagemaker_local.py")
