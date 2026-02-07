"""
Download your trained model from S3
Run this BEFORE cleaning up S3 to save costs
"""
import boto3
import tarfile
import os

def download_and_extract_model():
    """Download the model you just trained"""
    
    # Your model S3 URI from the output
    model_s3_uri = "s3://sagemaker-REGION-YOUR_ACCOUNT_ID/sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-XXX/output/model.tar.gz"
    
    print("="*60)
    print("DOWNLOADING SAGEMAKER MODEL")
    print("="*60)
    
    # Parse S3 URI
    parts = model_s3_uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    
    print(f"\n📦 Source: {model_s3_uri}")
    print(f"   Bucket: {bucket}")
    print(f"   Key: {key}")
    
    # Download
    s3 = boto3.client('s3')
    local_tar = 'models/sagemaker_model.tar.gz'
    
    os.makedirs('models', exist_ok=True)
    
    print(f"\n⬇️  Downloading...")
    s3.download_file(bucket, key, local_tar)
    
    file_size = os.path.getsize(local_tar)
    print(f"✅ Downloaded: {file_size / (1024**2):.2f} MB")
    
    # Extract
    print(f"\n📂 Extracting...")
    extract_dir = 'models/sagemaker_production'
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    print(f"✅ Extracted to: {extract_dir}/")
    
    # List contents
    print(f"\n📄 Model files:")
    for file in os.listdir(extract_dir):
        file_path = os.path.join(extract_dir, file)
        size = os.path.getsize(file_path)
        print(f"   - {file} ({size / 1024:.2f} KB)")
    
    # Calculate S3 cost savings
    monthly_cost = (file_size / 1e9) * 0.023
    print(f"\n💰 S3 Storage Costs (if kept in S3):")
    print(f"   Monthly: ${monthly_cost:.6f} (~₹{monthly_cost * 84:.4f})")
    print(f"   Yearly:  ${monthly_cost * 12:.6f} (~₹{monthly_cost * 12 * 84:.4f})")
    print(f"\n💡 Model is now saved locally - you can delete from S3!")
    
    return extract_dir

if __name__ == "__main__":
    model_dir = download_and_extract_model()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Test model locally:")
    print(f"   python api/test_model.py")
    print("\n2. Clean up S3 to save costs:")
    print("   python sagemaker/s3_cost_manager.py")
    print("\n3. Proceed to Phase 4 (FastAPI deployment)")
    print("="*60)
