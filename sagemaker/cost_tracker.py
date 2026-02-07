"""
Track and display SageMaker costs
"""
import boto3
from datetime import datetime, timedelta

def get_training_costs():
    """Get SageMaker training job costs"""
    
    sagemaker = boto3.client('sagemaker')
    
    # List recent training jobs
    response = sagemaker.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10
    )
    
    print("Recent SageMaker Training Jobs:")
    print("="*80)
    
    for job in response['TrainingJobSummaries']:
        job_name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        duration = job.get('TrainingEndTime', datetime.now()) - job['CreationTime']
        
        # Get detailed info
        job_details = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        instance_type = job_details['ResourceConfig']['InstanceType']
        instance_count = job_details['ResourceConfig']['InstanceCount']
        
        billable_seconds = job_details.get('BillableTimeInSeconds', 0)
        training_seconds = job_details.get('TrainingTimeInSeconds', 0)
        
        # Estimate cost (ml.m5.xlarge spot ~$0.07/hour)
        cost_per_hour = 0.07
        estimated_cost = (billable_seconds / 3600) * cost_per_hour * instance_count
        
        print(f"\nJob: {job_name}")
        print(f"Status: {status}")
        print(f"Instance: {instance_type} x {instance_count}")
        print(f"Duration: {duration}")
        print(f"Billable Time: {billable_seconds}s")
        print(f"Estimated Cost: ${estimated_cost:.4f} (~₹{estimated_cost * 84:.2f})")
        print("-"*80)

if __name__ == "__main__":
    get_training_costs()
