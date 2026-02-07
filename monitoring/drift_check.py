"""
Data Drift Monitoring using Evidently AI
Compares production predictions against reference training data
"""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import smtplib
from email.message import EmailMessage

def send_email(report_path, drift_detected):
    """Send email alert if drift is detected"""
    msg = EmailMessage()
    msg["Subject"] = "ML Drift Alert: Customer Churn Model"
    msg["From"] = "your.email@gmail.com"
    msg["To"] = "your.email@gmail.com"  # you can add multiple emails
    msg.set_content("ML drift report attached. Review for potential retraining.")

    with open(report_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="text", subtype="html", filename="drift_report.html")

    # Send if drift detected
    if drift_detected:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            # Update email and app password
            smtp.login("your.email@gmail.com", "app_password_here")
            smtp.send_message(msg)
        print("📧 Drift alert sent via email!")
    else:
        print("No drift: email not sent.")

# File paths
REF_PATH = "monitoring/reference_sample.csv"
PROD_PATH = "logs/predictions.csv"
REPORT_PATH = "monitoring/drift_report.html"

print("=" * 60)
print("EVIDENTLY DATA DRIFT MONITORING")
print("=" * 60)

# Load reference and production data
ref = pd.read_csv(REF_PATH)
prod = pd.read_csv(PROD_PATH)

# Keep only columns present in BOTH sets, ignore log columns like 'prediction','status'
feature_cols = [col for col in ref.columns if col in prod.columns]
ref = ref[feature_cols]
prod = prod[feature_cols]

# Match sample sizes for fair comparison
if len(prod) > len(ref):
    prod = prod.sample(len(ref))

print(f"\nReference data: {len(ref)} rows, {len(ref.columns)} columns")
print(f"Production data: {len(prod)} rows, {len(prod.columns)} columns")
print(f"Common columns: {len(feature_cols)}")

# Generate drift report using Evidently
print("\nGenerating drift report with Evidently AI...")
report = Report([DataDriftPreset()])
report.run(reference_data=ref, current_data=prod)
report.save_html(REPORT_PATH)

print(f"✅ HTML report saved: {REPORT_PATH}")

# Parse drift results
try:
    drift_result = report.as_dict()
    dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
    n_drifted = drift_result['metrics'][0]['result']['number_of_drifted_columns']
    total_cols = drift_result['metrics'][0]['result']['number_of_columns']
    drift_share = drift_result['metrics'][0]['result']['share_of_drifted_columns']
    
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total columns analyzed: {total_cols}")
    print(f"Drifted columns: {n_drifted}")
    print(f"Drift percentage: {drift_share * 100:.1f}%")
    
    # Show drifted columns (detailed info available in HTML report)
    if n_drifted > 0:
        print(f"\n📊 View detailed drift analysis in the HTML report")
        print(f"   Open: {REPORT_PATH}")
    
    print("\n" + "=" * 60)
    if dataset_drift:
        print("⚠️  DRIFT DETECTED — Consider retraining model")
        print(f"   {n_drifted} out of {total_cols} features show significant drift")
        print("=" * 60)
        
        # Uncomment to enable email alerts
        # send_email(REPORT_PATH, dataset_drift)
        
        exit(1)  # Exit code 1 for CI/CD alerts
    else:
        print("✅ No significant drift detected")
        print(f"   Only {n_drifted} out of {total_cols} features show drift")
        print("=" * 60)
        exit(0)
        
except (KeyError, IndexError, TypeError) as e:
    print(f"\n⚠️  Error parsing drift results: {e}")
    print("Check the HTML report for details.")
    import traceback
    traceback.print_exc()
    exit(0)
