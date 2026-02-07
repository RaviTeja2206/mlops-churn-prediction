"""
Streamlit dashboard for Customer Churn Prediction
Interactive interface for business users
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Title
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### Predict customer churn using ML model trained on AWS SageMaker")

# Sidebar - API Status
with st.sidebar:
    st.header("⚙️ System Status")
    
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Online")
            
            # Model info
            model_info = requests.get(f"{API_URL}/model/info").json()
            st.metric("Model Accuracy", f"{model_info['accuracy']:.1%}")
            st.metric("F1-Score", f"{model_info['f1_score']:.3f}")
            st.metric("Training Cost", model_info['training_cost'])
            
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ Cannot connect to API")
        st.info("Start API: `uvicorn api.main:app --reload`")

    st.markdown("---")
    st.markdown("**Model Details:**")
    st.markdown(f"- Platform: AWS SageMaker")
    st.markdown(f"- Algorithm: XGBoost")
    st.markdown(f"- Cost Savings: 69% (spot instances)")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📋 Batch Prediction", "📈 Analytics"])

# TAB 1: Single Customer Prediction
with tab1:
    st.header("Single Customer Churn Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=100.0)
        
        col1a, col1b = st.columns(2)
        with col1a:
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            partner = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col1b:
            dependents = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        st.subheader("Services & Contract")
        
        gender = st.radio("Gender", ["Male", "Female"])
        gender_male = 1 if gender == "Male" else 0
        
        phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        internet_service = st.selectbox("Internet Service", ["None", "DSL", "Fiber optic"])
        internet_fiber = 1 if internet_service == "Fiber optic" else 0
        internet_no = 1 if internet_service == "None" else 0
        
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        contract_one_year = 1 if contract_type == "One year" else 0
        contract_two_year = 1 if contract_type == "Two year" else 0
        
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        payment_electronic = 1 if payment_method == "Electronic check" else 0
        payment_mailed = 1 if payment_method == "Mailed check" else 0
        payment_credit = 1 if payment_method == "Credit card" else 0
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Prepare request data
        customer_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "PaperlessBilling": paperless_billing,
            "gender_Male": gender_male,
            "MultipleLines_No_phone_service": 0,
            "MultipleLines_Yes": 0,
            "InternetService_Fiber_optic": internet_fiber,
            "InternetService_No": internet_no,
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
            "Contract_One_year": contract_one_year,
            "Contract_Two_year": contract_two_year,
            "PaymentMethod_Credit_card_automatic": payment_credit,
            "PaymentMethod_Electronic_check": payment_electronic,
            "PaymentMethod_Mailed_check": payment_mailed
        }
        
        try:
            # Make prediction
            with st.spinner("Analyzing customer data..."):
                response = requests.post(f"{API_URL}/predict", json=customer_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Risk", result['churn_prediction'], 
                             delta=f"{result['churn_probability']:.1%}")
                
                with col2:
                    st.metric("Confidence Level", result['confidence'])
                
                with col3:
                    risk_level = "🔴 High" if result['churn_probability'] > 0.7 else \
                                "🟡 Medium" if result['churn_probability'] > 0.4 else "🟢 Low"
                    st.metric("Risk Level", risk_level)
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['churn_probability'] * 100,
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.info(f"**💡 Recommendation:** {result['recommendation']}")
                
            else:
                st.error(f"❌ Prediction failed: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# TAB 2: Batch Prediction
with tab2:
    st.header("Batch Customer Churn Prediction")
    st.markdown("Upload a CSV file with customer data for bulk predictions")
    
    # Sample data template
    with st.expander("📥 Download Sample Template"):
        sample_data = pd.DataFrame({
            'tenure': [12, 48, 3],
            'MonthlyCharges': [70.0, 65.0, 85.0],
            'TotalCharges': [840.0, 3120.0, 255.0],
            'Contract_Type': ['Month-to-month', 'Two year', 'Month-to-month']
        })
        st.dataframe(sample_data)
        st.download_button(
            "Download Template",
            sample_data.to_csv(index=False),
            "customer_template.csv",
            "text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} customers")
        st.dataframe(df.head())
        
        if st.button("Predict All", type="primary"):
            st.info("Batch prediction feature coming soon!")

# TAB 3: Analytics
with tab3:
    st.header("Model Performance Analytics")
    
    # Confusion matrix (mock data for demo)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.816, 0.65, 0.55, 0.611]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Value', 
                     title="Model Performance Metrics",
                     color='Value',
                     color_continuous_scale='blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance (Top 10)")
        importance_df = pd.DataFrame({
            'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Contract Type',
                       'Internet Service', 'Payment Method', 'Senior Citizen', 
                       'Partner', 'Dependents', 'Phone Service'],
            'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02, 0.02]
        })
        
        fig = px.bar(importance_df.sort_values('Importance', ascending=True), 
                     x='Importance', y='Feature',
                     orientation='h',
                     title="Feature Importance",
                     color='Importance',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit • Model trained on AWS SageMaker • Cost: $0.0006</p>
</div>
""", unsafe_allow_html=True)
