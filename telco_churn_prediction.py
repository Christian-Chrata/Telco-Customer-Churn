import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
with open("tuned_logistic_regression.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📉 Telco Customer Churn Prediction")
st.write("Fill in customer information to predict churn probability.")

# Sidebar input form
def create_user_input():
    st.sidebar.header("Customer Information")

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
    cltv = st.sidebar.slider("Customer Lifetime Value (CLTV)", 0, 10000, 4000)
    satisfaction_score = st.sidebar.selectbox("Satisfaction Score (1-5)", [1, 2, 3, 4, 5])

    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    age_group = st.sidebar.selectbox("Age Group", ["Under 30", "Middle Age (30–59)", "Senior (≥ 60)"])

    user_input = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "CLTV": [cltv],
        "Satisfaction Score": [satisfaction_score],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "PaperlessBilling": [paperless_billing],
        "Partner": [partner],
        "MultipleLines": [multiple_lines],
        "AgeGroup": [age_group],
    })

    return user_input

# Get user input
data = create_user_input()

# Show input
st.subheader("Customer Data Preview")
st.write(data)

# Predict churn
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

# Display result
st.subheader("Churn Prediction Result")
if prediction == 1:
    st.error(f"❌ This customer is likely to churn. (Probability: {probability:.2%})")
else:
    st.success(f"✅ This customer is likely to stay. (Probability of churn: {probability:.2%})")