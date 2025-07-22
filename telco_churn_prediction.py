# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Telco Customer Churn Prediction')
st.text('Predict your Customer Lifetime Value for this Automotive Insurance Company')



# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    
    # Numerical Features
    Number_of_Policies = st.sidebar.slider('Number of Policies', min_value=1, max_value=9, value=2)
    Monthly_Premium_Auto = st.sidebar.slider('Monthly Premium Auto', min_value=61, max_value=297, value=82)
    Total_Claim_Amount = st.sidebar.slider('Total Claim Amount', min_value=0.4, max_value=2759.7, value=374.4)
    Income = st.sidebar.slider('Income', min_value=0, max_value=99934, value=34322)
    

    # Categorical Features
    Vehicle_Class = st.sidebar.radio('Vehicle Class', ['Four-Door Car', 'Two-Door Car', 'SUV', 'Sports Car', 'Luxury SUV','Luxury Car'])
    Coverage = st.sidebar.radio('Coverage', ['Extended', 'Basic' ,'Premium'])
    Renew_Offer_Type=st.sidebar.radio('Renew Offer Type', ['Offer1', 'Offer3', 'Offer2', 'Offer4'])
    EmploymentStatus = st.sidebar.radio('EmploymentStatus', ['Retired', 'Employed', 'Disabled', 'Medical Leave', 'Unemployed'])
    Marital_Status = st.sidebar.radio('Marital Status', ['Divorced', 'Married', 'Single'])
    Education = st.sidebar.radio('Education', ['High School or Below', 'College', 'Master', 'Bachelor', 'Doctor'])

    # Creating a dictionary with user input
    user_data = {
        'Number of Policies': Number_of_Policies,
        'Monthly Premium Auto': Monthly_Premium_Auto,
        'Total Claim Amount': Total_Claim_Amount,
        'Income': Income,
        'Vehicle Class': Vehicle_Class,
        'Coverage': Coverage,
        'Renew Offer Type':Renew_Offer_Type,
        'EmploymentStatus':EmploymentStatus,
        'Marital Status':Marital_Status,
        'Education':Education
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'tuned_logistic_regression.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
# probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    # if kelas == 1:
    #     st.write('Class 1: This customer will Survive')
    # else:
    #     st.write('Class 2: This customer will Survive')
    
    # Displaying the probability of the customer buying
    st.write(f"Your CLV prediction (USD): {kelas[0]:.2f}")  # Probability of class 1 (BUY)
