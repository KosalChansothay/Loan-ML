import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 4px;
        transition-duration: 0.4s;
    }
    .stButton button:hover {
        background-color: green;
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }
    .stTitle h1 {
        color: #4CAF50;
    }
    .stSubheader h2 {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)


# Define the Streamlit app
st.title("Loan Eligibility Prediction by G4")

# Collect user input
def user_input_features():
    gender = st.selectbox('Gender', ('Male', 'Female'))
    married = st.selectbox('Married', ('Yes', 'No'))
    dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
    education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = st.selectbox('Self Employed', ('Yes', 'No'))
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term (Days)')
    credit_history = st.selectbox('Credit History', ('0', '1'))
    property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))
    
    data = {'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Preprocess user input
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Encode categorical features
label_encoders = {
    'Gender': LabelEncoder().fit(['Male', 'Female']),
    'Married': LabelEncoder().fit(['Yes', 'No']),
    'Education': LabelEncoder().fit(['Graduate', 'Not Graduate']),
    'Self_Employed': LabelEncoder().fit(['Yes', 'No']),
    'Property_Area': LabelEncoder().fit(['Urban', 'Semiurban', 'Rural'])
}

for column, encoder in label_encoders.items():
    df[column + "_Encode"] = encoder.transform(df[column])

# Apply log transformation to numerical features
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for column in numerical_columns:
    df[column + '_log'] = np.log(df[column] + 1)  # Adding 1 to avoid log(0)

# Drop original columns
df = df.drop(columns=numerical_columns + list(label_encoders.keys()))

# Ensure the order of columns matches the training data
expected_columns = ['Credit_History',
                    'ApplicantIncome_log','CoapplicantIncome_log', 'LoanAmount_log', 'Loan_Amount_Term_log']

df = df[expected_columns]

# Button to make prediction
if st.button('Predict'):
    # Scaling
    df_scaled = scaler.transform(df)

    # Make predictions
    prediction = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)

    st.subheader('Prediction')
    st.write('Congratulation, Your proposal has been approved' if prediction[0] == 1 else 'Unfortunately, Your proposal has been rejected')

    #st.subheader('Prediction Probability')
    #st.write(f'Probability of being approved is: {prediction_proba[0][1]:.2f}')

