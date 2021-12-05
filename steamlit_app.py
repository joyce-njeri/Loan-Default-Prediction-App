import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classifier_loaded = data["classifier"]
trainer = data["trainer"]
user_inputs = []
helper_function = data["helper_function"]
classifier_predict = data["predictor"]
results = data["results"]

level_of_education = (
    "Post-Graduate",
    "Primary",
    "Secondary",
)

employment_status = (
    "Permanent",
    "Retired",
    "Self-Employed",
    "Student",
    "Unemployed",
)

bank_account_types = (
    "Other",
    "Savings",
)

st.write(
    """
    # Loan Default Prediction
    A Machine Learning App that predicts Loan Defaults of SuperLender
    customers, that is a local digital lending company in Nigeria, 
    based on a collection of detailed attributes for every customer.
    """
)

st.write('Please enter customer attributes to predict their likelihood of defaulting on a loan:')

st.subheader('Profile')
f1=st.number_input('birth date',16,45)
f2 = st.selectbox("Level of Education", level_of_education)
f3 = st.selectbox("Employment Status", employment_status)
f4 = st.selectbox("Type of bank Account", bank_account_types)

st.subheader('Previous Loan')
f5=st.number_input('Date of Approval',16,45)
f6=st.number_input('Term Days',16,45)
f7=st.number_input('Date of Closure',16,45)

# Prediction
ok = st.button('Predict Loan Default')

if ok:

    user_inputs=[f7,f5,f6,f1,f4,f3,f2]
    X_test = helper_function(trainer, user_inputs)

    pred = results(X_test, classifier_predict)
    st.subheader(f"The probability of defaulting on a loan is {pred}")