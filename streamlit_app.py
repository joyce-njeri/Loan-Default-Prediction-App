import streamlit as st
import pickle
import numpy as np
import datetime as dt


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classifier_loaded = data["classifier"]
testing_attr = data["testing_attr"]
classifier_predict = data["predictor"]

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
    customers which is a local digital lending company in Nigeria. This model is 
    based on a collection of detailed attributes for every customer.
    """
)

st.write('Please enter customer attributes to predict their likelihood of defaulting on a loan:')

st.subheader('Profile')
f1=st.number_input('Age',1,100)
f2 = st.selectbox("Level of Education", level_of_education)
f3 = st.selectbox("Employment Status", employment_status)
f4 = st.selectbox("Type of bank Account", bank_account_types)

st.subheader('Previous Loan')
f5 = st.date_input("Date of Approval")
f6=st.number_input('Term Days',1,1000000)
f7 = st.date_input("Date of Closure")

st.subheader('Make Prediction')
# Prediction
ok = st.button('Predict Loan Default')

def usersdata(data, user_inputs):
    closeddate = user_inputs[0]
    approveddate = user_inputs[1]
    returndays = (closeddate - approveddate).days
    termdays = user_inputs[2] # user
    data.insert(2, (termdays - returndays)) # 3

    data.insert(3, termdays) # 4
    age = user_inputs[3]
    data.insert(15, age) # 16

    if user_inputs[4] == 'Other':
        data.insert(17, 1) # 18
        data.insert(18, 0) # 19
    else:
        data.insert(17, 0) # 18
        data.insert(18, 1) # 19

    if user_inputs[5] == 'Permanent': # 20
        data.insert(19, 1) # 20
        data.insert(20, 0) # 21
        data.insert(21, 0) # 22
        data.insert(22, 0) # 23
        data.insert(23, 0) # 24
    elif user_inputs[5] == 'Retired': # 21
        data.insert(19, 0) # 20
        data.insert(20, 1) # 21
        data.insert(21, 0) # 22
        data.insert(22, 0) # 23
        data.insert(23, 0) # 24
    elif user_inputs[5] == 'Self-Employed': # 22
        data.insert(19, 0) # 20
        data.insert(20, 0) # 21
        data.insert(21, 1) # 22
        data.insert(22, 0) # 23
        data.insert(23, 0) # 24
    elif user_inputs[5] == 'Student': # 23
        data.insert(19, 0) # 20
        data.insert(20, 0) # 21
        data.insert(21, 0) # 22
        data.insert(22, 1) # 23
        data.insert(23, 0) # 24
    else: # 24
        data.insert(19, 0) # 20
        data.insert(20, 0) # 21
        data.insert(21, 0) # 22
        data.insert(22, 0) # 23
        data.insert(23, 1) # 24

    if user_inputs[6] == 'Post-Graduate': # 25
        data.insert(24, 1) # 25
        data.insert(25, 0) # 26
        data.insert(26, 0) # 27
    elif user_inputs[6] == 'Primary': # 26
        data.insert(24, 0) # 25
        data.insert(25, 1) # 26
        data.insert(26, 0) # 27
    else: # 27
        data.insert(24, 0) # 25
        data.insert(25, 0) # 26
        data.insert(26, 1) # 27

    if age > 0 and age < 13: # 28
        data.insert(27, 1) # 28
        data.insert(28, 0) # 29
        data.insert(29, 0) # 30
    elif age > 13 and age < 31: # 29
        data.insert(27, 0) # 28
        data.insert(28, 1) # 29
        data.insert(29, 0) # 30
    elif age > 31 and age < 100: # 30
        data.insert(27, 0) # 28
        data.insert(28, 0) # 29
        data.insert(29, 1) # 30

    return data

if ok:

    user_inputs=[f7,f5,f6,f1,f4,f3,f2]

    X_test = usersdata(testing_attr, user_inputs)
    X_test = np.array(X_test).reshape(1,-1)

    def load_results(test_sub_X, algorithm):
        sub_prob = algorithm.predict_proba(test_sub_X)[:,1]
        sub_prob = "%.0f%%" % (100 * sub_prob)
        return sub_prob

    pred = load_results(X_test, classifier_predict)
    st.subheader(f"The probability of defaulting on a loan is {pred}")