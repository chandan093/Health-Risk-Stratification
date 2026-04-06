import streamlit as st
import pandas as pd
import joblib
model = joblib.load('risk_model.ipynb') # Load the trained model
st.title("Healthcare Risk stratification App")  
age = st.number_input("Age", min_value=0)
Length_of_stay = st.number_input("Length of Stay (days)", min_value=0)
Treatment_cost = st.number_input("Treatment Cost ($)", min_value=0.0)
AbnormalLabCount = st.number_input("Abnormal Lab Count", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'LengthOfStay': [Length_of_stay],
        'TreatmentCost': [Treatment_cost],
        'AbnormalLabCount': [AbnormalLabCount]  
    })
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Risk Prediction: {'High Risk' if prediction == 1 else 'Low'}")
    st.write(f"Probability: {round(probability, 2)}")

    