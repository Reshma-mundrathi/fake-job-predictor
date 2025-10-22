import streamlit as st
import pickle
import pandas as pd

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🧠 Fake Job Posting Predictor")

st.write("Enter job details below to check if it's real or fake:")

# Input fields
title = st.text_input("Job Title")
location = st.text_input("Location")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")

if st.button("Predict"):
    # Convert inputs to a DataFrame (adjust based on your model)
    input_data = pd.DataFrame([[title, location, company_profile, description, requirements]],
                              columns=["title", "location", "company_profile", "description", "requirements"])
    
    # Get prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠ This job posting seems *Fake*.")
    else:
        st.success("✅ This job posting seems *Real*.")
