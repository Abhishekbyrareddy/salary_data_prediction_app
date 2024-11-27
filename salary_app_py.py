#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import os

def main():
    st.title("Salary Prediction App")

    # Load the pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), 'C:\Users\abhishek\OneDrive\Desktop\streamlit files\Test 1\salary_model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    # Input fields for user to provide values
    st.header("Provide the following details:")
    
    age = st.number_input("Age", min_value=18, max_value=65, step=1, format="%d")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    education = st.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])
    job_title = st.text_input("Job Title")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1, format="%d")

    # Preprocessing inputs
    if st.button("Predict Salary"):
        try:
            # Encode categorical features
            gender_encoded = 0 if gender == "Male" else 1
            education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            education_encoded = education_mapping.get(education, 0)
            
            # Input transformation
            input_data = np.array([[age, gender_encoded, education_encoded, len(job_title), experience]])
            
            # Prediction
            prediction = model.predict(input_data)
            st.success(f"Predicted Salary: ${round(prediction[0], 2)}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()

