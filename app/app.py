import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/salary_predictor.pkl")
edu_enc = joblib.load("model/education_encoder.pkl")
role_enc = joblib.load("model/jobrole_encoder.pkl")


st.title("💼 Employee Salary Predictor")
st.markdown("Estimate an employee's salary based on experience, education, and job role.")


experience = st.slider("Years of Experience", 0, 30, 1)
education = st.selectbox("Education Level", edu_enc.classes_)
jobrole = st.selectbox("Job Role", role_enc.classes_)


if st.button("Predict Salary"):
    # Convert categorical inputs to encoded values
    edu_code = edu_enc.transform([education])[0]
    role_code = role_enc.transform([jobrole])[0]

    # Combine into input array
    features = np.array([[experience, edu_code, role_code]])
    prediction = model.predict(features)[0]

    # Show result
    st.success(f"💰 Estimated Salary: ₹{int(prediction):,} per month")
