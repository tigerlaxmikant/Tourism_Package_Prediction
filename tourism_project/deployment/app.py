import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="/laxmikantdeshpande/tourism-package-prediction", filename="best_tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Product Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18.0, max_value=100.0, value=18, step=1)
TypeOfContact = st.selectbox("Type Of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0, max_value=50, value=6)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Large Business", "Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
NumberOfPersonVisiting = st.selectbox("Number Of Person Visiting", ["1", "2", "3", "4", "5"])
NumberOfFollowups = st.selectbox("Number Of Followups", ["1", "2", "3", "4", "5", "6"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "King", "Deluxe", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=10.0, value=1, step=1)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"])
OwnCar = st.selectbox("Own Car", ["1", "0"])
NumberOfChildrenVisiting = st.selectbox("Number Of Children Visiting", ["1", "2", "3"])
Designation = NumberOfChildrenVisiting = st.selectbox("Designation", ["AVP", "VP", "Executive", "Manager", "Senior Manager", "Regional Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=100, max_value=100000.0, value=20993, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeOfContact': TypeOfContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Product Prediction" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
