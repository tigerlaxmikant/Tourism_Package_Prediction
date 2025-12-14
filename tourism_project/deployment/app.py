%%writefile tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, create_repo
import os
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="laxmikantdeshpande/tourism_project_model",
    filename="best_tourism_project_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Product Prediction App")
st.write("""
This application predicts whether a customer is likely to take a tourism product package.
Please enter the customer details below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
TypeOfContact = st.selectbox("Type Of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0, max_value=50, value=6)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Large Business", "Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.selectbox("Number Of Person Visiting", ["1", "2", "3", "4", "5"])
NumberOfFollowups = st.selectbox("Number Of Followups", ["1", "2", "3", "4", "5", "6"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "King", "Deluxe", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=10, value=1, step=1)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"])
OwnCar = st.selectbox("Own Car", ["0", "1"])
NumberOfChildrenVisiting = st.selectbox("Number Of Children Visiting", ["0", "1", "2", "3"])
Designation = st.selectbox("Designation", ["AVP", "VP", "Executive", "Manager", "Senior Manager", "Regional Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=100, max_value=100000, value=20000, step=100)

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
    result = "Package Taken" if prediction == 1 else "Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")




hf_token = os.getenv("HF_TOKEN")
api = HfApi()

space_id = "laxmikantdeshpande/tourism_project_model"

# Create the Space if missing
try:
    api.repo_info(repo_id=space_id, repo_type="space", token=hf_token)
    print(f"Space '{space_id}' already exists.")
except Exception:
    create_repo(repo_id=space_id, repo_type="space", space_sdk="streamlit", private=False, token=hf_token)
    print(f"Space '{space_id}' created.")

# Upload app.py and requirements.txt
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=space_id,
    repo_type="space",
    token=hf_token,
)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=space_id,
    repo_type="space",
    token=hf_token,
)
