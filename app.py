import streamlit as st
import pandas as pd
import joblib
import time

model = joblib.load("best_model.pkl")
pipeline = joblib.load("pipeline.pkl")
categorical_cols = ['workclass','education', 'marital-status', 'occupation', 
                   'relationship', 'race', 'gender', 'native-country']

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 38)
workclass=st.sidebar.selectbox("Work Class", [
    "Private","Self-emp-not-inc","Local-gov","Others","State-gov","Self-emp-inc","Federal-gov"
])
education = st.sidebar.selectbox("Education Level", [
    "HS-grad","Some-college","Bachelors","Masters","Assoc-voc","11th","Assoc-acdm","10th","7th-8th","Prof-school","9th","12th","Doctorate",
    "5th-6th","1st-4th","Preschool"
])
maritalStatus = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse","Never-married","Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse" 
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
relationship= st.sidebar.selectbox("Relationship", [
    "Husband","Not-in-family","Own-child","Unmarried","Wife","Other-relative"
])
race=st.sidebar.selectbox("Race", [
    "White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"
])
gender=st.sidebar.selectbox("Gender", [
    "Male","Female"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 100, 40)
nativeCountry=st.sidebar.selectbox("Native Country", [
    "United-States",
"Mexico","Others","Philippines","Germany","Puerto-Rico","Canada","El-Salvador","India","Cuba","England","China","South","Jamaica","Italy","Dominican-Republic",
"Japan","Guatemala","Poland","Vietnam","Columbia","Haiti","Portugal","Taiwan","Iran","Nicaragua","Greece","Peru","Ecuador","France","Ireland",
"Thailand","Hong","Cambodia","Trinadad&Tobago","Laos","Outlying-US(Guam-USVI-etc)","Yugoslavia","Scotland","Honduras","Hungary","Holand-Netherlands"
])
net_capital=st.sidebar.slider("Net Capital", 0, 10000, 100000)

input_df = pd.DataFrame({
    'age': [age],
    'workclass' : [workclass],
    'education': [education],
    'marital-status': [maritalStatus],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [nativeCountry],
    'net_capital': [net_capital]
})

st.write("### 🔎 Input Data")
st.write(input_df)

input_df=pipeline.transform(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    placeholder = st.empty()
    placeholder.progress(0, "Wait for it...")
    time.sleep(1)
    placeholder.progress(50, "Wait for it...")
    time.sleep(1)
    placeholder.progress(100, "Wait for it...")
    time.sleep(1)
    st.success(f"✅ Prediction: {prediction[0]}")
    placeholder.empty()

st.write("#### Made by Sayan")
