import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Load the trained model
joblib_file = "model.pkl"
model = joblib.load(joblib_file)

st.title("Stacking Regressor Prediction")

st.write("This app predicts the arrival time using a pre-trained Stacking Regressor model.")

# Define input fields
def user_input_features():
    distance = st.number_input("Distance (KM)", value=0.0)
    placement_time = st.time_input("Placement Time")
    pickup_time = st.time_input("Pickup Time")
    
    # Convert input times to datetime
    placement_time = pd.to_datetime(placement_time, format='%H:%M:%S').time()
    pickup_time = pd.to_datetime(pickup_time, format='%H:%M:%S').time()
    
    today = pd.Timestamp.today().normalize()
    placement_time = pd.Timestamp.combine(today, placement_time)
    pickup_time = pd.Timestamp.combine(today, pickup_time)
    
    return pd.DataFrame({
        'Distance (KM)': [distance],
        'Placement - Time': [placement_time],
        'Pickup - Time': [pickup_time]
    })

# Get user input
input_data = user_input_features()

# Convert datetime columns to numerical (timestamps)
label_encoder = preprocessing.LabelEncoder() 

input_data['Placement - Time']= label_encoder.fit_transform(input_data['Placement - Time']) 
input_data['Pickup - Time']= label_encoder.fit_transform(input_data['Pickup - Time']) 

if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_data)
    predicted_arrival_time = pd.to_datetime(prediction[0], unit='s')
    
    # Convert input pickup time back to datetime
    pickup_time = pd.to_datetime(input_data['Pickup - Time'][0], unit='s')
    
    # Calculate time differences
    #time_diff = predicted_arrival_time - pickup_time
    #time_diff_seconds = time_diff.total_seconds()
    #time_diff_minutes = time_diff_seconds / 60
    #time_diff_hours = time_diff_minutes / 60
    
    # Display the results
    st.write(f"Predicted Arrival Time: {predicted_arrival_time.time()}")
    #st.write(f"Time from Pickup to Arrival: {int(time_diff_seconds)} seconds ({time_diff_minutes:.2f} minutes, {time_diff_hours:.2f} hours)")
