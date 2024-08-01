# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the serialized model
pipeline = joblib.load("pipeline.pkl")

# Function to convert time in 24-hour format to seconds since midnight
def time_to_seconds_24h(time_str):
    if not time_str:
        return None  # Return None if the input string is empty
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds

# Function to convert seconds since midnight to HH:MM:SS format
def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# Streamlit web application
st.title("Arrival Time Prediction")

# User input fields
vehicle_type = st.selectbox("Enter Vehicle Type", ["Car", "Bike"])
platform_type = st.number_input("Enter Platform Type", min_value=1, max_value=2, value=1)
personal_or_business = st.selectbox("Enter Personal or Business", ["Personal", "Business"])
pickup_day = st.number_input("Enter Pickup - Day of Month", min_value=1, max_value=31, value=1)
pickup_weekday = st.number_input("Enter Pickup - Weekday (Mo = 1)", min_value=1, max_value=7, value=1)
pickup_time = st.text_input("Enter Pickup - Time (HH:MM:SS 24-hour format): ")
distance_km = st.number_input("Enter Distance (KM): ", min_value=0.0, value=1.0)
temperature = st.number_input("Enter Temperature: ", min_value=-10.0, max_value=50.0, value=20.0)

# Button to trigger prediction
if st.button("Predict"):
    if not pickup_time:
        st.write("Please enter a valid pickup time.")
    else:
        # Convert user input to seconds since midnight
        pickup_time_seconds = time_to_seconds_24h(pickup_time)

        # Prepare input data
        user_input = {
            "Vehicle Type": vehicle_type,
            "Platform Type": platform_type,
            "Personal or Business": personal_or_business,
            "Pickup - Day of Month": pickup_day,
            "Pickup - Weekday (Mo = 1)": pickup_weekday,
            "Pickup - Time(Seconds since midnight)": pickup_time_seconds,
            "Distance (KM)": distance_km,
            "Temperature": temperature
        }

        # Create a DataFrame for the input data
        user_input_df = pd.DataFrame([user_input])

        # Predict using the model
        prediction = pipeline.predict(user_input_df)

        # Convert predicted time in seconds since midnight to HH:MM:SS format
        predicted_time = seconds_to_time(prediction[0])

        # Display predicted arrival time
        st.write("Predicted Arrival at Destination Time (24-hour format):", predicted_time)