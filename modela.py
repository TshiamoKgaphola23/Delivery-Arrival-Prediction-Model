import streamlit as st
import joblib
import pandas as pd
import os

# Load the serialized model
pipeline = joblib.load("pipeline.pkl")

# Set page config
st.set_page_config(page_title="Arrival Time Prediction", page_icon=":chart_with_upwards_trend:")

background_image_url ="data:image/jpeg;base64,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"

background_css = f"""
<style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)


def time_to_seconds_24h(time_str):
    if not time_str:
        return None
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# title
st.markdown(f"<h1 style='font-size: 45px; text-align: center;'>Arrival Time Prediction</h1>", unsafe_allow_html=True)

#sidebar
if st.sidebar.button("Home", key="Home_button"):
    os.system("streamlit run /Users/damacm1143/Downloads/home_page.py")
    pass
elif st.sidebar.button("Overview", key="overview_button"):
    os.system("streamlit run /Users/damacm1143/Downloads/sendy_streamlit.py")



col1,col2,col3,col4,col5 = st.columns([1,2,2,2 ,1])
with col2:
    vehicle_type = st.radio("Select Vehicle Type", ["Car", "Bike"])
    platform_type = st.number_input("Select Platform Type", min_value=1, max_value=2, step=1, value=1)
    personal_or_business = st.radio("Select Personal or Business", ["Personal", "Business"])
    distance_km = st.number_input("Enter Distance (KM)", min_value=0.0, max_value=100.0, step=1.0, value=1.0)
    temperature = st.number_input("Enter Temperature", min_value=-10.0, max_value=50.0, step=1.0, value=20.0)
    
with col4:
    pickup_day = st.slider("Enter Pickup - Day of Month", min_value=1, max_value=31, step=1, value=1)
    pickup_weekday = st.slider("Enter Pickup - Weekday (Mo = 1)", min_value=1, max_value=7, step=1, value=1)
    pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, step=1, value=7)
    pickup_minute = st.slider("Pickup Minute", min_value=0, max_value=59, step=1, value=0)
    pickup_second = st.slider("Pickup Second", min_value=0, max_value=59, step=1, value=0)







pickup_time_str = f"{pickup_hour}:{pickup_minute}:{pickup_second}"

if st.button(" ⏰ Predict"):
    pickup_time_seconds = time_to_seconds_24h(pickup_time_str)
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
    user_input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(user_input_df)
    predicted_time = seconds_to_time(prediction[0])
    st.markdown(f"<p style='font-size: 50px;'>Predicted Arrival at Destination Time: {predicted_time}</p>", unsafe_allow_html=True)
    st.balloons()

# Button to go to another page
if st.button("Overview"):
    # Redirect to the other page by running it as a subprocess
    os.system("streamlit run /Users/damacm1143/Downloads/sendy_streamlit.py")
