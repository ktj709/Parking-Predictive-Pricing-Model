import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Paths to model and encoders
MODEL_PATH = "parking_price_model_compressed.pkl"
ENCODER_PATH = "label_encoders_compressed.pkl"

# Set page config
st.set_page_config(page_title="Smart Parking Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Smart Parking Price Predictor")
st.markdown("This app predicts dynamic parking prices based on time, demand, and location data.")


# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        return model, encoders
    except Exception as e:
        st.error(f"Failed to load model or encoders: {str(e)}")
        return None, None


model, encoders = load_model_and_encoders()


# Function to predict price
def predict_price(features_dict):
    st.write("🛠 Features Sent to Model:", features_dict)  # Debugging
    df = pd.DataFrame([features_dict])

    # Encode categorical features using the encoders
    for col in ['location', 'duration_of_parking']:
        if col in df and col in encoders:
            try:
                # Ensure the feature is properly encoded
                df[col] = encoders[col].transform([features_dict[col]])[0]
            except ValueError:
                df[col] = 0
                st.warning(
                    f"Warning: {col} value '{features_dict[col]}' not seen during training. Using default encoding.")

    # Ensure that the features are passed in the same order as training data
    st.write("🧠 Final DataFrame passed to model:", df)  # Debugging
    return model.predict(df)[0]


if model is not None and encoders is not None:
    col1, col2 = st.columns(2)

    with col1:
        # Selecting inputs for location, duration, date, and time
        location = st.selectbox("Location", encoders['location'].classes_.tolist())
        duration = st.selectbox("Duration of Parking", encoders['duration_of_parking'].classes_.tolist())
        date = st.date_input("Date", value=datetime.date.today())
        time = st.time_input("Time", value=datetime.time(12, 0))

    with col2:
        # Selecting inputs for total slots, available slots, and demand level
        total_slots = st.number_input("Total Slots", min_value=1, max_value=1000, value=100)
        available_slots = st.number_input("Available Slots", min_value=0, max_value=total_slots, value=50)
        demand = st.slider("Demand Level", 0, 100, 60)

    if st.button("Predict Price"):
        # Constructing the feature dictionary
        features = {
            'day': date.day,
            'hour': time.hour,
            'location': location,
            'total_slots': total_slots,
            'available_slots': available_slots,
            'demand_level': demand,
            'duration_of_parking': duration
        }

        # Predicting the price
        price = predict_price(features)
        st.success(f"Estimated Parking Price: **${price:.2f}**")

        # Occupancy rate calculation
        occ_rate = (total_slots - available_slots) / total_slots * 100
        st.progress(min(occ_rate / 100, 1.0), text=f"Occupancy: {occ_rate:.1f}%")
else:
    st.stop()
