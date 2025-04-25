import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Parking Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Smart Parking Price Predictor")
st.markdown("""
This app predicts the optimal parking price based on location, time, and demand factors.
""")

# Paths to your pre-loaded model and encoder files in PyCharm
MODEL_PATH = "parking_price_model.pkl"
ENCODER_PATH = "label_encoders.pkl"


# Function to load the model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders, True
    except Exception as e:
        st.error(f"Error loading model or encoders: {str(e)}")
        return None, None, False


# Load the model and encoders when the app starts
model, encoders, load_success = load_model_and_encoders()


# Function to make predictions
def predict_price(features):
    # Convert categorical features using the encoders
    features_df = pd.DataFrame([features])

    # Encode categorical features
    if 'location' in features and 'location' in encoders:
        try:
            features_df['location'] = encoders['location'].transform([features['location']])[0]
        except ValueError:
            st.error(f"Location '{features['location']}' not found in training data. Using default value.")
            features_df['location'] = 0

    if 'duration_of_parking' in features and 'duration_of_parking' in encoders:
        try:
            features_df['duration_of_parking'] = \
            encoders['duration_of_parking'].transform([features['duration_of_parking']])[0]
        except ValueError:
            st.error(f"Duration '{features['duration_of_parking']}' not found in training data. Using default value.")
            features_df['duration_of_parking'] = 0

    # Make prediction
    predicted_price = model.predict(features_df)[0]
    return predicted_price


# Main interface
tab1, tab2, tab3 = st.tabs(["Price Prediction", "Data Explorer", "Model Information"])

with tab1:
    st.header("Predict Parking Price")

    if load_success:
        # Create columns for inputs
        col1, col2 = st.columns(2)

        with col1:
            # Get location options from encoder
            location_options = list(encoders['location'].classes_) if 'location' in encoders else ["Downtown", "Uptown",
                                                                                                   "Midtown"]
            selected_location = st.selectbox("Location", options=location_options)

            # Get duration options from encoder
            duration_options = list(
                encoders['duration_of_parking'].classes_) if 'duration_of_parking' in encoders else ["1 hour",
                                                                                                     "2 hours",
                                                                                                     "All day"]
            selected_duration = st.selectbox("Duration of Parking", options=duration_options)

            # Date and time picker
            selected_date = st.date_input("Date", value=datetime.date.today())
            selected_time = st.time_input("Time", value=datetime.time(12, 0))

        with col2:
            total_slots = st.number_input("Total Parking Slots", min_value=1, max_value=1000, value=100)
            available_slots = st.number_input("Available Slots", min_value=0, max_value=total_slots, value=50)
            demand_level = st.slider("Demand Level", min_value=0, max_value=100, value=70)

        # Create features dictionary for prediction
        features = {
            'day': selected_date.day,
            'hour': selected_time.hour,
            'location': selected_location,
            'total_slots': total_slots,
            'available_slots': available_slots,
            'demand_level': demand_level,
            'duration_of_parking': selected_duration
        }

        # Predict button
        if st.button("Predict Price"):
            with st.spinner("Calculating optimal price..."):
                predicted_price = predict_price(features)

                # Display the result
                st.success(f"### Recommended Parking Price: ${predicted_price:.2f}")

                # Display occupancy rate
                occupancy_rate = (total_slots - available_slots) / total_slots * 100
                st.info(f"Current Occupancy Rate: {occupancy_rate:.1f}%")

                # Create a gauge chart for occupancy
                occupancy_color = "green" if occupancy_rate < 70 else "orange" if occupancy_rate < 90 else "red"
                st.progress(min(occupancy_rate / 100, 1.0), text=f"Occupancy: {occupancy_rate:.1f}%")

                # Factors influencing the price
                st.subheader("Pricing Factors:")
                factors_col1, factors_col2 = st.columns(2)

                with factors_col1:
                    st.write("📊 **Location impact:** High" if selected_location in ["Downtown",
                                                                                    "City Center"] else "📊 **Location impact:** Medium")
                    st.write(f"🕒 **Time impact:** {'High' if 7 <= selected_time.hour <= 19 else 'Low'}")

                with factors_col2:
                    st.write(
                        f"🚗 **Demand impact:** {'High' if demand_level > 70 else 'Medium' if demand_level > 40 else 'Low'}")
                    st.write(f"📅 **Day impact:** {'High' if selected_date.weekday() < 5 else 'Low'}")
    else:
        st.error("Failed to load the model and encoders. Please check the file paths.")

with tab2:
    st.header("Data Explorer")
    st.write("Upload a CSV file to explore parking data:")

    data_file = st.file_uploader("Upload parking data (CSV)", type=['csv'])

    if data_file:
        try:
            data = pd.read_csv(data_file)
            st.write(f"Data shape: {data.shape[0]} rows, {data.shape[1]} columns")

            # Data overview
            st.subheader("Data Overview")
            st.dataframe(data.head())

            # Basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(data.describe())

            # Column selection for visualization
            if not data.empty:
                st.subheader("Data Visualization")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select column for histogram", options=numeric_cols)
                        st.bar_chart(data[selected_col])

                with viz_col2:
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("X-axis", options=numeric_cols, index=0)
                        y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols) - 1))
                        st.scatter_chart(data=data, x=x_col, y=y_col)

        except Exception as e:
            st.error(f"Error processing data file: {str(e)}")
    else:
        st.info("Upload a CSV file to explore your parking data.")

        # Sample data visualization
        st.subheader("Sample Data Visualization")
        sample_data = pd.DataFrame({
            'hour': list(range(24)),
            'avg_price': [2.5, 2.3, 2.0, 1.8, 1.5, 2.0, 3.0, 4.5, 5.0, 4.8, 4.5,
                          4.3, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 5.0, 4.5, 4.0, 3.5, 3.0, 2.8]
        })
        st.line_chart(sample_data, x='hour', y='avg_price')
        st.caption("Sample hourly price variation")

with tab3:
    st.header("Model Information")

    if load_success:
        # Display model type
        st.write(f"**Model Type:** {type(model).__name__}")

        # Display model parameters
        st.subheader("Model Parameters")
        model_params = model.get_params()
        st.json(model_params)

        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")

            # Get feature importance
            importances = model.feature_importances_
            feature_names = ['day', 'hour', 'location', 'total_slots', 'available_slots', 'demand_level',
                             'duration_of_parking']

            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            # Display as DataFrame and chart
            st.dataframe(importance_df)
            st.bar_chart(importance_df.set_index('Feature'))

        # Display encoder information
        st.subheader("Label Encoders")
        for name, encoder in encoders.items():
            st.write(f"**{name}**: {len(encoder.classes_)} unique values")
            st.write(f"Classes: {', '.join(encoder.classes_[:10])}" + ("..." if len(encoder.classes_) > 10 else ""))
    else:
        st.error("Failed to load the model information.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Smart Parking Price Predictor v1.0")
st.sidebar.caption("© 2025 Parking Solutions Inc.")

# Add sidebar options for additional features
st.sidebar.header("Options")
show_advanced = st.sidebar.checkbox("Show Advanced Options", False)

if show_advanced:
    st.sidebar.subheader("Advanced Settings")

    model_confidence = st.sidebar.slider(
        "Model Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        help="Adjust the confidence threshold for price recommendations"
    )

    price_range = st.sidebar.slider(
        "Price Range Adjustment (±$)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Adjust the allowed range for price variations"
    )