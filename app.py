import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Real Estate Property Tier Prediction",
    page_icon="🏠",
    layout="centered"
)
    
st.title("🏠 Real Estate Property Tier Prediction")
st.write("Enter property details to predict the property tier: Low, Medium, or High.")

# -----------------------------
# Load saved ML files
# -----------------------------
try:
    model = joblib.load("real_estate_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error(
        "Model files are missing. Please keep these files in the same folder as app.py:\n"
        "real_estate_model.pkl, label_encoders.pkl, scaler.pkl"
    )
    st.stop()

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2020, step=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)

    locality_options = list(label_encoders["Locality"].classes_)
    locality = st.selectbox("Locality", locality_options)

    property_options = list(label_encoders["Property"].classes_)
    property_type = st.selectbox("Property Type", property_options)

with col2:
    estimated_value = st.number_input(
        "Estimated Value", min_value=0.0, value=200000.0, step=10000.0
    )
    sale_price = st.number_input(
        "Sale Price", min_value=0.0, value=300000.0, step=10000.0
    )
    residential_options = list(label_encoders["Residential"].classes_)
    residential = st.selectbox("Residential Type", residential_options)

    face_options = list(label_encoders["Face"].classes_)
    face = st.selectbox("Face", face_options)

num_rooms = st.number_input("Number of Rooms", min_value=0, max_value=50, value=4, step=1)
num_bathrooms = st.number_input(
    "Number of Bathrooms", min_value=0, max_value=50, value=2, step=1
)
carpet_area = st.number_input(
    "Carpet Area", min_value=0.0, value=1200.0, step=50.0
)
property_tax_rate = st.number_input(
    "Property Tax Rate", min_value=0.0, value=1.05, step=0.01
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Property Tier", use_container_width=True):

    # Create input DataFrame using the same features as the notebook.
    input_df = pd.DataFrame([{
        "Year": year,
        "Locality": locality,
        "Estimated Value": estimated_value,
        "Sale Price": sale_price,
        "Property": property_type,
        "Residential": residential,
        "num_rooms": num_rooms,
        "num_bathrooms": num_bathrooms,
        "carpet_area": carpet_area,
        "property_tax_rate": property_tax_rate,
        "Face": face,
        "Day": day,
        "Month": month
    }])

    # Encode categorical columns using the encoders saved from the notebook.
    categorical_cols = ["Locality", "Property", "Residential", "Face"]

    try:
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Scale the same numerical columns used during training.
    numerical_cols = [
        "Estimated Value",
        "Sale Price",
        "num_rooms",
        "num_bathrooms",
        "carpet_area",
        "property_tax_rate"
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Match the exact feature order used by the trained model.
    expected_columns = [
        "Year",
        "Locality",
        "Estimated Value",
        "Sale Price",
        "Property",
        "Residential",
        "num_rooms",
        "num_bathrooms",
        "carpet_area",
        "property_tax_rate",
        "Face",
        "Day",
        "Month"
    ]

    input_df = input_df[expected_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Property Tier: **{prediction}**")

    # Optional explanation
    if str(prediction).lower() == "low":
        st.info("This property is classified in the Low price tier.")
    elif str(prediction).lower() == "medium":
        st.info("This property is classified in the Medium price tier.")
    elif str(prediction).lower() == "high":
        st.info("This property is classified in the High price tier.")
