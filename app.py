import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict the price.")

med_inc = st.number_input("Median Income", value=5.0)
house_age = st.number_input("House Age", value=20.0)
ave_rooms = st.number_input("Average Rooms", value=5.0)
ave_bedrooms = st.number_input("Average Bedrooms", value=1.0)
population = st.number_input("Population", value=1000.0)
ave_occup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict House Price"):

    df = pd.DataFrame({
        "MedInc":[med_inc],
        "HouseAge":[house_age],
        "AveRooms":[ave_rooms],
        "AveBedrms":[ave_bedrooms],
        "Population":[population],
        "AveOccup":[ave_occup],
        "Latitude":[latitude],
        "Longitude":[longitude]
    })

    scaled = scaler.transform(df)
    poly_data = poly.transform(scaled)

    prediction = model.predict(poly_data)

# Prediction in USD
price_usd = prediction[0] * 100000

# USD to INR conversion (1 USD = ₹86, change if needed)
usd_to_inr = 86

price_inr = price_usd * usd_to_inr

st.success(f"🏡 Predicted House Price: ₹ {price_inr:,.2f}")
