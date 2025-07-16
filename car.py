import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load the car data
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit UI
st.title("🚗 Car Price Predictor")
st.markdown("### Please enter car details below:")

# Dropdown for company
company = st.selectbox("Select Company", sorted(car['company'].unique()))

# Filter car models based on selected company
filtered_models = car[car['company'] == company]['name'].unique()
car_model = st.selectbox("Select Car Model", sorted(filtered_models))

# Other fields
year = st.selectbox("Select Manufacturing Year", sorted(car['year'].unique(), reverse=True))
fuel_type = st.selectbox("Select Fuel Type", car['fuel_type'].unique())
kms_driven = st.number_input("Enter Kilometers Driven", value=1000, step=100)

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {np.round(prediction, 2):,}")
