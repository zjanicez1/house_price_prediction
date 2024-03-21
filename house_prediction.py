import streamlit as st
from joblib import load
import numpy as np
import pandas as pd

# Load the model
model = load('linear_regression_model.joblib')

# Create a simple user input for single predictions
user_input = st.number_input('Enter house size (sqft):', min_value=100, max_value=10000, step=50)

# Reshape the input for the model and predict
if st.button('Predict Price for Single House'):
    input_array = np.array([user_input]).reshape(-1, 1)
    predicted_price = model.predict(input_array)
    st.write(f"The predicted house price is: ${predicted_price[0]:.2f}")

# File upload section
uploaded_file = st.file_uploader("Upload your input file (.txt or .xlsx):", type=['txt', 'xlsx'])

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.txt'):
                # Assuming txt file contains one house size per line
                data = np.loadtxt(uploaded_file, dtype=float).reshape(-1, 1)
            elif uploaded_file.name.endswith('.xlsx'):
                # Assuming Excel file has one column with header 'house_size'
                df = pd.read_excel(uploaded_file)
                data = df.values.reshape(-1, 1)
            else:
                st.error("Unsupported file type!")
                return

            # Predict prices
            predicted_prices = model.predict(data)
            # Round the predicted prices to 2 decimal places
            predicted_prices_rounded = np.round(predicted_prices, 2)
            # Create a DataFrame to display results, using the rounded prices
            result_df = pd.DataFrame({
                'House Size (sqft)': data.flatten(),
                'Predicted Price ($)': predicted_prices_rounded.flatten()
            })
            
            return result_df
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
    else:
        return

if uploaded_file is not None:
    result_df = process_uploaded_file(uploaded_file)
    if result_df is not None:
        st.write("Predicted Prices:")
        st.dataframe(result_df)
