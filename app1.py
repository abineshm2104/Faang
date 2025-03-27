import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle
import os


# Load the saved column names

def fetchFeatureColumn(CompanyName):
    feature_columns = joblib.load(f'feature_columns_{CompanyName}.pkl')
 
    return feature_columns
def LoadModel(CompanyName):
    # Load the model
    
    model = joblib.load(f'{CompanyName}_best_model.pkl')
    return model
def ScalerModel(CompanyName):
    # Load the model
    
    model = joblib.load(f'standard_scaler_{CompanyName}.pkl')
    return model
# Streamlit UI for the app
st.title("Close Price Prediction App")
st.write("Enter stock features to predict the Close Price:")

# Sidebar for user inputs
st.sidebar.header("Stock Input Features")

open_price = st.sidebar.slider("Open Price", min_value=None, max_value=500000000, value=None, step=None,label_visibility="visible")
high_price = st.sidebar.slider("High Price", min_value=None, max_value=500000000, value=None, step=None,label_visibility="visible")
low_price = st.sidebar.slider("Low Price", min_value=None, max_value=500000000, value=None, step=None,label_visibility="visible")
volume = st.sidebar.slider("Volume", min_value=None, max_value=500000000, value=None, step=None,label_visibility="visible")
Company = st.sidebar.selectbox("Company", ['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google'])


# Prediction Button
if st.sidebar.button("Predict Close Price"):
    # Create input data array
    user_input = np.array([[open_price, high_price, low_price,volume]])

    # Scale input data
    user_input_scaled = ScalerModel(Company).transform(user_input)

    # Make prediction
    prediction = LoadModel(Company).predict(user_input_scaled)

    # Display the result
    st.subheader("Predicted Closing Price")
    st.write(f"${prediction[0]:.2f}")
