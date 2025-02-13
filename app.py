# import streamlit as st
# import pandas as pd
# import mlflow.sklearn
# import numpy as np

# # Load the best model from MLflow
# run_id = "c11d354cffbb4b6daf81c6dd3ad2c5bd"  # Replace with the actual run ID from MLflow
# model_uri = f"runs:/{run_id}/best_model"
# model = mlflow.sklearn.load_model(model_uri)
# #model = mlflow.sklearn.load_model("models:/FAANG_Best_Model/1")

# # Streamlit UI
# st.title("FAANG Stock Price Prediction")

# # User Inputs
# st.sidebar.header("Input Stock Features")
# open_price = st.sidebar.number_input("Open Price", min_value=0.0, format="%.2f")
# high_price = st.sidebar.number_input("High Price", min_value=0.0, format="%.2f")
# low_price = st.sidebar.number_input("Low Price", min_value=0.0, format="%.2f")
# volume = st.sidebar.number_input("Volume", min_value=0.0, format="%.0f")

# # Convert input to DataFrame
# input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], 
#                            columns=["Open", "High", "Low", "Volume"])

# # Predict
# if st.sidebar.button("Predict"):
#     prediction = model.predict(input_data)
#     st.write(f"### Predicted Close Price: ${prediction[0]:.2f}")






import pandas as pd
import joblib
import pandas as pd
import streamlit as st
# Load the saved column names
feature_columns = joblib.load('feature_columns.pkl')

# Load the model
model = joblib.load('best_model.pkl')

df=pd.read_csv("Final_Dataframe.csv")
# User input function
def user_input():
    open_price = st.sidebar.number_input("Open Price", value=df['Open'].mean(),step=100.00)
    high = st.sidebar.number_input("High Price", value=df['High'].mean(),step=100.00)
    low = st.sidebar.number_input("Low Price", value=df['Low'].mean(),step=100.00)
    volume = st.sidebar.number_input("Volume", value=df['Volume'].mean(),step=100.00)
    #stock = st.sidebar.selectbox("Stock", ['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google'])
    
    # Create a DataFrame with the same structure as the training data
    input_data = {
        'Open': [open_price],
        'High': [high],
        'Low': [low],
        'Volume': [volume],
        # 'Company_Facebook': [1 if stock == 'Facebook' else 0],
        # 'Company_Apple': [1 if stock == 'Apple' else 0],
        # 'Company_Amazon': [1 if stock == 'Amazon' else 0],
        # 'Company_Netflix': [1 if stock == 'Netflix' else 0],
        # 'Company_Google': [1 if stock == 'Google' else 0]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Ensure the columns match the training data
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    return input_df

# Get user input
input_df = user_input()

# Make prediction
prediction = model.predict(input_df)
st.subheader("Predicted Close Price")
st.write(prediction[0])