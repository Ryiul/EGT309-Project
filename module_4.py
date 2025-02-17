import streamlit as st
import requests

# Function to call the model inference API
def get_prediction(features):
    #url = "http://<model-inference-service>:80/predict/"  # Replace with your model inference service URL
    url = "http://localhost:5000/predict/"
    try:
        # Send the features as a JSON request to the API
        response = requests.post(url)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.json()['prediction']
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Streamlit UI layout
st.title("Model Inference Web UI")

# Button to trigger prediction
if st.button("Get Prediction"):
    prediction = get_prediction()

    # Display the prediction
    if "error" in prediction:
        st.error(f"Error: {prediction['error']}")
    else:
        st.write("Prediction:", prediction)
        st.table([prediction])  # Optionally display in a table format