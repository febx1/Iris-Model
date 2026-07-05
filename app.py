import requests
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib


# Set page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("<h1 style='text-align: center; color: #FF5722; font-family: Arial, Helvetica, sans-serif;'>Iris Flower Prediction</h1>", unsafe_allow_html=True)
st.write("This app predicts the species of an Iris flower based on its sepal and petal measurements.")

with st.expander("About the App"):
    st.write("""
        This app uses a Random Forest Classifier to predict the species of an Iris flower.
        The model was trained on the famous Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for three species of Iris flowers: Setosa, Versicolor, and Virginica.
    """)

    st.markdown("<h2 style='text-align: center; color: #FF5722; font-family: Arial, Helvetica, sans-serif;'>Enter Flower Features</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 1.0, 7.0, 4.3, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 2.0, 5.0, 3.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)


input_data = [sepal_length, sepal_width, petal_length, petal_width]

FLASK_URL="http://127.0.0.1:5000/predict"

try:
    response = requests.post(FLASK_URL, json={"features": input_data})
    if response.status_code == 200:
        result = response.json()
        if 'error' in result:
            st.error(result['error'])
            st.stop()
        predicted_species = result['label']
        probabilities = list(result['probabilities'].values())
        class_names = list(result['probabilities'].keys()) 
    else:

        st.error("Error in prediction. Please try again.")
        st.stop()
except Exception as e:
    st.error(f"Error connecting to the prediction API: {e}")
    st.stop()

st.markdown("<h2 style='text-align: center; color: #FF5722; font-family: Arial, Helvetica, sans-serif;'>Prediction Result</h2>", unsafe_allow_html=True)
st.write(f"Predicted Species: **{predicted_species}**")


prob_df = pd.DataFrame({
    'Species':class_names, 
    'Probability':probabilities
    })

fig = px.bar(prob_df, 
             x='Species', y='Probability', 
             color='Species', 
             title="Prediction Probabilities", labels={'Probability': 'Probability', 'Species': 'Species'})
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)
