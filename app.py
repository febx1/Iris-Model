import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.express as px    # Interactive visualization 
import requests 

# Page configuration 
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌼",
    layout="centered"
)

# CSS styling 
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom, #0A0A0A, #1A1A1A);
            color: #F5F5F5;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2 {
            color: #FFA500;
        }
        .stButton > button {
            background-color: #FFA500;
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title for the frontend web application 
st.markdown("<h1>🌼 Iris Flower Species Prediction</h1>",unsafe_allow_html=True)
st.write("Predict the species of an Iris flower based on its pyhsical characteristics using a train ML model served through Flask API.")

# Information box 
with st.expander("ℹ️ About this model", expanded=False):
    st.write("""
        This app uses a **Random Forest Classifier** trained on the classic Iris dataset.
        The model is served through a Flask REST API. Adjust the sliders below to enter flower measurements and receive real-time predictions.
    """)

# Introduce sliders in our front end 
st.markdown("## ➤ Enter Flower Meansurements")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    petal_length = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
with col2:
    sepal_width = st.slider("🌿 Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
    petal_width = st.slider("🌸 Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# Get data from Flask 
input_data = [sepal_length, sepal_width, petal_length, petal_width]

# URL where my flask app is running 
FLASK_URL = "https://iris-model-deployment-rhg9.onrender.com/predict"  # we will display the model predictions here 

# Generate predictions via Flask API 
# Exception handling 
try:
    response = requests.post(FLASK_URL, json={"features": input_data})

    if response.status_code == 200:  # 200 means "success"
        result = response.json()
        if 'error' in result:
            st.error(f"Flask API Error: {result['error']}")
            st.stop()

        predicted_species = result['label']
        probabilities = list(result['probabilities'].values())
        class_names = list(result['probabilities'].keys())
    else:
        st.error("❌ Failed to get prediction from Flask API.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Could not connect to Flask API.\n\nError: `{e}`")
    st.stop()

# Show the predicted values 
st.markdown("## 🧪 Prediction Result")
st.success(f"🌼 **Predicted Species:** {predicted_species}")

prob_df = pd.DataFrame({
    'Species': class_names,
    'Probability': probabilities
})

# Plotting bar chart for predictiona probability against flower species in our problem statememt 
fig = px.bar(
    prob_df,
    x='Species',
    y='Probability',
    color='Species',
    color_discrete_sequence=['#FFA500', '#FFB347', '#FF8C00'],
    title='Prediction Probability by Species',
    text=prob_df['Probability'].apply(lambda x: f"{x:.2%}")
)

# Designing the layout for our bar chart 
fig.update_layout(
    plot_bgcolor="#1A1A1A",
    paper_bgcolor="#0A0A0A",
    font=dict(color="#F5F5F5"),
    title_font=dict(size=20, color="#FFA500"),
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1]
)

# This is to make the bar chart interactive 
fig.update_traces(textposition='outside')

# Display the bar chart 
st.plotly_chart(fig, use_container_width=True)

# Footer 
st.markdown("---")  # Seperator between 2 sections of the frontend 

