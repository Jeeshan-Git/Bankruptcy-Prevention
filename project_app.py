import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Display the image and app title
st.image("C:\\Data science\\Projects\\ExcelR.png")
st.title("Bankruptcy Prevention Project")

# Load the dataset
data = pd.read_excel("C:\\Data science\\Projects\\Bankruptcy (2).xlsx")

# Preprocessing: Encode the target variable
data['class'] = data['class'].replace({'non-bankruptcy': 0, 'bankruptcy': 1})

# Split data into features and target
X = data.drop('class', axis=1)
y = data['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Streamlit title and description
st.write("This app predicts the likelihood of bankruptcy based on financial risk factors.")

# User input features using sliders in Streamlit sidebar
industrial_risk = st.sidebar.slider("Industrial Risk (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)
management_risk = st.sidebar.slider("Management Risk (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)
financial_flexibility = st.sidebar.slider("Financial Flexibility (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)
credibility = st.sidebar.slider("Credibility (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)
competitiveness = st.sidebar.slider("Competitiveness (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)
operating_risk = st.sidebar.slider("Operating Risk (0: Low, 0.5: Medium, 1: High)", 0.0, 1.0, 0.5)

# Create a dataframe from user input
user_input = pd.DataFrame([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]],
                          columns=["industrial_risk", "management_risk", "financial_flexibility", "credibility", "competitiveness", "operating_risk"])

# Show the input data on the Streamlit interface
st.write("### Input Data:")
st.write(user_input)

# Make predictions
prediction = model.predict(user_input)[0]
prediction_proba = model.predict_proba(user_input)[0]

# Display the prediction
st.write("### Prediction:")
st.write("Bankruptcy" if prediction == 1 else "Non-Bankruptcy")

# Display prediction probabilities
st.write(f"Probability of Bankruptcy: {prediction_proba[1]:.2f}")
st.write(f"Probability of Non-Bankruptcy: {prediction_proba[0]:.2f}")
