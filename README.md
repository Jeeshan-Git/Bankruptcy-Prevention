## Bankruptcy Prevention Project

This project is designed to predict the likelihood of bankruptcy for businesses using key financial and operational risk factors. It is a classification problem where the target variable is binary: bankruptcy or non-bankruptcy. The project leverages machine learning models and is deployed using Streamlit for user interaction.

## Table of Contents
1. Project Description
2. Dataset Details
3. Technologies Used
4. Steps Performed
5. Deployment
6. Result

## Project Description
The goal of this project is to model the probability of business bankruptcy using financial and operational risk factors. This model can assist stakeholders in identifying potential risks early.

## Dataset Details
The dataset contains 250 companies with the following 7 features:

* industrial_risk: 0 (Low), 0.5 (Medium), 1 (High)
* management_risk: 0 (Low), 0.5 (Medium), 1 (High)
* financial_flexibility: 0 (Low), 0.5 (Medium), 1 (High)
* credibility: 0 (Low), 0.5 (Medium), 1 (High)
* competitiveness: 0 (Low), 0.5 (Medium), 1 (High)
* operating_risk: 0 (Low), 0.5 (Medium), 1 (High)
* class: Target variable (0: Non-Bankruptcy, 1: Bankruptcy)

## Technologies Used
* Programming Language: Python
* Libraries:
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Model: Random Forest Classifier
* Deployment Tool: Streamlit

## Steps Performed
1. Exploratory Data Analysis (EDA):
Explored relationships between risk factors and the target variable.
Checked for missing values and handled data preprocessing.

2. Model Building:
Trained a Random Forest Classifier with balanced class weights.
Used an 80-20 split for training and testing datasets.

3. Model Evaluation:
Evaluated the model's performance using metrics like accuracy and probability predictions.

4. Deployment:
Developed a Streamlit application for real-time bankruptcy prediction based on user inputs.

## Deployment
The project is deployed as a Streamlit app, allowing users to interact with the model through sliders for input features. The app predicts the likelihood of bankruptcy and displays probabilities for both classes.

## Results
The app provides:

* A classification prediction: Bankruptcy or Non-Bankruptcy.
* Probabilities for each class to enhance interpretability.

