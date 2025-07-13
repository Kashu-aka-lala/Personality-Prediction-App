# Personality-Prediction-App
Personality Prediction App built with Python, Streamlit, and XGBoost. Predicts personality types based on behavioral ratings like social habits, stage fear, and time spent alone.

 
👉 Interactive UI with sliders 🎯 ML-powered predictions 📊 Trained on structured psychological data

# 🧠 Personality Prediction App

A machine learning-based web app that predicts a user's personality type based on behavioral traits using a trained XGBoost model.

This project was built using:
- Python
- XGBoost
- Scikit-learn
- Streamlit

## 🔍 Description

This app allows users to input behavioral traits such as how often they socialize, go out, feel drained after socializing, and more. Based on these inputs, the model predicts their likely personality category.

The model is trained on a structured dataset with the following features:
- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`

### Input Format
- All inputs are ratings from 1–10, except for `Friends_circle_size`, which can be from 1–20.

