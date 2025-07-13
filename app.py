import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("personality_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Personality Predictor", layout="centered")
st.title("🧠 Personality Prediction App")
st.markdown("Rate each of the following behaviors from 1 to 10 (except friends circle which goes up to 20):")

# Input features with sliders
time_alone = st.slider("1. Time Spent Alone", 1, 10, 5)
stage_fear = st.slider("2. Stage Fear", 1, 10, 5)
social_events = st.slider("3. Social Event Attendance", 1, 10, 5)
going_out = st.slider("4. Going Outside", 1, 10, 5)
drained_socializing = st.slider("5. Drained After Socializing", 1, 10, 5)
friends_circle = st.slider("6. Friends Circle Size", 1, 20, 10)
post_frequency = st.slider("7. Post Frequency on Social Media", 1, 10, 5)

# Prepare input data
input_data = pd.DataFrame([{
    "Time_spent_Alone": time_alone,
    "Stage_fear": stage_fear,
    "Social_event_attendance": social_events,
    "Going_outside": going_out,
    "Drained_after_socializing": drained_socializing,
    "Friends_circle_size": friends_circle,
    "Post_frequency": post_frequency
}])

# Predict when button clicked
if st.button("🔍 Predict Personality"):
    try:
        prediction = model.predict(input_data)[0]
        personality = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🎯 Predicted Personality: **{personality}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
