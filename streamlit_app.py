import streamlit as st
import joblib

st.set_page_config(page_title="Fake Job Predictor", page_icon="🕵️")
st.title("🕵️ Fake Job Posting Predictor")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")          # MultinomialNB
    vect  = joblib.load("vectorizer.pkl")     # CountVectorizer
    return model, vect

model, vect = load_artifacts()

txt = st.text_area("Paste a job description (title + details)", height=220, placeholder="e.g., Customer Service Associate...")

if st.button("Predict"):
    if not txt.strip():
        st.warning("Please paste some text.")
    else:
        X = vect.transform([txt])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]
        label = "Fraudulent" if pred == 1 else "Real"
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{proba:.2%}**")
