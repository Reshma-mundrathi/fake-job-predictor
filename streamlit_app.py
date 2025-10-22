import streamlit as st
import os, joblib, pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fake Job Predictor", page_icon="🕵️")
st.title("🕵️ Fake Job Posting Predictor")

MODEL_F = Path("model.pkl")
VECT_F  = Path("vectorizer.pkl")

@st.cache_resource
def load_or_train():
    # Try loading artifacts
    if MODEL_F.exists() and VECT_F.exists():
        model = joblib.load(MODEL_F)
        vect  = joblib.load(VECT_F)
        return model, vect, "loaded"

    # Fallback: train once (needs CSV in data/fake_job_postings.csv)
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    csv_path = Path("data/fake_job_postings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "Artifacts not found and dataset missing. "
            "Add model.pkl/vectorizer.pkl or place CSV at data/fake_job_postings.csv"
        )

    df = pd.read_csv(csv_path)
    cols = ["title","location","company_profile","description","requirements","benefits","industry"]
    for c in cols: df[c] = df[c].fillna("")
    df["text"] = df[cols].agg(" ".join, axis=1).str.lower()
    X, y = df["text"], df["fraudulent"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    vect = CountVectorizer().fit(Xtr)
    Xtr_dtm = vect.transform(Xtr)

    model = MultinomialNB().fit(Xtr_dtm, ytr)

    # Save for future runs
    joblib.dump(model, MODEL_F)
    joblib.dump(vect, VECT_F)
    return model, vect, "trained"

model, vect, source = load_or_train()
st.caption(f"Artifacts {source} successfully.")

txt = st.text_area("Paste a job description (title + details)", height=220,
                   placeholder="e.g., Customer Service Associate... attendance policy...")

if st.button("Predict"):
    if not txt.strip():
        st.warning("Please paste some text.")
    else:
        X = vect.transform([txt])
        pred = model.predict(X)[0]
        proba = getattr(model, "predict_proba", lambda _ : [[1,1]])(X)[0][pred]
        label = "Fraudulent" if pred == 1 else "Real"
        st.success(f"Prediction: **{label}**")
        if isinstance(proba, (float, int)):
            st.write(f"Confidence: **{proba:.2%}**")
