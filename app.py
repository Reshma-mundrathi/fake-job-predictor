import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Fake Job Predictor", page_icon="🕵️")
st.title("🕵️ Fake Job Posting Predictor")

REAL = [
    "customer service associate handle inbound queries adhere attendance policy computer software",
    "onsite mechanical engineer cad experience benefits health insurance relocation paid",
    "lab technician hplc gc documentation gmp regulated environment shift based schedule",
]
FAKE = [
    "work from home data entry earn 6000 week no interview immediate joining",
    "send id and bank details to proceed selected without resume urgent requirement",
    "payment upfront bitcoin limited slots click link verify account now",
]
X_train = REAL + FAKE
y_train = [0]*len(REAL) + [1]*len(FAKE)

vect = CountVectorizer(stop_words="english").fit(X_train)
clf  = MultinomialNB().fit(vect.transform(X_train), y_train)

txt = st.text_area("Paste a job description (title + details)", height=220)
if st.button("Predict"):
    if not txt.strip():
        st.warning("Please paste some text.")
    else:
        X = vect.transform([txt])
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0][pred]
        st.success(f"Prediction: **{'Fraudulent' if pred==1 else 'Real'}**")
        st.write(f"Confidence: **{proba:.2%}**")

