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
X = REAL + FAKE
y = [0]*len(REAL) + [1]*len(FAKE)

vect = CountVectorizer(stop_words="english").fit(X)
clf  = MultinomialNB().fit(vect.transform(X), y)

txt = st.text_area("Paste a job description", height=200)
if st.button("Predict") and txt.strip():
    P = vect.transform([txt])
    pred = clf.predict(P)[0]
    proba = clf.predict_proba(P)[0][pred]
    st.success(f"Prediction: **{'Fraudulent' if pred==1 else 'Real'}**")
    st.write(f"Confidence: **{proba:.2%}**")


