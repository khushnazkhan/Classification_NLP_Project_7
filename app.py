import streamlit as st
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load models and vectorizers

@st.cache_resource
def load_artifacts():
    ml_model = joblib.load("disaster_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    dl_model = load_model("disaster_lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return ml_model, tfidf_vectorizer, dl_model, tokenizer

ml_model, tfidf_vectorizer, dl_model, tokenizer = load_artifacts()

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="⚠️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>⚠️ Disaster Tweet Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Check whether a tweet is about a real disaster using ML & LSTM models.</div>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Input Area
# ----------------------------
default_text = st.session_state.get("tweet", "")
tweet_input = st.text_area("Enter Tweet", default_text, placeholder="e.g., 'Flood in Mumbai after heavy rain'", height=100)

model_choice = st.radio("Choose Model", ["Machine Learning (TF-IDF + Logistic Regression)", "Deep Learning (LSTM)"], horizontal=True)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict"):
    if not tweet_input.strip():
        st.warning("Please enter a tweet.")
    else:
        if model_choice.startswith("Machine"):
            X = tfidf_vectorizer.transform([tweet_input])
            prediction = ml_model.predict(X)[0]
            proba = ml_model.predict_proba(X)[0]
        else:
            seq = tokenizer.texts_to_sequences([tweet_input])
            padded = pad_sequences(seq, maxlen=100)
            proba_dl = dl_model.predict(padded)[0][0]
            prediction = int(proba_dl >= 0.5)
            proba = [1 - proba_dl, proba_dl]

        label = "🚨 Disaster" if prediction == 1 else "✅ Not a Disaster"
        color = "#ffe6e6" if prediction == 1 else "#e6ffe6"
        border = "red" if prediction == 1 else "green"
        confidence = proba[prediction] * 100

        st.markdown(f"""
            <div style="background-color: {color}; border-left: 5px solid {border}; padding: 15px; border-radius: 5px;">
                <h3>{label}</h3>
                <p>This tweet {'does' if prediction == 1 else 'does not'} appear to be about a disaster.</p>
                <strong>Confidence: {confidence:.2f}%</strong>
            </div>
        """, unsafe_allow_html=True)

        # ----------------------------
        # Confidence Bar Plot
        # ----------------------------
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.barplot(x=["Non-Disaster", "Disaster"], y=proba, palette=["#4CAF50", "#f44336"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        for i, v in enumerate(proba):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

# ----------------------------
# Sample Tweets
# ----------------------------
st.markdown("---")
st.subheader("📌 Try Sample Tweets")
sample_tweets = [
    "Forest fire near La Ronge Sask. Canada",
    "I'm enjoying a sunny day at the beach!",
    "Building collapse in downtown district, multiple casualties reported",
    "New iPhone launched today. Amazing camera!",
    "Flood rescue operation underway in Chennai.",
    "Earthquake measuring 6.3 magnitude strikes Indonesia"
]

sample_cols = st.columns(len(sample_tweets))
for i, tweet in enumerate(sample_tweets):
    if sample_cols[i].button(f"Sample {i+1}"):
        st.session_state.tweet = tweet
        st.experimental_rerun = st.rerun if hasattr(st, "rerun") else lambda: None
        st.experimental_rerun()

# ----------------------------
# Model Info
# ----------------------------
st.markdown("---")
with st.expander("📊 Model & Project Information"):
    st.markdown("""
    **Machine Learning Model**  
    - Technique: TF-IDF + Logistic Regression  
    - Accuracy: 92.3%  
    - Precision: 89.5%  
    - Recall: 85.2%  
    - F1 Score: 87.3%

    **Deep Learning Model**  
    - Architecture: LSTM with embedding & padding  
    - Trained on: 10,000 labeled tweets  
    - Input Shape: Padded sequences (maxlen=100)

    _This app uses both ML and LSTM models for Disaster Tweet Classification (Project 7)._
    """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr><center style='color:gray;'>Disaster Tweet Classifier | July 2025</center>", unsafe_allow_html=True)
