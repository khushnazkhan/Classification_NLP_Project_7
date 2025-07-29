# ⚠️ Disaster Tweet Classifier

> NLP Project 7 – Classify tweets as disaster or non-disaster using ML & DL models
> <img width="892" height="521" alt="distribution" src="https://github.com/user-attachments/assets/16da8f14-5e4c-425d-a873-9b6578d12e55" />


This project uses **Natural Language Processing (NLP)** to classify tweets as either related to a disaster or not. It includes both **Machine Learning (TF-IDF + Logistic Regression)** and **Deep Learning (LSTM)** models, deployed with an interactive **Streamlit web app**.

---

## 🚀 Project Overview

In the digital age, Twitter is often the first platform to reflect real-world events. This project helps identify disaster-related tweets to support early detection and response. The app is user-friendly, accurate, and deployable on cloud platforms.

---

## 📊 Dataset

- Total Tweets: ~10,000 (manually labeled)
- Label: 
  - `1` = Disaster
  - `0` = Not a Disaster
- Format: CSV or Pandas DataFrame
- Source: Public NLP dataset

---

## 🎯 Objectives

- Classify tweets as disaster or not
- Use both traditional ML and LSTM-based DL
- Build an interactive web app (Streamlit)
- Achieve high accuracy and confidence in predictions

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML**: TF-IDF + Logistic Regression
- **DL**: LSTM (Keras/TensorFlow)
- **Visualization**: Matplotlib, Seaborn

---

## 🔍 Features

- Real-time tweet classification
- Choose between ML or DL model
- Confidence score with bar graph
- Sample tweet buttons
- Model info section (expandable)

---

## 📈 Model Performance

### 🔹 Machine Learning (TF-IDF + Logistic Regression)
- Accuracy: 92.3%
- Precision: 89.5%
- Recall: 85.2%
- F1-Score: 87.3%

### 🔹 Deep Learning (LSTM)
- Embedding + Padding + Dense
- Input: Tokenized padded tweets
- Trained on: 10,000 tweets

---

## 💻 How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/khushnazkhan/Classification_NLP_Project_7.git
cd Classification_NLP_Project_7

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
<img width="1332" height="597" alt="dash 1" src="https://github.com/user-attachments/assets/7201fbc2-e872-4bfc-959b-f2c4fb2339df" />

