
import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MODEL_PATH = "saved_model.pkl"

# Title
st.title("🇿🇼 Enhanced AI Bank Reconciliation & Classification System")

# Sidebar for file uploads and settings
st.sidebar.header("📂 Upload Files")
training_file = st.sidebar.file_uploader("Upload training_data.csv", type="csv")
bank_file = st.sidebar.file_uploader("Upload bank_statement.csv", type="csv")

threshold = st.sidebar.slider("🔧 Fuzzy Match Threshold", min_value=70, max_value=100, value=85, step=1)

# Load or train model
model = None

def train_model(df_train):
    X = df_train['Description']
    y = df_train['Label']
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Main logic
if training_file and bank_file:
    df_train = pd.read_csv(training_file)
    df_bank = pd.read_csv(bank_file)

    st.subheader("📘 Training Data Preview")
    st.dataframe(df_train.head())

    st.subheader("🏦 Bank Statement Preview")
    st.dataframe(df_bank.head())

    # Check for retraining
    if st.sidebar.button("🔁 Retrain Model"):
        model = train_model(df_train)
        st.sidebar.success("✅ Model retrained and saved.")
    else:
        model = load_model()
        if model is None:
            model = train_model(df_train)
            st.sidebar.warning("⚠️ No saved model found, trained a new one.")

    # Predict and visualize
    df_bank['Predicted_Label'] = model.predict(df_bank['Description'])

    st.subheader("📊 Predicted Transaction Categories")
    fig, ax = plt.subplots()
    sns.countplot(data=df_bank, x="Predicted_Label", palette="Set2", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Reconciliation
    reconciled = []
    unmatched = []

    for _, bank_row in df_bank.iterrows():
        best_score = 0
        best_match = None
        for _, train_row in df_train.iterrows():
            score = fuzz.token_sort_ratio(bank_row['Description'], train_row['Description'])
            if abs(bank_row['Amount'] - train_row['Amount']) < 5:
                score += 10
            if score > best_score:
                best_score = score
                best_match = train_row
        if best_score >= threshold:
            reconciled.append({
                "Bank_Date": bank_row['Date'],
                "Bank_Description": bank_row['Description'],
                "Bank_Amount": bank_row['Amount'],
                "Predicted_Label": bank_row['Predicted_Label'],
                "Matched_Internal_Desc": best_match['Description'],
                "Internal_Amount": best_match['Amount'],
                "Internal_Label": best_match['Label'],
                "Score": best_score
            })
        else:
            unmatched.append(bank_row)

    # Show reconciliation
    df_rec = pd.DataFrame(reconciled)
    df_unmatched = pd.DataFrame(unmatched)

    st.subheader("✅ Reconciled Transactions")
    st.dataframe(df_rec)
    st.download_button("📥 Download Reconciled", df_rec.to_csv(index=False), "reconciled.csv")

    st.subheader("❌ Unmatched Transactions")
    st.dataframe(df_unmatched)
    st.download_button("📥 Download Unmatched", df_unmatched.to_csv(index=False), "unmatched.csv")
else:
    st.info("👈 Please upload both training and bank files to continue.")
