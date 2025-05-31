import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

# Title
st.title("🇿🇼 AI Bank Reconciliation & Classification System")

# Upload section
st.sidebar.header("📂 Upload CSV Files")
training_file = st.sidebar.file_uploader("Upload training_data.csv", type="csv")
bank_file = st.sidebar.file_uploader("Upload bank_statement.csv", type="csv")

# Proceed only if both files are uploaded
if training_file and bank_file:
    df_train = pd.read_csv(training_file)
    df_bank = pd.read_csv(bank_file)

    st.subheader("Preview: Internal Training Data")
    st.dataframe(df_train.head())

    st.subheader("Preview: Bank Statement")
    st.dataframe(df_bank.head())

    # Train model
    X = df_train['Description']
    y = df_train['Label']

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)

    # Predict bank labels
    df_bank['Predicted_Label'] = model.predict(df_bank['Description'])

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
        if best_score >= 85:
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

    df_rec = pd.DataFrame(reconciled)
    df_unmatched = pd.DataFrame(unmatched)

    # --- Accuracy Calculation ---
    if not df_rec.empty:
        label_accuracy = (df_rec['Predicted_Label'] == df_rec['Internal_Label']).mean() * 100
        desc_accuracy = (df_rec['Bank_Description'] == df_rec['Matched_Internal_Desc']).mean() * 100
        amount_accuracy = (abs(df_rec['Bank_Amount'] - df_rec['Internal_Amount']) <= 5).mean() * 100
        overall_accuracy = (
            (df_rec['Predicted_Label'] == df_rec['Internal_Label']) &
            (df_rec['Bank_Description'] == df_rec['Matched_Internal_Desc']) &
            (abs(df_rec['Bank_Amount'] - df_rec['Internal_Amount']) <= 5)
        ).mean() * 100

        st.subheader("📊 Reconciliation Accuracy")
        st.metric("✅ Label Accuracy", f"{label_accuracy:.2f}%")
        st.metric("✅ Description Accuracy", f"{desc_accuracy:.2f}%")
        st.metric("✅ Amount Match Accuracy (±5)", f"{amount_accuracy:.2f}%")
        st.metric("✅ Overall Reconciliation Accuracy", f"{overall_accuracy:.2f}%")
    else:
        st.warning("No reconciled transactions found. Cannot compute accuracy.")

    # Show results
    st.subheader("🔁 Reconciled Transactions")
    st.dataframe(df_rec)
    st.download_button("📥 Download Reconciled Transactions", df_rec.to_csv(index=False), file_name="reconciled_transactions.csv")

    st.subheader("❌ Unmatched Bank Transactions")
    st.dataframe(df_unmatched)
    st.download_button("📥 Download Unmatched Transactions", df_unmatched.to_csv(index=False), file_name="unmatched_transactions.csv")

else:
    st.info("👈 Upload both training and bank CSV files to begin.")
