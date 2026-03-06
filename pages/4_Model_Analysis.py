import streamlit as st
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

st.title("🤖 Model & Analysis")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

if "processed_data" not in st.session_state:
    st.warning("Please complete preprocessing first.")
else:
    df = st.session_state["processed_data"]

    X = df["clean_text"]
    y = df["airline_sentiment"]

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_tfidf, y_train)

            y_pred = model.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Save model
            joblib.dump(model, "models/sentiment_model.pkl")
            joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

            # Store metrics
            st.session_state["model_metrics"] = {
                "accuracy": accuracy,
                "f1": f1
            }

        st.success("Model trained successfully 🎉")

        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")
