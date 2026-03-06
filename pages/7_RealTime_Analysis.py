import streamlit as st
import joblib
import os

st.title("⚡ Real-Time Sentiment Analysis")

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Please train the model first.")
else:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    user_text = st.text_area(
        "Enter text to analyze sentiment",
        placeholder="Type a tweet or sentence here..."
    )

    if st.button("Analyze Sentiment"):
        if user_text.strip() == "":
            st.error("Please enter some text.")
        else:
            text_vector = vectorizer.transform([user_text])
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]

            st.subheader("🔎 Prediction Result")
            st.success(f"**Sentiment:** {prediction}")

            st.subheader("📊 Confidence Scores")
            for label, prob in zip(model.classes_, probabilities):
                st.progress(float(prob))
                st.write(f"{label}: {prob:.2f}")
