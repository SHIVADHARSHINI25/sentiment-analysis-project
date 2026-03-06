import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

st.title("📋 Detailed Insights")
st.write("Tweet-level sentiment predictions, confidence scores, and explanations.")

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# -----------------------------
# Validation
# -----------------------------
if "processed_data" not in st.session_state:
    st.warning("Please upload and preprocess the data first.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.warning("Please train the model first from Model Analysis page.")
    st.stop()

# -----------------------------
# Load resources
# -----------------------------
df = st.session_state["processed_data"].copy()
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -----------------------------
# Predictions + confidence
# -----------------------------
X_vec = vectorizer.transform(df["clean_text"])
predictions = model.predict(X_vec)
probabilities = model.predict_proba(X_vec)

df["predicted_sentiment"] = predictions
df["confidence"] = np.max(probabilities, axis=1).round(3)

# -----------------------------
# Filters
# -----------------------------
st.subheader("🔎 Filters")

sentiment_filter = st.multiselect(
    "Filter by Predicted Sentiment",
    options=df["predicted_sentiment"].unique(),
    default=df["predicted_sentiment"].unique()
)

confidence_threshold = st.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

filtered_df = df[
    (df["predicted_sentiment"].isin(sentiment_filter)) &
    (df["confidence"] >= confidence_threshold)
]

# -----------------------------
# Main table
# -----------------------------
st.subheader("📊 Tweet-Level Predictions")

display_cols = [
    "clean_text",
    "airline_sentiment",
    "predicted_sentiment",
    "confidence"
]

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True
)

# -----------------------------
# Download as CSV
# -----------------------------
st.subheader("⬇️ Download Insights")

csv = filtered_df[display_cols].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="sentiment_detailed_insights.csv",
    mime="text/csv"
)

# -----------------------------
# Misclassification analysis
# -----------------------------
st.subheader("⚠️ Misclassified Tweets")

misclassified = filtered_df[
    filtered_df["airline_sentiment"] != filtered_df["predicted_sentiment"]
]

if misclassified.empty:
    st.success("No misclassifications found 🎉")
else:
    st.dataframe(
        misclassified[display_cols],
        use_container_width=True
    )

# -----------------------------
# Explainable AI (Top words)
# -----------------------------
st.subheader("🧠 Explainable AI – Influential Words")

if hasattr(model, "coef_"):
    feature_names = vectorizer.get_feature_names_out()
    class_labels = model.classes_

    selected_class = st.selectbox(
        "Select sentiment to explain",
        class_labels
    )

    class_index = list(class_labels).index(selected_class)
    coefficients = model.coef_[class_index]

    top_positive = pd.DataFrame({
        "word": feature_names,
        "weight": coefficients
    }).sort_values(by="weight", ascending=False).head(10)

    top_negative = pd.DataFrame({
        "word": feature_names,
        "weight": coefficients
    }).sort_values(by="weight", ascending=True).head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔼 Words pushing sentiment **towards** this class")
        st.dataframe(top_positive, use_container_width=True)

    with col2:
        st.markdown("### 🔽 Words pushing sentiment **away** from this class")
        st.dataframe(top_negative, use_container_width=True)
else:
    st.info("Explainability is available only for linear models.")
