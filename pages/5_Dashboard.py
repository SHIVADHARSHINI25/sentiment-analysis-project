import streamlit as st
import plotly.express as px

st.title("📊 Sentiment Analysis Dashboard")

if "processed_data" not in st.session_state:
    st.warning("Please upload and preprocess data first.")
else:
    df = st.session_state["processed_data"]

    st.subheader("🔍 Dataset Overview")
    st.write(df.head())

    # -------------------------------
    # Sentiment Distribution
    # -------------------------------
    st.subheader("📈 Sentiment Distribution")

    sentiment_count = df["airline_sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]

    fig = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Sentiment Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Percentage View
    # -------------------------------
    st.subheader("📊 Sentiment Percentage")

    fig2 = px.pie(
        sentiment_count,
        names="Sentiment",
        values="Count",
        title="Sentiment Percentage"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Text Length Analysis
    # -------------------------------
    st.subheader("📝 Tweet Length Analysis")

    df["tweet_length"] = df["clean_text"].apply(len)

    fig3 = px.histogram(
        df,
        x="tweet_length",
        nbins=50,
        title="Distribution of Tweet Lengths"
    )
    st.plotly_chart(fig3, use_container_width=True)
