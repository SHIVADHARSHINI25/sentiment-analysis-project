import streamlit as st

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main {
            padding: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Sentiment Analysis Dashboard")
st.subheader("Analyze customer opinions using AI & Machine Learning")

st.info(
    "Use the sidebar to navigate through the application pages."
)
