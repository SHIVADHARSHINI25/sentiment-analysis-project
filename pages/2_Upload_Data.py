import streamlit as st
import pandas as pd

st.title("📤 Upload Data")

uploaded_file = st.file_uploader(
    "Upload Twitter Dataset (CSV)", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Store dataset in session
    st.session_state["raw_data"] = df

    st.success("Dataset uploaded successfully!")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Info")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
