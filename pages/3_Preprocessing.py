import streamlit as st
from utils.text_cleaning import clean_text

st.title("🧹 Data Preprocessing")

if "raw_data" not in st.session_state:
    st.warning("Please upload a dataset first.")
else:
    df = st.session_state["raw_data"]

    st.markdown("### Text Column: `text`")
    st.markdown("### Target Column: `airline_sentiment`")

    if st.button("Run Preprocessing"):
        with st.spinner("Cleaning text data..."):
            df["clean_text"] = df["text"].astype(str).apply(clean_text)

            st.session_state["processed_data"] = df

        st.success("Text preprocessing completed!")

        st.markdown("### Cleaned Text Preview")
        st.dataframe(df[["text", "clean_text"]].head())
        # df.to_csv("processed_data.csv", index=False)

