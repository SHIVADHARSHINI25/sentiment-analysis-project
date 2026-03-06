import streamlit as st
import joblib
import os

st.title("🤖 Sentiment-Aware Chatbot")

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Check model availability
if not os.path.exists(MODEL_PATH):
    st.warning("Please train the model first from the Model Analysis page.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def bot_response(user_text, sentiment):
    if sentiment == "positive":
        return "😊 I'm glad you're feeling positive! Tell me more."
    elif sentiment == "negative":
        return "😔 I'm sorry to hear that. Want to talk about what went wrong?"
    else:
        return "😐 I see. Would you like to explain further?"

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # User message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Predict sentiment
    vector = vectorizer.transform([user_input])
    sentiment = model.predict(vector)[0]

    bot_reply = bot_response(user_input, sentiment)

    # Bot message
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": f"{bot_reply}\n\n**Detected sentiment:** `{sentiment}`"
        }
    )

    st.rerun()
