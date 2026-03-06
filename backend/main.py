from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from utils.text_cleaning import clean_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.post("/predict")
def predict(text: str):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]
    sentiment = model.classes_[prob.argmax()]
    confidence = prob.max()
    return {
        "sentiment": sentiment,
        "confidence": float(confidence)
    }
