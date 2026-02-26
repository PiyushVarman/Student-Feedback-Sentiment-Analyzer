from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
analyzer=SentimentIntensityAnalyzer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackRequest(BaseModel):
    score: int
    feedback: str


def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hybrid_sentiment(original_text: str,cleaned_text: str):
    scores = analyzer.polarity_scores(original_text)
    compound = scores["compound"]

    # If strong sentiment, trust VADER
    if compound >= 0.6:
        return "positive"
    elif compound <= -0.6:
        return "negative"
    elif -0.4 < compound < 0.4:
        return "neutral"

    # If weak/uncertain sentiment → use ML model
    vec = vectorizer.transform([cleaned_text])
    return model.predict(vec)[0]

def credibility_score(score, sentiment):
    score_norm = score / 50

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    sent_value = sentiment_map[sentiment]

    if score_norm >= 0.75:
        expected = 1
    elif score_norm >= 0.45:
        expected = 0
    else:
        expected = -1

    credibility = 1 - abs(expected - sent_value) / 2
    return round(credibility, 2)


@app.post("/predict")
def predict(data: FeedbackRequest):

    if len(data.feedback.strip()) < 3:
        return {"error": "Feedback too short. Please enter a meaningful sentence."}

    cleaned = clean_text(data.feedback)

    prediction = hybrid_sentiment(data.feedback,cleaned)
    
    credibility = credibility_score(data.score, prediction)

    if credibility >= 0.7:
        flag = "Genuine"
    elif credibility >= 0.4:
        flag = "Suspicious"
    else:
        flag = "Highly Inconsistent"

    return {
        "sentiment": prediction,
        "credibility_score": credibility,
        "flag": flag
    }
