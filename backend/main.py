from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

# Allow frontend
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


# Improved rule-based sentiment
def rule_based_sentiment(text: str):

    negative_words = ["bad", "worst", "waste", "terrible", "poor", "boring", "useless", "awful"]
    positive_words = ["good", "excellent", "amazing", "great", "best", "wonderful", "awesome"]
    neutral_words = ["ok","okay", "fine", "average", "moderate", "satisfactory"]

    # Handle negation case
    if "not bad" in text:
        return "neutral"

    pos_found = 0
    neg_found = 0
    neutral_found=0

    # Count negative matches
    for word in negative_words:
        if re.search(rf"\b{word}\b", text):
            neg_found += 1

    # Count positive matches
    for word in positive_words:
        if re.search(rf"\b{word}\b", text):
            pos_found += 1

    for word in neutral_words:
        if re.search(rf"\b{word}\b", text):
            neutral_found += 1

    # If both exist -> mixed sentiment -> neutral
    if (pos_found > 0 and neg_found > 0) or (neutral_found > pos_found and neutral_found > neg_found):
        return "neutral"

    # Only negative words
    if neg_found > 0:
        return "negative"

    # Only positive words
    if pos_found > 0:
        return "positive"

    return None  # no rule match


def credibility_score(score, sentiment):
    score_norm = score / 50

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    sent_value = sentiment_map[sentiment]

    if score_norm >= 0.7:
        expected = 1
    elif score_norm >= 0.4:
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

    # Rule-based first
    rule_sent = rule_based_sentiment(cleaned)

    if rule_sent is not None:
        prediction = rule_sent
    else:
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

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
