import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# Load dataset
df = pd.read_csv("../dataset/clean_feedback.csv")


# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["feedback"] = df["feedback"].apply(clean_text)

# Remove very short feedback (optional)
df = df[df["feedback"].str.len() > 3]


# -------- DATASET CLEANING (IMPORTANT) -------- #
negative_words = ["bad", "worst", "waste", "terrible", "poor", "boring", "useless", "awful"]

# Remove rows where sentiment is positive but contains strong negative words
df = df[~((df["sentiment"] == "positive") & (df["feedback"].str.contains("|".join(negative_words), na=False)))]

# Remove rows where sentiment is negative but contains strong positive words
positive_words = ["excellent", "amazing", "very good", "good", "best", "wonderful", "great"]

df = df[~((df["sentiment"] == "negative") & (df["feedback"].str.contains("|".join(positive_words), na=False)))]

# ---------------------------------------------- #


X = df["feedback"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Custom stopwords
custom_stopwords = list(ENGLISH_STOP_WORDS) + ["course", "subject"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    max_features=8000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Cleaned Naive Bayes Model saved as model.pkl")
print("✅ Vectorizer saved as vectorizer.pkl")
