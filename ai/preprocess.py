from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", "", text)
    return text.strip()

def train_vectorizer(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, "ai/vectorizer.pkl")
    return X

def load_vectorizer():
    return joblib.load("ai/vectorizer.pkl")