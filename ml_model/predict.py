import os
import pickle

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_pipeline.pkl")

# Load pipeline once
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)


def predict_news(text):
    prediction = pipeline.predict([text])
    probability = pipeline.predict_proba([text])

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        label = "Real News"
    else:
        label = "Fake News"

    return label, round(confidence, 2)
