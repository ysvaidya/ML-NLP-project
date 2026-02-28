import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from ml_model.text_processing import clean_text


# ==============================
# Load Dataset
# ==============================

true_data = pd.read_csv(r"C:\Users\mauli\Downloads\True.csv")
fake_data = pd.read_csv(r"C:\Users\mauli\Downloads\Fake.csv")


# Remove last 10 rows for manual testing
fake_data = fake_data.iloc[:-10]
true_data = true_data.iloc[:-10]


# Add labels
fake_data["Label"] = 0
true_data["Label"] = 1


# Combine datasets
data = pd.concat([fake_data, true_data])

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Create full content column
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["Label"]

# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Build Pipeline
# ==============================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        max_df=0.7
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train
pipeline.fit(X_train, y_train)



# Evaluate
# ==============================

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Save Pipeline
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_pipeline.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully at:", MODEL_PATH)
