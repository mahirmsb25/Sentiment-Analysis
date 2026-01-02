import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline


DATA_PATH = "data/book_reviews_sample.csv"

data = pd.read_csv(DATA_PATH)
print(data.head(), "\n")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


data["reviewText_clean"] = data["reviewText"].apply(clean_text)
print(data["reviewText_clean"].iloc[0], "\n")


vader = SentimentIntensityAnalyzer()

data["vader_sentiment_score"] = data["reviewText_clean"].apply(
    lambda review: vader.polarity_scores(review)["compound"]
)

bins = [-1, -0.1, 0.1, 1]
labels = ["negative", "neutral", "positive"]

data["vader_sentiment_label"] = pd.cut(
    data["vader_sentiment_score"],
    bins=bins,
    labels=labels
)

data["vader_sentiment_label"].value_counts().plot(kind="bar")
plt.show()


transformer_pipeline = pipeline(
    "sentiment-analysis",
    truncation=True
)

transformer_results = transformer_pipeline(
    data["reviewText_clean"].tolist(),
    batch_size=16
)

data["transformers_sentiment_label"] = [
    result["label"] for result in transformer_results
]

data["transformers_sentiment_label"].value_counts().plot(kind="bar")
plt.show()


comparison_table = pd.crosstab(
    data["vader_sentiment_label"],
    data["transformers_sentiment_label"]
)

print(comparison_table)

comparison_table.plot(kind="bar", figsize=(8, 5))
plt.show()
