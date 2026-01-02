Book Review Sentiment Analysis (VADER vs Transformers)
A comparative sentiment analysis of book reviews using lexicon-based (VADER) and transformer-based models to examine accuracy–efficiency trade-offs in sentiment classification.

📊 Overview

This project analyzes sentiment in book review text using two fundamentally different approaches:

VADER (Valence Aware Dictionary and sEntiment Reasoner) — a fast, rule-based sentiment analyzer used as a baseline model

Transformer-based Model — a pre-trained deep learning model from HuggingFace capable of contextual sentiment understanding

VADER is treated as a baseline to contextualize the performance gains achieved by transformer-based sentiment analysis.

🎯 Key Findings

Contextual Accuracy: Transformer models capture nuanced sentiment and contextual meaning more effectively than lexicon-based methods

Speed vs Accuracy: VADER offers significantly faster inference but struggles with complex or implicit sentiment

Practical Trade-offs: VADER is suitable for speed-critical, real-time applications, while transformer models are better suited for accuracy-driven analysis

🚀 Getting Started
Prerequisites
Python 3.7 or higher
pip package manager

Installation
Clone the repository and install dependencies:
git clone <repository-url>
cd Book-Review-Sentiment-Analysis
pip install -r requirements.txt

Usage
Run the analysis script:
python src/sentiment_analysis.py

The script performs the following steps:
Loads book review data from data/book_reviews_sample.csv
Cleans and preprocesses the text
Applies sentiment analysis using both VADER and transformer models
Generates comparative visualizations of sentiment distributions

📁 Project Structure
Book-Review-Sentiment-Analysis/
├── data/
│   └── book_reviews_sample.csv
├── src/
│   └── sentiment_analysis.py
├── requirements.txt
└── README.md

🛠️ Technologies Used
pandas, numpy — data manipulation and analysis
NLTK — VADER sentiment analysis
Transformers (HuggingFace) — transformer-based sentiment modeling
PyTorch — deep learning backend
Matplotlib — visualization

📈 Results
The project produces visual comparisons of sentiment classifications (positive, neutral, negative) across both models, highlighting differences in prediction behavior and model limitations.

📝 License
This project is intended for educational and research purposes.
