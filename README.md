# Book Review Sentiment Analysis (VADER vs Transformers)

A comparative sentiment analysis of book reviews using lexicon-based (VADER) and transformer-based models to examine accuracy–efficiency trade-offs in sentiment classification.

## Overview

This project analyzes sentiment in book review text using two fundamentally different approaches:

- **VADER**: A fast, rule-based sentiment analyzer used as a baseline model  
- **Transformer-based model**: A pre-trained deep learning model from HuggingFace capable of contextual sentiment understanding  

VADER is treated as a baseline to contextualize the performance gains achieved by transformer-based sentiment analysis.

## Key Findings

- Transformer models capture nuanced and contextual sentiment more effectively  
- VADER provides significantly faster inference but struggles with complex or implicit sentiment  
- The choice of model depends on the trade-off between speed and accuracy  

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

```bash
git clone <repository-url>
cd "Book Review"
pip install -r requirements.txt
```

### Usage

```bash
python src/sentiment_analysis.py
```

#### What the Script Does

1. Loads book review data from `data/book_reviews_sample.csv`
2. Cleans and preprocesses the review text
3. Applies sentiment analysis using VADER and transformer models
4. Generates comparative visualizations of sentiment distributions

## Project Structure

```
Book Review/
├── data/
│   └── book_reviews_sample.csv
├── src/
│   └── sentiment_analysis.py
├── requirements.txt
└── README.md
```

## Technologies Used

- **pandas, numpy** - Data manipulation and analysis
- **NLTK** - VADER sentiment analysis
- **Transformers** - Transformer-based sentiment modeling (HuggingFace)
- **PyTorch** - Deep learning backend
- **Matplotlib** - Visualization

## Results

The project produces visual comparisons of sentiment classifications (positive, neutral, negative) across both models, highlighting differences in prediction behavior and model limitations.

## License

This project is intended for educational and research purposes.





