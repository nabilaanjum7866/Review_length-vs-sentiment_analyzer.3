# Review Length vs Sentiment Analyzer

This project analyzes movie reviews and predicts whether the sentiment is positive or negative.
It also studies the relationship between review length and sentiment.

## Features
- Sentiment prediction using Machine Learning
- Handles negative sentences (e.g., "not good" → negative)
- Displays review length
- Charts (Boxplot & Pie chart)
- Keyword extraction
- History tracking

## Dataset
A subset of the IMDB dataset (1000 reviews) is used for faster processing.

## How to Run
1. Install required libraries:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib