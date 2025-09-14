# sentiment_analyzer.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from typing import List, Dict

# Global models (loaded once at startup)
sentiment_tokenizer = None
sentiment_model = None

def load_sentiment_model():
    """Loads the sentiment analysis model and tokenizer."""
    global sentiment_tokenizer, sentiment_model
    try:
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        print("✅ Sentiment model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading sentiment model: {e}")
        raise e

def get_sentiment_scores(texts: List[str]) -> List[Dict]:
    """Analyzes sentiment for a list of texts using the RoBERTa model."""
    if sentiment_model is None:
        raise RuntimeError("Sentiment model has not been loaded. Call load_sentiment_model() first.")
    
    results = []
    for text in texts:
        encoded_text = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = sentiment_model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment_label = scores.argmax()
        results.append({
            'text': text,
            'sentiment': int(sentiment_label),  # 0=negative, 1=neutral, 2=positive
            'scores': {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
        })
    return results

def separate_by_sentiment(sentiment_results: List[Dict]) -> tuple:
    """Separates reviews into positive and negative/neutral lists."""
    positive_texts = []
    negative_neutral_texts = []
    for result in sentiment_results:
        if result['sentiment'] == 2:
            positive_texts.append(result['text'])
        else:
            negative_neutral_texts.append(result['text'])
    return positive_texts, negative_neutral_texts