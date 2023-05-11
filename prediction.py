from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np


def preprocess(tweet): 
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@'):
            word = '@user' 
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
        
    tweet_processed = ' '.join(tweet_words)
    return tweet_processed

def sentiment_analysis(tweet):
    roberta = 'cardiffnlp/twitter-roberta-base-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive', ]
    encoded_tweet = tokenizer(tweet,return_tensors='pt')
    output = model(**encoded_tweet)
    sentiment = output[0][0].detach().numpy()
    sentiment = softmax(sentiment)
    index = np.argmax(sentiment)
    sentiment_labelled = labels[index]
    return sentiment_labelled

