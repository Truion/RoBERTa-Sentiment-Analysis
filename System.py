import torch
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Note: The model takes a few seconds to load.")
print("Enter a piece of text and the model will predict the sentiment score.")
# print("0: Positive, 1: Negative 2: Neutral")
print("Enter 'exit' to quit.")

def preprocess_text(text):
    # Convert text to lowercase
    # print(text)
    text = text.lower()

    # Remove links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # Remove emojis
    text = re.sub(r'[\U0001f600-\U0001f650]', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text into individual words
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    processed_text = ' '.join(words)

    return processed_text


#Reload the Fine Tuned Model
from transformers import RobertaTokenizer, RobertaForSequenceClassification

#device specification
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def get_sentiment_score(text, model_path, tokenizer_path):
    try:
        # Load the pre-trained model and tokenizer
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, is_split_into_words=True)
        
        # Preprocess the text
        text = preprocess_text(text)
        
        # Tokenize the text
        encoded_input = tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass through the model
        outputs = model(**encoded_input)
        logits = outputs.logits
        
        # Get the predicted sentiment label
        _, predicted_label = torch.max(logits, dim=1)
        
        # Convert the predicted label to sentiment score
        sentiment_score = predicted_label.item()
        
        return sentiment_score
    
    except Exception as e:
        # Handle any exceptions and provide meaningful error messages
        error_msg = f"Error occurred during sentiment scoring: {str(e)}"
        raise ValueError(error_msg)


class SentimentAnalyzer:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path, is_split_into_words=True)
        except Exception as e:
            error_msg = f"Error occurred during model loading: {str(e)}"
            raise ValueError(error_msg)

    def analyze_sentiment(self, text):
        try:
            text = preprocess_text(text)
            encoded_input = self.tokenizer.encode_plus(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**encoded_input)
            logits = outputs.logits
            _, predicted_label = torch.max(logits, dim=1)
            sentiment_score = predicted_label.item()
            return sentiment_score
        except Exception as e:
            error_msg = f"Error occurred during sentiment analysis: {str(e)}"
            raise ValueError(error_msg)


model_path = "SA_roberta_model/model"
tokenizer_path = "SA_roberta_model/tokenizer"
analyzer = SentimentAnalyzer(model_path, tokenizer_path)

while True:
    text = input("Enter a piece of text (or 'exit' to quit): ")
    if text.lower() == "exit":
        break
    sentiment_score = analyzer.analyze_sentiment(text)
    if sentiment_score == 0:
        print("Sentiment Score: Positive")
    elif sentiment_score == 1:
        print("Sentiment Score: Negative")
    else:
        print("Sentiment Score: Neutral")