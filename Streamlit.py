import streamlit as st
import torch
from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# importing the zipfile module
from zipfile import ZipFile
  
#Reload the Fine Tuned Model
from transformers import RobertaTokenizer, RobertaForSequenceClassification

@st.cache
def get_model(model_path, tokenizer_path):

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    cloud_model_location=""
    f_checkpoint = Path("./model.zip")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
        
    # loading the temp.zip and creating a zip object
    with ZipFile("./model.zip", 'r') as zObject:
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(path="./")
    # Load the pre-trained model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, is_split_into_words=True)
    model.eval()
    return model, tokenizer

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
            self.model, self.tokenizer = get_model(self.model_path, self.tokenizer_path)
            
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


#device specification
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')



def main():
    model_path = "SA_roberta_model/model"
    tokenizer_path = "SA_roberta_model/tokenizer"
    analyzer = SentimentAnalyzer(model_path, tokenizer_path)

    st.title("Sentiment Analyzer")
    text = st.text_area("Enter a piece of text:")
    if st.button("Get Score"):
        if text:
            sentiment_score = analyzer.analyze_sentiment(text)
            if sentiment_score == 0:
                st.write("Sentiment Score: Positive")
            elif sentiment_score == 1:
                st.write("Sentiment Score: Negative")
            else:
                st.write("Sentiment Score: Neutral")

if __name__ == "__main__":
    main()