# Sentiment Analysis Using Bert

# Importing all the necessary dependencies
import json
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import torch


class Sentiment:
    def __init__(self):
        self.text = "Sentiment Analysis"
        self.lemmatizer = WordNetLemmatizer()

        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    def sentiment(self, sent):
        """
        This function is used to pass through a sentence and get the sentiment
        :param sent: Input Sentence
        :return: Returns the position of the highest probable value which indicates the sentiment.
        1 = Negative
        2 = Neutral
        3 = Positive
        """

        tokens = self.tokenizer.encode(sent, return_tensors='pt')
        result = self.model(tokens)
        return int(torch.argmax(result.logits))

    def run_sentiments(self):
        """
        This function is used to get the sentiment of the entire corpus
        :return: Saves the created dataframe consisting of sentences and their respective sentiments as a json file
        """
        # ------------ Loading Data --------------

        with open('data.json', 'r') as json_file:
            corpus = json.load(json_file)['Sentences']

        # ----------------------------------------

        sentences = [x for para in corpus for x in sent_tokenize(para)]
        # We are removing sentences above 512 characters since the model would only take in 512 characters
        filtered_sentences = [x for x in sentences if len(x) < 512]

        results = {}
        for s in filtered_sentences:
            res = self.sentiment(s)
            results[filtered_sentences] = res

        sentiments = pd.DataFrame(data=results, columns=['Sentence', 'Sentiment'])
        sentiments.to_json('sentiment_data.json', orient='records', lines=True)

    # ----------- Model Training ----------- #

    # Data PreProcessing
    def data_cleaning(self, sent):
        """
        This function cleans sentences with the help of regex. URLs and hashtags are removed.
        :param sent: Sentence to be cleaned
        :return: Returns cleaned sentence after Lemmatization
        """
        texts = sent.lower()
        texts = re.sub('[^a-z0-9]', ' ', texts)
        texts = re.sub(r'http\S+', '', texts)  # Remove URLs
        texts = re.sub(r'@[A-Za-z0-9]+', '', texts)  # Remove user mentions
        texts = re.sub(r'#[A-Za-z0-9]+', '', texts)  # Remove hashtags
        texts = re.sub(r'[^a-zA-Z\s]', '', texts)  # Remove non-alphabetic characters
        words = word_tokenize(texts)
        words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatization
        return ' '.join(words)

    def run_model(self):
        """
        This function is used to train a Naive Bayes Model for sentiment analysis based on the saved sentiment data
        :return: Returns the model, vectorizer used, and the accuracy of the train and test sets.
        """
        model_data = pd.read_json('sentiment_data.json', lines=True)
        model_data['stemmed_data'] = model_data['Sentences'].apply(self.data_cleaning)

        x = model_data['stemmed_data'].values
        y = model_data['Sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Converting textual data into numerical data
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Creating the Logistic Regression model
        log_model = MultinomialNB()
        log_model.fit(X_train, y_train)

        # Getting model accuracy
        X_train_pred = log_model.predict(X_train)
        accuracy = accuracy_score(y_train, X_train_pred)

        X_test_pred = log_model.predict(X_test)
        accuracy_scr = accuracy_score(y_test, X_test_pred)

        return log_model, vectorizer, accuracy, accuracy_scr
