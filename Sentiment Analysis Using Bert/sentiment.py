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
import pickle


class Sentiment:
    def __init__(self):
        self.text = "Sentiment Analysis"
        self.lemmatizer = WordNetLemmatizer()

        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    def sentiment(self, sent):
        tokens = self.tokenizer.encode(sent, return_tensors='pt')
        result = self.model(tokens)
        # Returns a tensor which consists of the probabilities of all the possibilities of sentiment from 1 to 3.
        '''From printing the result we can see that sentiment 1/bad sentiment has the probability of 3.1262 and is the 
        highest sentiment. Lets see how we can turn this into a use-able value'''
        '''Here, take the highest value and print the representative value by adding 1 to the position of the tensor'''
        return int(torch.argmax(result.logits))+1

    def run_sentiments(self):
        # ------------ Loading Data --------------

        with open('data.json', 'r') as json_file:
            corpus = json.load(json_file)['Sentences']

        # ----------------------------------------

        sentences = [x for para in corpus for x in sent_tokenize(para)]
        '''We are removing sentences above 512 characters since the BERT model would only take in sentences less than 512 
        characters'''
        filtered_sentences = [x for x in sentences if len(x) < 512]

        results = []
        for s in filtered_sentences:
            res = self.sentiment(s)
            results.append(res)

        sentiments = pd.DataFrame({'Sentences': filtered_sentences, 'Sentiment': results})
        sentiments.to_json('sentiment_data.json', orient='records', lines=True)

    # ----------- Model Training ----------- #

    # Data PreProcessing
    def data_cleaning(self, sent):
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
        print(f'The accuracy score of the training data is: {accuracy}')

        X_test_pred = log_model.predict(X_test)
        accuracy_scr = accuracy_score(y_test, X_test_pred)
        print(f'The accuracy of the test model is: {accuracy_scr}')

        return log_model, vectorizer, accuracy, accuracy_scr


