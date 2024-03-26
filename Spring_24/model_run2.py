import os
from glob import glob
import pandas as pd
import numpy as np
import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class Model():
    """
    Takes in an INFO DataFrame that contains AT LEAST the following information: document titles (index) and document label (columns). For best results, use the extraction_run.py file and run create_info() and add_labels() to generate an INFO DataFrame for current use. Or for future use, save that DataFrame as a CSV file, and when needed, open the CSV as a DataFrame for this file.
    
    Performs the following operations on the INFO DataFrame:
    1. Creates a CORPUS with the following information: document titles, sentence number, token number (index) and the token str, term str, and POS tag (defined by NLTK) (columns). 
    """

    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    
    def __init__(self, test_df, train_df, overwrite=True):
        """
        Intializes INFO DataFrame. If overwrite=True, clears previous data (can start fresh with a new INFO DataFrame).
        If overwrite=False, concats old INFO DataFrame and new INFO DataFrame.
        """
        if overwrite==True:
            self.train_df = train_df
            self.test_df = test_df
        else:
            self.train_df = pd.concat([self.train_df, train_df])
            self.test_df = pd.concat([self.test_df, test_df])

    def clean_narrative(narrative, stop_words):
        # Tokenize the input string into words
        words = word_tokenize(narrative)
        # Tag the words with POS tags
        pos_tags = pos_tag(words)
        # Create a list of tuples (word, pos_tag)
        word_pos_tuples = [(word, pos_tag) for word, pos_tag in pos_tags]
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered = [(word, pos_tag) for word, pos_tag in word_pos_tuples if word.lower() not in stop_words]
        # Lemmatize words with POS tags
        lemmatizer = WordNetLemmatizer()
        lemm_words = [lemmatizer.lemmatize(word, pos=pos_tag) for word, pos_tag in filtered]
        return ' '.join(lemm_words)

    def make_XY(self, stop_words):
        self.train_df['clean_text'] = self.train_df.narrative.apply(lambda x: clean_narrative(x, stop_words))
        self.test_df['clean_text'] = self.test_df.narrative.apply(lambda x: clean_narrative(x, stop_words))

        self.train_X = self.train_df.clean_narrative
        self.test_X = self.test_df.clean_narrative
        self.train_Y = self.train_df.label
        self.test_Y = self.test_df.label
    
    def vec_engine(self, ngram_range, min_df = 2, lang='english', norm='l2', idf=True):
        # vectorize documents
        tfidf_engine = TfidfVectorizer(
            stop_words = lang,
            ngram_range = ngram_range,
            norm = norm, 
            use_idf = idf, 
            min_df = min_df)
        
        self.train_X = tfidf_engine.fit_transform(self.train_X)
        self.test_X = tfidf_engine.transform(self.test_X)
        
        tfidf_df = pd.DataFrame(self.train_X.toarray(), columns=tfidf_engine.get_feature_names_out(), index=train_X.index)
        tfidf_df['label'] = self.train_Y

        return tfidf_df.groupby('label').mean()

    def log_class_train(self):
        # Initialize a logistic regression classifier
        classifier = LogisticRegression()

        # Define hyperparameters to tune
        params = {'C': [0.1, 1, 10], 'penalty': ['l2']}

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(classifier, params, cv=5)
        grid_search.fit(self.train_X, self.train_Y)

        # Get the best classifier from grid search
        best_classifier = grid_search.best_estimator_

        # Train the best classifier on the entire training set
        best_classifier.fit(self.train_X, self.train_Y)

        # Make predictions on the test set
        y_pred = best_classifier.predict(self.test_X)

        # Evaluate the accuracy of the classifier
        accuracy = accuracy_score(self.test_Y, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='positive')
        recall = recall_score(y_test, y_pred, pos_label='positive')
        f1 = f1_score(y_test, y_pred, pos_label='positive')

        eval = {'Accuracy:':accuracy, 'Precision:':precision, 'Recall:':recall, 'F1 score:':f1}
        return pd.DataFrame(eval, orient='index', columns=['score'])