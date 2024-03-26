import os
from glob import glob
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Model():
    """
    Takes in an INFO DataFrame that contains AT LEAST the following information: document titles (index) and the path to the txt file (labeled as txt_path) and document label (columns). For best results, use the extraction_run.py file and run create_info() and add_labels() to generate an INFO DataFrame for current use. Or for future use, save that DataFrame as a CSV file, and when needed, open the CSV as a DataFrame for this file.
    
    Performs the following operations on the INFO DataFrame:
    1. Creates a CORPUS with the following information: document titles, sentence number, token number (index) and the token str, term str, and POS tag (defined by NLTK) (columns). 
    """

    INFO = pd.DataFrame()
    CORPUS = pd.DataFrame()
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    
    def __init__(self, INFO, overwrite=True):
        """
        Intializes INFO DataFrame. If overwrite=True, clears previous data (can start fresh with a new INFO DataFrame).
        If overwrite=False, concats old INFO DataFrame and new INFO DataFrame.
        """
        if overwrite==True:
            self.INFO = INFO
        else:
            self.INFO = pd.concat([self.INFO, INFO])

    def vec_engine(self, ngram_range, split=0.2, lang='english', norm='l2', idf=True):

        train_X, test_X, self.train_Y, self.test_Y = model_selection.train_test_split(self.INFO.narrative, self.INFO.label, test_size=split)
        
        # vectorize documents
        tfidf_engine = TfidfVectorizer(
            stop_words = lang,
            ngram_range = ngram_range,
            norm = norm, 
            use_idf = idf)
        
        self.train_X = tfidf_engine.fit_transform(train_X)
        self.test_X = tfidf_engine.transform(test_X)
        
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

        # Evaluate the accuracy, precision, recall, and f1 score of the classifier
        accuracy = accuracy_score(self.test_Y, y_pred)
        return print("Accuracy after tuning:", accuracy)

        print('Precision:', precision_score(self.test_Y, y_pred, pos_label='positive'))
        print('Recall:', recall_score(self.test_Y, y_pred, pos_label='positive'))
        print('F1 score:', f1_score(self.test_Y, y_pred, pos_label='positive'))

        


                                                                                      