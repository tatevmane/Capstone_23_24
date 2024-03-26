import os
from glob import glob
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

    def view_info(self):
        return self.INFO

    def view_corpus(self):
        return self.CORPUS
    
    def make_corpus(self):
        """
        making the CORPUS
        CORPUS df: multindex = doc name/index, sent. num, token num
        columns = pos tag, token str, term str (token str normalized)
        """

        # initiate list of dictionaries
        # narratives_list = []
        # for doc_idx, txt_path in enumerate(self.INFO['txt_path']):
            # base_name = self.INFO.index[doc_idx]
            # txt_path = os.path.join(self.INFO['txt_path'][doc_idx], f"{base_name}.txt")
            
            # with open(txt_path, 'r',  encoding='utf-8') as file:
                # narrative = file.read()
            # narratives_list.append({"title": self.INFO.index[doc_idx], "narrative": narrative})

        # Convert the list of dictionaries to a DataFrame
        # narratives = pd.DataFrame(narratives_list)
        # narratives = narratives.reset_index().set_index("title")
        # narratives = narratives.drop(columns=['index'])

        narratives = self.INFO.drop(columns=['pdf_path','txt_path'])
        
        # make a new df with same index as narratives df
        df = pd.DataFrame(index=narratives.index)
        # tokenize by sentence level for narratives for each document
        df['sent_str'] = [nltk.sent_tokenize(narratives.narrative[x]) for x in range(len(narratives))]
        df = df.explode('sent_str')
        s1 = df.index.to_series()
        s2 = s1.groupby(s1).cumcount()
        df.index = [df.index, s2]
        df.index.names = ['title','sent_num']
        # tokenize by word level for narratives for each document
        # get pos tag for each tokenized word
        df['token_pos'] = [nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(df.sent_str[x])) for x in range(len(df))]
        df = df.explode('token_pos')
        s1 = df.index.to_series()
        s2 = s1.groupby(s1).cumcount()
        df.index = [df.index.get_level_values(level=0), df.index.get_level_values(level=1), s2]
        df.index.names = ['title','sent_num', 'token_num']
        df.drop(columns=['sent_str'], inplace=True)

        # format df into a CORPUS
        df['token_str'] = df.token_pos.apply(lambda x: x[0].strip())
        df['term_str'] = df.token_pos.apply(lambda x: x[0].lower().strip())
        df['pos_tag'] = df.token_pos.apply(lambda x: x[1])
        self.CORPUS = df.drop(columns="token_pos")

        return self.CORPUS

    def gather_docs(self, CORPUS, ohco_level=1, term_col='term_str'):
        OHCO = self.CORPUS.index.names
        CORPUS = self.CORPUS
        CORPUS[term_col] = CORPUS[term_col].astype('str')
        DOC = CORPUS.groupby(OHCO[:ohco_level])[term_col].apply(lambda x:' '.join(x)).to_frame('doc_str')
        DOC['n_tokens'] = DOC.doc_str.apply(lambda x: len(x.split()))
        return DOC

    def vec_engine(self, ngram_range, ohco_level=1, split=0.2, lang='english', norm='l2', idf=True):
        DOC = self.gather_docs(self.CORPUS, ohco_level=ohco_level)

        train_X, test_X, self.train_Y, self.test_Y = model_selection.train_test_split(DOC.doc_str, self.INFO.label, test_size=split)
        
        # vectorize documents
        tfidf_engine = TfidfVectorizer(
            stop_words = lang,
            ngram_range = ngram_range,
            norm = norm, 
            use_idf = idf)
        
        self.train_X = tfidf_engine.fit_transform(train_X)
        self.test_X = tfidf_engine.transform(test_X)

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
        return print("Accuracy after tuning:", accuracy)



        