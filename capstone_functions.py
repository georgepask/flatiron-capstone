import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import string
from io import StringIO
import os
import regex as re
import nltk.corpus
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# HELPER FUNCTIONS

def preprocess_title(doc, stopwords=[], tokenize=False):
    '''
    Input: doc --> string
           stopwords --> list of stopwords
           tokenize - bool, do I want to tokenize the doc or not
    
    Output: transformed doc
    
    '''
    
    #Clean Data
    doc = re.sub('\W+',' ', doc)
    doc = doc.lower()
    doc = re.sub('\w*\d\w*','', doc)

    
    #Remove Stopwords
    if stopwords:
        doc = " ".join(doc for doc in doc.split() if doc not in stopwords)
        
    else:
        pass
    
    if tokenize:
        #Tokenize
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        doc = tokenizer.tokenize(doc)
        return doc
    
    else:
        return doc
        
        
def lemmatize_title(title):
    '''
    Input: a video title as a list of tokens
    
    Output: a list of lemmatized tokens
    
    '''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in title]


def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Input: model -> a classifier model 
           X_test = your testing data as pd.DataFrame
           y_tst = your testing target as pd.Series
           X_train = your training data as pd.DataFrame
           y_train = your training target as pd.Series
     
    This function does a complete evaluation of a classification model.
    It displays the confusion matrix, prints a classification report, and 
    also clearly displays the accuracy score on training and test sets.
    
    """

    # Get predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # Print classification report
    report = classification_report(y_test, y_hat_test)
    print("Classification Report: \n")
    print(classification_report(y_test, y_hat_test))

    print('\n---------------\n')
    print('Training Accuracy:', accuracy_score(y_train, y_hat_train))
    print('Testing Accuracy:', accuracy_score(y_test, y_hat_test))
    print('\n---------------\n')

    # Build and display confusion matrix
    cmatrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.BuPu)
    plt.show()

    # Print ratio for predicted values
    correct = np.sum(y_hat_test == y_test)
    incorrect = np.sum(y_hat_test != y_test)
    print(f"Correctly classified videos from test set: {correct}, {round((correct / len(y_test) * 100), 2)}%")
    print(f"Incorrectly classified videos from test set: {incorrect}, {round((incorrect / len(y_test) * 100), 2)}%")


def compare_models(X_tr, X_tst, y_tr, y_tst, models, names):
    """
    Inputs:
    X_tr = your training data as pd.DataFrame
    X_tst = your testing data as pd.DataFrame
    y_tr = your training target as pd.Series
    y_tst = your testing target as pd.Series
    models = a list of model objects you want to compare
    names = a list of strings containing the names of your modelds


    Retruns: a comparison table inclduing Accuracy on Train and Test sets for each.

    ------------------------
    """

    X_train, X_test, y_train, y_test = X_tr, X_tst, y_tr, y_tst

    accuracy_train_results = []
    accuracy_results = []

    for i in range(len(models)):
        clf = models[i].fit(X_train, y_train)

        print("Currently evaluating the {} model \n".format(names[i]))

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        accuracy_results.append(accuracy)

        accuracy2 = accuracy_score(y_train, clf.predict(X_train))
        accuracy_train_results.append(accuracy2)

    col1 = pd.DataFrame(names)
    col2 = pd.DataFrame(accuracy_train_results)
    col3 = pd.DataFrame(accuracy_results)

    results = pd.concat([col1, col2, col3], axis='columns')
    results.columns = ['Model', "Accuracy (Train)", "Accuracy (Test)"]

    return results
