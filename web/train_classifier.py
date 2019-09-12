import sys

import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import re
import nltk
import xgboost as xgb
from sqlalchemy import create_engine
from nltk.corpus import stopwords as nl_stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from nltk.stem import WordNetLemmatizer
  
nltk.download('stopwords')


import dill
import pickle

#I've rearranged the functions a bit to tailor them to my implementation

def load_data(database_filepath):
    """
    Load data and split it into target and explanotary variables
    
    Args:
    database_filepath(str): path to the database
    
    Out:
    X_text(Series): a Series of messages
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('categorized_messages', con=engine)
    X_text = df['message']
    X_genre = df['genre']
    Y = df.drop(['id','message','original','genre'], axis=1)
    Y = Y.drop(['child_alone'], axis=1)
    X_text.reset_index()
    Y.reset_index()
    return X_text, Y

def tokenize(text,stopwords=None):
    """
    Function performs basic tokenization:
    1. Conversion to lowercase
    2. Removal of special characters
    3. Tokenization using NLTK
    4. Removal of stopwords
    
    Args:
    text (str): text to be tokenized
    
    Out:
    words (list): a list of tokens
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]'," ",text)
    words = nltk.word_tokenize(text)
    if stopwords:
        words = [w for w in words if w not in stopwords]
    
    return words

def get_vector_func(w2v, w_placeholder='supercollider'):
    """
    The function sets the Word2Vec model
    for the inner get_vector function
    and returns it.

    Args:
    w2v (Word2VecKeyedVectors) - a Word2Vec model
    w_placeholder (str) - the word that we'll replace
        missing words with. Doesn't matter what it is
        as long as it's rare and has nothing to do
        with natural disasters

    Out:
    try_get_vector (func) - a function that allows
    words tobe missing from the vocabulary
    """
    def try_get_vector(word):
        """
        This inner function implements exception handling
        for Word2VecKeyedVectors.get_vector
        """
        try:
            vect = w2v.get_vector(word)
        except:
            # Doesn't matter what we use for words that aren't found
            # as long as it's rare and has nothing to do with
            # natural disasters
            vect = w2v.get_vector(w_placeholder)
        return vect
    return try_get_vector


# Feature engineering: word2vec and clusterization
class w2vClusters(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters, words_mappings, 
                 random_state=42, n_jobs=1, tokenize=tokenize):
        """
        Args:
        n_clusters (int) - number of clusters to use for KMeans
        words_mappings (dict) - a dict mapping words to vectors
        random_state (float) - initialization random state for KMeans
        n_jobs (int) - number of parallel jobs for KMeans
        tokenize (func) - sentence tokenizing function
        """
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.words_mappings = words_mappings
        self.tokenize = tokenize
        self.cl_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=n_jobs)
    
    def ws_to_vs(self, words):
        def w_to_v(word):
            try:
                vec = self.words_mappings[word]
            except:
                # This is meant as a harmless joke
                # I very well know the dangers of hard-coding
                # stuff like this deep into the implementation
                vec = self.words_mappings['supercollider']
            return vec
        vecs = list(map(w_to_v,words))
        return vecs
    
    def fit(self, X, y=None):
        """
        Fits the kmeans model that is used to assign clusters
        to words
        """
        X_token = X.apply(self.tokenize)
        words = list(itertools.chain.from_iterable(X_token))
        vecs = self.ws_to_vs(words)
        self.cl_model.fit(vecs)        

        # Tweets are short, so we need a conflict resolution mechanism for
        # when we'll have just one word in a second or third most
        # frequent cluster
        #
        # We'll be prioritizing the less frequent clusters, so we need to
        # calculate the frequencies
        clusters = self.cl_model.predict(vecs)
        self.cl_counts = pd.Series(clusters).value_counts()
        
        return self
    
    def transform(self, X):
        """
        Attributes words to KMeans clusters and outputs three
        clusters with the highest frequencies in a tweet. In
        case of conflicts takes the globally less frequent tweets.

        Args:
        X - tweet series
        """
        # Clean and extract words
        def count_clusters(tweet):
            words = self.tokenize(tweet)
            
            if not words:
                return [-1000000, -1000000, -1000000]
            # Check if we got any words
            vecs = self.ws_to_vs(words)
        
            # Get clusters
            clusters = self.cl_model.predict(vecs)

            # Count words in each cluster and sort
            x_cl_counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
            #Get three most prominent clusters
            prev_count = 0
            cls = []
            curr_idces = []
            for index, item in x_cl_counts.iteritems():
                if item < prev_count:
                    # Among the most frequent local clusters pick
                    # the least frequent global clusters
                    gl_cl_counts = self.cl_counts.loc[curr_idces].sort_values(ascending=True)
                    to_add = min(3-len(cls),len(gl_cl_counts))
                    cls += list(gl_cl_counts.iloc[:to_add].index)
                    if len(cls) >= 3:
                        break
                    curr_idces = []
                curr_idces.append(index)
                prev_count = item

            # If we didn't get three clusters, add the final ones
            if len(cls) < 3:
                gl_cl_counts = self.cl_counts.loc[curr_idces].sort_values(ascending=True)
                to_add = min(3-len(cls),len(gl_cl_counts))
                cls += list(gl_cl_counts.iloc[:to_add].index)

            # If still not enough, pad with -1000000
            if len(cls) < 3:
                cls += [-1000000] * (3 - len(cls))
            return cls
        X_clusters = X.apply(count_clusters)
        # From https://stackoverflow.com/questions/35491274/pandas-split-column-of-lists-into-multiple-columns
        X_clusters = pd.DataFrame(X_clusters.values.tolist(), index=X_clusters.index, 
                                  columns=['cluster1','cluster2','cluster3'])
        return X_clusters


def build_model(words_mappings,tokenize,stopwords):
    # This pipeline incorporates the w2vClusters transformer to incorporate some
    # word meanings into the model

    # Advanced pipeline
    pipeline_advanced3 = Pipeline([
        ('feat', FeatureUnion(
            [('BoW',Pipeline(
                [('vec',CountVectorizer(tokenizer=lambda x: tokenize(x, stopwords))),
                ('tfidf',TfidfTransformer())])),
            ('cl_freqs',Pipeline([
                # I tried different n_clusters here. Between 20 and 30 works best
                # Too little, and everything ends up in one cluster
                # Too much, and every word in a tweet is in a different cluster.
                ('w2v',w2vClusters(n_clusters=27, n_jobs=3, words_mappings=words_mappings, tokenize=tokenize)),
                ('one_hot',OneHotEncoder(categories='auto',handle_unknown='ignore'))]))
            ])),
        ('clf',MultiOutputClassifier(estimator=xgb.XGBClassifier(
            random_state=42,n_estimators=200,subsample=0.8,max_depth=4,
            learning_rate=0.1,colsample_bytree=0.4,scale_pos_weight=3)))
    ])
        
    return pipeline_advanced3


def evaluate_model(model, X_test, Y_test):
    # Testing the efficiency
    Y_pred = model.predict(X_test)
    model_fscores = {}

    for ind, col in enumerate(list(Y_test.columns)):
        y_test = list(Y_test.iloc[:,ind])
        y_pred = list(Y_pred[:,ind])
        print(col)
        try:
            model_fscores[col] = f1_score(y_test, y_pred)
        except:
            model_fscores[col] = f1_score(y_test, y_pred, average='weighted')
        print('F1-score is {}'.format(advanced3_fscores[col]))  

        #print(classification_report(y_test,y_pred))


def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_text, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)
        
        stopwords = nl_stopwords.words('english')
        
        # We load the word-to-vecotr mappings that we prepared
        # This is done to avoid uploading the 3.5GB Google model into the
        # workspace

        words_mappings = pickle.load(open('words_mappings.pkl','rb'))
        
        print('Building model...')
        model = build_model(words_mappings,tokenize,stopwords)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()