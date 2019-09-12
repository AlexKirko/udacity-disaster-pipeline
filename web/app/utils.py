# import libraries
import pandas as pd
import itertools
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from nltk.corpus import stopwords as nl_stopwords

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

nltk.download('stopwords')
stopwords = nl_stopwords.words('english')
def tkn(x):
    return tokenize(x, stopwords)