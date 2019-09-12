import json
import plotly
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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

nltk.download('stopwords')

from utils import tkn, w2vClusters, tokenize

app = Flask(__name__)

# load model
print('Unpickling')
model = joblib.load("../models/pipeline_advanced3.pkl")
print('Done unpickling')

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('categorized_messages', con=engine)
Y = df.drop(['id','message','original','genre',
            'related','request','offer','aid_related','direct_report',
            ], axis=1)
Y_full = df.drop(['id','message','original','genre','child_alone'], axis=1)

#Get service counts
service_cnts = Y.sum(axis=0).sort_values(ascending=False)
requests_and_offers = df[['request','offer']].sum(axis=0)
top10 = service_cnts.iloc[:10]
            
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top10.index,
                    y=top10.values
                )
            ],

            'layout': {
                'title': 'Top 10 Emergency Tweet Themes',
                'yaxis': {
                    'title': "Tweet Count"
                },
                'xaxis': {
                    'title': "Tweet Theme"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=requests_and_offers.index,
                    y=requests_and_offers.values
                )
            ],
        
            'layout': {
                'title': 'Requests vs. Offers',
                'yaxis': {
                    'title': "Tweet Count"
                },
                'xaxis': {
                    'title': "Tweet type"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    #query = pd.Series(['We are buried under the building. Need medical assistance. Please help.'])
    query = pd.Series([request.args.get('query', '')])
    
    cols = Y_full.columns
    
    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(cols, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query.iloc[0],
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()