# Dependencies
1. Anaconda 3.x - full installation
2. gensim package
3. xgboost package
4. dill package
5. pickle package
6. To unpickle a trined model without warnings: scikit-learn==0.21.3, pandas=0.25.1

Google's word2Vec pre-trained model is required to train pipeline_advanced3 from scratch.
It is available [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

**Caution:** it takes up more than 3GB of space unzipped. You can skip using it and 
just unpickle the pre-trained pipeline from the root folder.
# Installation
As long as you have all the dependencies installed, this will run on the standard 
distribution of Anaconda with Python 3.
# Project motivation
For this project, I use the Figure Eight disaster response data to build a classifier
that flags messages for various emergency services.
# File descriptions
* ETL Pipeline Preparation.ipynb - data parsing, preparation, and saving
* ML Pipeline Preparation.ipynb - classification (main file). DOES requrie Google's
                                pretrained word2Vec to run fully (but you can always
                                just unpickle).
* pipeline_advanced3.pkl - pre-trained classifier that can be unpickled
* ./raw_data/categories.csv - message categories data
* ./raw_data/messages.csv - messages themselves
* ./web/process_data.py - file for command-line ETL
* ./web/train_classifier.py - file for command-line classifier training. Does NOT
require Google's pre-trained word2Vec (uses cached mappings)
* ./web/run.py - file to run with flask (used in conjunction with Udacity's IDE,
won't run by itself)
* ./py_scripts/my_etl_pipeline/etlfuncs.py - my ETL helper functions
* ./py_scripts/my_etl_pipeline/etl_pipeline.py - another implementation of the
ETL pipeline

# Licensing, Authors, Acknowledgements
You are free to use the code as you like. 

Raw data and it's terms of usage can be found [here](https://www.figure-eight.com/dataset/combined-disaster-response-data/)
