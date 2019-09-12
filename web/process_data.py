import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories files
    returns DataFrames
    
    CAUTION: Data should really be cleaned before merging.
    Merging dirty data and then cleaning, in my experience,
    can lead to unnecessary problems.
    
    This is why I don't merge here and do it in clean_data instead
    
    Args:
    messages_filepath(str): path to the messages file
    categories_filepath(str): path to the categories file
    
    Out:
    messages (Series) - a Series with the messages
    categories (Series) - a Series with the categories
    """
    messages = pd.read_csv(messages_filepath,sep=',',quotechar='"')
    categories = pd.read_csv(categories_filepath,sep=',',quotechar='"')
    return messages, categories


def clean_data(messages, categories):
    """
    Cleans the messages and categories
    
    Args:
    messages (Series) - a Series with the messages
    categories (Series) - a Series with the categories
    
    Out:
    messages (Series) - a Series with the messages
    categories (Series) - a Series with the categories
    """
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()
    
    categories = categories.drop_duplicates(subset=['id'],keep='first')
    
    return messages, categories


def preprocess_data(messages, categories):
    """
    Does the rest of preprocessing: splits the categories into
    separate classes and merges
    
    
    Args:
    messages (Series) - a Series with the messages
    categories (Series) - a Series with the categories
    
    Out:
    df (DataFrame): cleaned, preprocessed, and merged DataFrame
    
    """
    def get_cat_values(text):
        """
        This basic function takes an encoded category string
        and returns a list of 0s and 1s indicating 
        categories. It could be written as a lambda,
        but it would be less readable.

        Args:
        text (str): text with values encoded in them

        Output:
        val_list (list): a list of 0s and 1s flags
        """

        vals = [int(x[-1]) for x in text.split(';')]
        return vals
    
    
    # Get column names
    cols = categories.loc[0,'categories'].split(';')
    cols = [x[:-2] for x in cols]
    
    # Get data
    
    #Turn a Series of str in to a Series of lists
    data = categories['categories'].apply(get_cat_values)
    #Turn the Series into a list of lists
    data=list(data)
    #Direct cast to np.array didn't work as expected (made an array of lists)
    data=np.array(data)
    #Make the ID column
    ids = np.array(categories['id'])[np.newaxis].T
    #Add ID to the values and to the column names list
    data_full = np.hstack((data,ids))
    cols_full = cols + ['id']
    #We'll make a new categories DataFrame instead
    categories_new = pd.DataFrame(data=data_full,columns=cols_full)
    
    df = pd.merge(messages, categories_new)
    
    return df

def save_data(df, database_filename):
    """
    Saves the df DataFrame in a database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('categorized_messages', engine, index=False)


    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        messages, categories = clean_data(messages, categories)
        
        print('Finishing preprocessing...')
        df = preprocess_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()