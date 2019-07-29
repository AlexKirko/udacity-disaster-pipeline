import pandas as pd
import numpy as np
from dateutil.parser import parser
from sqlalchemy import create_engine

from etlfuncs import test_categories, get_cat_values

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Disaster Response ETL Pipeline',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--messages_path', type=str, default='./raw_data/messages.csv', help="""
    Path to the messages csv dataset""")
    parser.add_argument('--categories_path', type=str, default='./raw_data/categories.csv', help="""
    Path to the categories csv dataset""")
    parser.add_argument('--sql_db_path', type=str, default='./database/disaster_response.db', help="""
    Path to the SQLite database to save the results into""")
    parser.add_argument('--sql_table', type=str, default='categorized_messages', help="""
    SQL table name to save the results into""")
    args = parser.parse_args()

    messages_path = args.messages_path
    categories_path = args.categories_path
    sql_db_path = args.sql_db_path
    sql_table = args.sql_table

    #Check if the database is a db file
    #We'll be dropping it if it exists, so better make sure
    if sql_db_path[:-3] != '.db':
        raise ValueError('sql_db_path must point to an existing db file or be a path to a new one.')

    #Load the datasets
    messages = pd.read_csv(messages_path, sep=',', quotechar='"')
    categories = pd.read_csv(categories_path, sep=',', quotechar='"')

    #Remove the duplicates
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()

    #Additionally remove the duplicates from categories
    id_counts = categories.groupby(by=['id']).count()
    id_dupl = id_counts[id_counts.values > 1]

    categories = categories.drop_duplicates(subset=['id'], keep='first')

    #Split the categories into separate columns
    # Get column names
    cols = categories.loc[0, 'categories'].split(';')
    cols = [x[:-2] for x in cols]

    # Turn a Series of str in to a Series of lists
    data = categories['categories'].apply(get_cat_values)
    # Turn the Series into a list of lists
    data = list(data)
    # Direct cast to np.array didn't work as expected (made an array of lists)
    data = np.array(data)

    # Make the ID column
    ids = np.array(categories['id'])[np.newaxis].T

    # Add ID to the values and to the column names list
    data_full = np.hstack((data, ids))
    cols_full = cols + ['id']

    # We'll make a new categories DataFrame
    categories_new = pd.DataFrame(data=data_full, columns=cols_full)

    #Now we can safely merge messages with categories
    df = pd.merge(messages, categories_new)

    #Drop the database if exists
    try:
        os.remove(sql_db_path)
    except:
        pass

    engine = create_engine('sqlite://' + sql_db_path)
    df.to_sql(sql_table, engine, index=False)
