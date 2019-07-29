import pandas as pd
import numpy as np
from dateutil.parser import parser
from sqlalchemy import create_engine

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

    #Now we can safely merge
    df = pd.merge(messages, categories)

    