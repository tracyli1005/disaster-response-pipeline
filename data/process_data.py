import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''load and merge messages and categories data for preprocessing
    Args:
        file paths of messages and categories data
    Returns:
        merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,how='outer',\
                    on=['id'])
    
    return df


def clean_data(df):
    '''clean merged data for machine learning.Split categories and assign 1 to relevant categories and 0 to irrelevant categories
    Drop duplicated records.
    Args:
        merged data of messages and categories
    Returns:
        cleaned dataframe
    '''
    # rename the columns of `categories`
    categories = df.categories.str.split(';',expand=True)    
    category_colnames=[]
    for i in range(36):
        x=categories.iloc[0,i].split('-')[0]
        category_colnames.append(x)
        
    categories.columns = category_colnames
    
    # set each value to be the last character of the string and convert to numeric
    for column in categories.columns:
        categories[column] = categories[column].astype(str).apply(lambda x:x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    # replace the original categories column with the new `categories` dataframe
    df=df.drop('categories',axis=1)
    categories['id']=df['id']
    df = df.merge(categories,how='outer',\
             on='id')
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filepath):
    '''save the cleaned data as sql file in sqlite
    Args:
        df=cleaned dataframe
        database_filepath=sqlite file path to save sql file to
    Returns:
        None
    '''
    engine = create_engine('sqlite:///../data/database_filepath')
    df.to_sql('merged', con=engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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