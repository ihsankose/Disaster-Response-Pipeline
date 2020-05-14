import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    This function loads messages and categories then merges them together. 
  
    Parameters: 
    messages_filepath (str): Messages csv file  path
    categories_filepath (str): Categories csv file  path
  
    Returns: 
    pandas.DataFrame: Merged DataFrame
    """
    # load messages dataset
	messages = pd.read_csv(messages_filepath)
	# load categories dataset
	categories = pd.read_csv(categories_filepath)
	# merge datasets
	df = pd.merge(messages,categories, on='id')
	return df


def clean_data(df):
    """ 
    This function cleans the input dataframe. 
    1. Get Category Names
    2. Get Category Values-Dummies
    3. Convert category values to integers
    4. Remove Duplicates
  
    Parameters: 
    df (pd.DataFrame): Merged DataFrame
  
    Returns: 
    pandas.DataFrame: Cleaned DataFrame
    """
	# create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(';', expand=True)
	# select the first row of the categories dataframe
	row = categories.loc[0]
	# extract a list of new column names for categories. Remove unnecessary chars.
	category_colnames = row.str.replace(r'-\w','')
	# rename the columns of `categories`
	categories.columns = category_colnames
	# Convert category values to just numbers 0 or 1.
	categories = categories.applymap(lambda x: int(x.split('-')[1]))
	# drop the original categories column from `df`
	df.drop(['categories'],axis=1, inplace=True)
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df,categories],axis=1)
	# find duplicates
	dups = df.duplicated(subset=None, keep='first')
	# drop duplicates
	df = df[~(dups)]
	return df


def save_data(df, database_filename):
    """ 
    This function saves input DataFrame to a SQL Database. 
  
    Parameters: 
    df (pd.DataFrame): Messages csv file  path
    database_filename (str): Database file name
  
    Returns: 
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # get a cursor
    #cur = engine.cursor()
    # drop the test table in case it already exists
    result = engine.execute("DROP TABLE IF EXISTS messages")
    
    df.to_sql('messages', engine, index=False)
    result.close()



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