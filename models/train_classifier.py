import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """ 
    This function loads database from SQL Database. 
    
  
    Parameters: 
    database_filepath (str): SQL Database file path
  
    Returns: 
    X (array): Messages
    y (array): Message Categories
    categories(array) : Category Names
    """
    # load data from database
	engine = create_engine('sqlite:///{}'.format(database_filepath))
	df = pd.read_sql_table('messages',con=engine)
	X = np.array(df['message'])
	y = np.array(df.iloc[:,4:])
	categories = df.columns[4:]
	return X, y, categories
	


def tokenize(text):
    """Standart Word Tokenizer, Input: (str) """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ 
    This function builds the classifier model. 
  
    Parameters: 
    None
  
    Returns: 
    Pipeline  : Model
    """
	pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=tokenize)),
			('tfidf', TfidfTransformer()),
			('clf', MultiOutputClassifier(RandomForestClassifier()))
			])
	parameters = {
		'vect__ngram_range': ((1, 1),(1,2))
		#'vect__max_df': (0.5, 0.75, 1.0),
		#'vect__max_features': (None, 5000, 10000),
		#'tfidf__use_idf': (True, False)
		}
    # This takes really long time to complete, comment out below lines to use gridsearchcv
	# cv = GridSearchCV(pipeline, param_grid=parameters)
	# return cv
	return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """ 
    This function evaluates the model with test data.
    Then prints out the prediction score for each category column.    
  
    Parameters: 
    model (pipeline): Previously trained model
    X_test (array): Test input 
    y_test (array): Test output
    category_names (array): category names for printing out
  
    Returns: 
    None,
    Only prints out the scores
    """
	y_hat = model.predict(X_test)
	df_y_hat=pd.DataFrame(y_hat)
	df_y_test=pd.DataFrame(y_test)
	for i in range (len(df_y_hat.columns)):
		c_hat = df_y_hat.iloc[:,i]
		c_test = df_y_test.iloc[:,i]
		print(category_names[i])
		print(classification_report(c_test, c_hat))


def save_model(model, model_filepath):
	# save the model to disk
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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